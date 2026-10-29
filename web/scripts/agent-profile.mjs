// @ts-check
import { readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { relative, resolve, sep } from "node:path";
import { AgentError, atMost, range, variants } from "./agent-protocol.mjs";

const MAX_ARTIFACT_BYTES = 32 * 1024 * 1024;
const protocol = JSON.parse(readFileSync(new URL("../src/generated/protocol.json", import.meta.url), "utf8"));
const LIMITS = {
  duration_ms: { minimum: 100, maximum: 5000, default: 5000 },
  max_events: { minimum: 256, maximum: protocol.profile_max_events, default: protocol.profile_max_events },
};

export const profileDescriptor = {
  name: "ProfileCapture", kind: "tooling",
  schema: {
    type: "object", additionalProperties: false, properties: {
      deep: {
        type: "object", additionalProperties: false, properties: {
          duration_ms: { type: "integer", ...LIMITS.duration_ms },
          max_events: { type: "integer", ...LIMITS.max_events }, cpu_samples: { type: "boolean", default: true },
        },
      },
    },
  },
};

export const nativeProfileDescriptor = {
  ...profileDescriptor,
  schema: {
    ...profileDescriptor.schema,
    properties: {
      deep: {
        ...profileDescriptor.schema.properties.deep,
        properties: { ...profileDescriptor.schema.properties.deep.properties, vendor: { enum: ["pix"] } },
      },
    },
  },
};

export async function captureProfile({ page, requestBrowser, command, artifacts, workspace }) {
  if (!command || typeof command !== "object" || Array.isArray(command)) throw new AgentError("invalid_type", "$.deep", { expected: "object" });
  const fields = ["duration_ms", "max_events", "cpu_samples"];
  const unknown = Object.keys(command).find(key => !fields.includes(key));
  if (unknown) throw new AgentError("unknown_field", `$.deep.${unknown}`, { allowed: variants(fields), got: unknown });
  const within = (value, { minimum, maximum }) => Number.isInteger(value) && value >= minimum && value <= maximum;
  const duration = command.duration_ms === undefined ? LIMITS.duration_ms.default : command.duration_ms;
  const maxEvents = command.max_events === undefined ? LIMITS.max_events.default : command.max_events;
  for (const [name, value] of [["duration_ms", duration], ["max_events", maxEvents]]) {
    if (!within(value, LIMITS[name])) throw new AgentError("out_of_range", `$.deep.${name}`, { allowed: range(LIMITS[name].minimum, LIMITS[name].maximum), got: value });
  }
  if (command.cpu_samples !== undefined && typeof command.cpu_samples !== "boolean") {
    throw new AgentError("invalid_type", "$.deep.cpu_samples", { expected: "boolean", got: command.cpu_samples });
  }
  const before = await requestBrowser({ verb: "query", name: "status" });
  if (!before.ok) return before;
  if (before.result?.paused !== true) throw new AgentError("pause_required", "$", { expected: "paused" });
  const available = await page.evaluate(() => typeof window.__engineProfile?.start === "function");
  if (!available) throw new AgentError("profiling_unavailable", "$", { expected: "supported" });
  const browser = await page.context().newCDPSession(page);
  let resumed = false, sampling = false, stoppedProfile, report;
  const cpuSamples = command.cpu_samples ?? true;
  try {
    const memoryStart = await browser.send("Runtime.getHeapUsage").catch(() => null);
    if (cpuSamples) {
      await browser.send("Profiler.enable");
      await browser.send("Profiler.setSamplingInterval", { interval: 1000 });
      await browser.send("Profiler.start");
      sampling = true;
    }
    const resumedReply = await requestBrowser({ verb: "control", name: "resume" });
    if (!resumedReply.ok) return resumedReply;
    resumed = true;
    await page.evaluate(options => window.__engineProfile.start(options), { duration_ms: duration, max_events: maxEvents });
    await page.waitForFunction(() => !window.__engineProfile.status().active, null, { timeout: 10_000 });
    await page.evaluate(() => window.__engineProfile.stop());
    if (sampling) { stoppedProfile = (await browser.send("Profiler.stop")).profile; sampling = false; }
    const pausedReply = await requestBrowser({ verb: "control", name: "pause" });
    if (!pausedReply.ok) return pausedReply;
    resumed = false;
    const serialized = await page.evaluate(limit => {
      const json = JSON.stringify(window.__engineProfile.export());
      return new TextEncoder().encode(json).byteLength > limit ? null : json;
    }, MAX_ARTIFACT_BYTES);
    if (serialized === null) throw new AgentError("profile_too_large", "$", { allowed: atMost(MAX_ARTIFACT_BYTES) });
    report = JSON.parse(serialized);
    report.agent = {
      browser_version: page.context().browser()?.version() ?? null,
      cpu_sampling: cpuSamples, sampling_interval_us: cpuSamples ? 1000 : null,
      heap_start: memoryStart, heap_end: await browser.send("Runtime.getHeapUsage").catch(() => null),
      observation_start: before.observation, observation_end: pausedReply.observation,
    };
    await mkdir(artifacts, { recursive: true });
    const stem = `profile-${randomUUID()}`;
    const save = async (name, value) => {
      const json = JSON.stringify(value);
      if (Buffer.byteLength(json) > MAX_ARTIFACT_BYTES) throw new AgentError("profile_too_large", "$", { allowed: atMost(MAX_ARTIFACT_BYTES), got: Buffer.byteLength(json) });
      const path = resolve(artifacts, name);
      await writeFile(path, json, { flag: "wx" });
      return relative(workspace, path).split(sep).join("/");
    };
    const trace = await save(`${stem}.json`, report);
    const sampled = stoppedProfile ? await save(`${stem}.cpuprofile`, stoppedProfile) : null;
    return { ok: true, observation: pausedReply.observation, result: {
      temporary: true, paused: true, complete: report.complete ?? false, reason: report.reason ?? null,
      artifacts: { trace, cpu_profile: sampled },
    } };
  } finally {
    if (sampling) await browser.send("Profiler.stop").catch(() => {});
    if (resumed) {
      await page.evaluate(() => window.__engineProfile?.stop()).catch(() => {});
      await requestBrowser({ verb: "control", name: "pause" }).catch(() => {});
    }
    await browser.detach().catch(() => {});
  }
}
