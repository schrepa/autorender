/** Optional tooling only: one bounded capture, artifacts instead of huge replies. */
import { mkdir, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { relative, resolve, sep } from "node:path";
import { AgentError } from "./agent-protocol.mjs";

const MAX_ARTIFACT_BYTES = 32 * 1024 * 1024;

export const profileDescriptor = {
  name: "ProfileCapture", kind: "tooling", description: "Bounded CPU/GPU/collision/memory/browser capture; artifacts saved by the local bridge.",
  schema: {
    type: "object", additionalProperties: false, required: ["op"], properties: {
      op: { const: "profile" }, duration_ms: { type: "integer", minimum: 100, maximum: 5000, default: 5000 },
      max_events: { type: "integer", minimum: 256, maximum: 65536, default: 65536 }, cpu_samples: { type: "boolean", default: true },
    },
  },
  constraints: "Requires paused profiling build. Runs ordinary frames for the capture, clears input, then pauses. Does not save source. Optional CDP sampling adds overhead. Inspect capture completeness, capabilities and dropped counts before comparing.",
};

export async function captureProfile({ page, requestBrowser, command, artifacts, workspace }) {
  if (Object.keys(command).some(key => !["op", "duration_ms", "max_events", "cpu_samples"].includes(key))) {
    throw new AgentError("invalid_request", "profile accepts duration_ms, max_events, and cpu_samples only.");
  }
  const duration = command.duration_ms === undefined ? 5000 : command.duration_ms;
  const maxEvents = command.max_events === undefined ? 65536 : command.max_events;
  if (!Number.isInteger(duration) || duration < 100 || duration > 5000 ||
      !Number.isInteger(maxEvents) || maxEvents < 256 || maxEvents > 65536 ||
      (command.cpu_samples !== undefined && typeof command.cpu_samples !== "boolean")) {
    throw new AgentError("invalid_request", "duration_ms must be 100..5000, max_events 256..65536, and cpu_samples boolean.");
  }
  const before = await requestBrowser({ op: "status" });
  if (!before.ok) return before;
  if (before.result?.paused !== true) throw new AgentError("pause_required", "Pause before starting a controlled profiling capture.");
  const available = await page.evaluate(() => typeof window.__engineProfile?.start === "function");
  if (!available) throw new AgentError("profiling_unavailable", "Use cargo xtask profile and the profiling build's stress.html.");
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
    const resumedReply = await requestBrowser({ op: "resume" });
    if (!resumedReply.ok) return resumedReply;
    resumed = true;
    await page.evaluate(options => window.__engineProfile.start(options), { duration_ms: duration, max_events: maxEvents });
    await page.waitForFunction(() => !window.__engineProfile.status().active, null, { timeout: 10_000 });
    // stop() also awaits a capture which is currently draining GPU readbacks.
    await page.evaluate(() => window.__engineProfile.stop());
    if (sampling) { stoppedProfile = (await browser.send("Profiler.stop")).profile; sampling = false; }
    const pausedReply = await requestBrowser({ op: "pause" });
    if (!pausedReply.ok) return pausedReply;
    resumed = false;
    const serialized = await page.evaluate(limit => {
      const json = JSON.stringify(window.__engineProfile.export());
      if (new TextEncoder().encode(json).byteLength > limit) throw new Error("Profile exceeds artifact limit.");
      return json;
    }, MAX_ARTIFACT_BYTES);
    report = JSON.parse(serialized);
    report.agent = {
      browser_version: page.context().browser()?.version() ?? null,
      cpu_sampling: cpuSamples, sampling_interval_us: cpuSamples ? 1000 : null,
      heap_start: memoryStart, heap_end: await browser.send("Runtime.getHeapUsage").catch(() => null),
      observation_start: before.observation, observation_end: pausedReply.observation,
      scope: "CDP heap values describe the V8 isolate, not process/GPU memory. CPU sampling adds overhead. Separate .cpuprofile uses the browser profiler clock.",
    };
    await mkdir(artifacts, { recursive: true });
    const stem = `profile-${randomUUID()}`;
    const save = async (name, value) => {
      const json = JSON.stringify(value);
      if (Buffer.byteLength(json) > MAX_ARTIFACT_BYTES) throw new AgentError("profile_too_large", "Profile artifact exceeds 32 MiB.");
      const path = resolve(artifacts, name);
      await writeFile(path, json, { flag: "wx" });
      return relative(workspace, path).split(sep).join("/");
    };
    const trace = await save(`${stem}.json`, report);
    const sampled = stoppedProfile ? await save(`${stem}.cpuprofile`, stoppedProfile) : null;
    return { ok: true, observation: pausedReply.observation, result: {
      temporary: true, paused: true, complete: report.complete ?? false, reason: report.reason ?? null,
      artifacts: { trace, cpu_profile: sampled },
      note: "Trace contains raw bounded measurements and capability gaps. CPU samples are separate and include profiler overhead.",
    } };
  } finally {
    if (sampling) await browser.send("Profiler.stop").catch(() => {});
    // Abort through the same browser API before restoring the pause boundary.
    if (resumed) {
      await page.evaluate(() => window.__engineProfile?.stop()).catch(() => {});
      await requestBrowser({ op: "pause" }).catch(() => {});
    }
    await browser.detach().catch(() => {});
  }
}
