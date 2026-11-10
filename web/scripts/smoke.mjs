// @ts-check
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { AgentError, VERSION, atMost, checkOptions, failure, options, variants } from "./agent-protocol.mjs";
import { launchBrowserSession } from "./agent-browser.mjs";
import { workspace } from "./workspace.mjs";

const NAMES = ["native", "url", "smoke", "limit-ms"];
const RELAYS = ["open", "request", "source"];
const CLOSED = { ok: false, error: { code: "session_closed", path: "$.relay" } };

/** @param {number} limitMs @param {number} deadline @param {() => Promise<any>} work */
async function within(limitMs, deadline, work) {
  const late = new AgentError("command_timeout", "$.limit_ms", { allowed: atMost(limitMs) });
  const remaining = deadline - Date.now();
  if (remaining <= 0) throw late;
  let timer;
  try {
    return await Promise.race([work(), new Promise((_, reject) => { timer = setTimeout(() => reject(late), remaining); })]);
  } finally { clearTimeout(timer); }
}

async function main() {
  const values = options(process.argv.slice(2));
  checkOptions(values, NAMES);
  for (const name of NAMES) if (!values[name]) throw new AgentError("invalid_arguments", `--${name}`, { expected: "present" });
  const limitMs = Number(values["limit-ms"]);
  if (!Number.isInteger(limitMs) || limitMs <= 0) throw new AgentError("invalid_arguments", "--limit-ms", { expected: "integer", got: values["limit-ms"] });
  const deadline = Date.now() + limitMs;
  const page = new URL(values.url);
  let opened;
  async function closeSession() {
    const opening = opened;
    opened = undefined;
    const session = await opening?.catch(() => undefined);
    await session?.close();
  }
  /** @param {any} message */
  async function answer(message) {
    if (message.relay === "open") {
      await closeSession();
      const url = new URL(page);
      if (Array.isArray(message.without) && message.without.length > 0) url.searchParams.set("without", message.without.join(","));
      opened = launchBrowserSession({ url });
      const session = await opened;
      console.log(JSON.stringify({ event: "page", url: session.url }));
      return { ok: true };
    }
    if (!RELAYS.includes(message.relay)) throw new AgentError("unknown_variant", "$.relay", { allowed: variants(RELAYS), got: message.relay });
    const session = await opened?.catch(() => undefined);
    if (!session) return CLOSED;
    return message.relay === "request" ? session.execute(message.command) : session.play(message.document);
  }
  const child = spawn(values.native, ["--smoke", values.smoke, "--limit-ms", String(limitMs), "--relay"], { cwd: workspace, stdio: ["pipe", "pipe", "inherit"] });
  const exited = new Promise(resolveExit => {
    child.once("error", () => resolveExit(null));
    child.once("close", code => resolveExit(code));
  });
  try {
    for await (const text of createInterface({ input: child.stdout })) {
      let message;
      try { message = JSON.parse(text); } catch { message = undefined; }
      if (typeof message?.relay !== "string") { console.log(text); continue; }
      const reply = await within(limitMs, deadline, () => answer(message)).catch(error => failure(error));
      child.stdin.write(`${JSON.stringify(reply)}\n`);
    }
  } finally {
    child.stdin.end();
    await closeSession();
  }
  const code = await exited;
  return typeof code === "number" ? code : 1;
}

await main().then(code => { process.exitCode = code; }, error => {
  console.log(JSON.stringify({ version: VERSION, ...failure(error) }));
  process.exitCode = 1;
});
