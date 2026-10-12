/** Bounded worker I/O and ownership of the compiler process tree. */
// @ts-check
import { spawn } from "node:child_process";

const MAX_OUTPUT = 1024 * 1024;

export async function stopProcessTree(child) {
  if (child.exitCode !== null || child.signalCode !== null || !child.pid) return;
  if (process.platform === "win32") {
    await new Promise(resolve => {
      const killer = spawn("taskkill.exe", ["/PID", String(child.pid), "/T", "/F"], { windowsHide: true, stdio: "ignore" });
      killer.once("error", resolve);
      killer.once("close", resolve);
    });
  } else {
    try { process.kill(-child.pid, "SIGKILL"); } catch (error) { if (error.code !== "ESRCH") throw error; }
  }
}

/** Uses the already compiled native tool, so broken engine Rust cannot break diagnostics. */
export function runBuild({ executable, root, output, signal, timeoutMs = 600_000 }) {
  return new Promise(resolve => {
    const child = spawn(executable, ["build-dev", "--output", output], {
      cwd: root, windowsHide: true, detached: process.platform !== "win32", stdio: ["ignore", "pipe", "pipe"],
    });
    const chunks = [];
    let bytes = 0;
    let overflow = false;
    let stderr = "";
    let cancelled;
    let settled = false;
    const cancel = reason => {
      cancelled ??= reason;
      void stopProcessTree(child);
    };
    const abort = () => cancel("build_cancelled");
    const timer = setTimeout(() => cancel("build_timeout"), timeoutMs);
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
    child.stdout.on("data", chunk => {
      if (bytes + chunk.length <= MAX_OUTPUT) { chunks.push(chunk); bytes += chunk.length; }
      else overflow = true;
    });
    child.stderr.on("data", chunk => { stderr = (stderr + chunk.toString()).slice(-16_384); });
    function finish(result) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
      resolve(result);
    }
    child.once("error", error => finish({ ok: false, diagnostics: [], error: { code: "build_worker_unavailable", path: executable, message: error.message } }));
    child.once("close", code => {
      if (cancelled) { finish({ ok: false, diagnostics: [], error: { code: cancelled, path: "$", message: "Build worker and its compiler tree were stopped." } }); return; }
      if (!overflow) {
        try {
          const result = JSON.parse(Buffer.concat(chunks).toString("utf8"));
          if (typeof result.ok === "boolean" && !(result.ok && code !== 0)) { finish(result); return; }
        } catch { /* Report malformed worker output with a bounded diagnostic tail. */ }
      }
      finish({ ok: false, diagnostics: [], error: { code: "build_worker_failed", path: "$", message: `Build worker exited ${code}; ${overflow ? "output exceeded 1 MiB" : stderr || "invalid JSON result"}` } });
    });
  });
}
