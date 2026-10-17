/** Shared preparation with immutable binaries leased for the caller's life. */
// @ts-check
import { spawn } from "node:child_process";
import { constants, setPriority } from "node:os";
import { workspace } from "./workspace.mjs";

export function cargoBinaries(cargoArguments) {
  const built = new Promise((resolveBuild, reject) => {
    const helper = process.env.ENGINE_XTASK_BIN;
    const child = spawn(helper || process.env.CARGO || "cargo", [...(helper ? [] : ["xtask"]), "prepare-binaries", "--hold", "--", ...cargoArguments],
      { cwd: workspace, env: { ...process.env, ENGINE_TOOL_WORKSPACE: workspace }, windowsHide: true, stdio: ["pipe", "pipe", "pipe"] });
    try { if (child.pid !== undefined) setPriority(child.pid, constants.priority.PRIORITY_BELOW_NORMAL); } catch { /* Normal priority remains valid. */ }
    let stdout = "", stderr = "", ready = false;
    const release = () => child.stdin.destroy();
    process.once("exit", release);
    child.stderr.on("data", bytes => { stderr = (stderr + bytes).slice(-16_384); });
    child.stdout.on("data", bytes => {
      if (ready) return;
      stdout += bytes;
      if (stdout.length > 1024 * 1024) { release(); reject(new Error("Prepared binary response exceeds 1 MiB")); return; }
      const end = stdout.indexOf("\n");
      if (end < 0) return;
      try {
        const result = JSON.parse(stdout.slice(0, end));
        if (result.version !== 1 || !result.binaries) throw new Error("Invalid prepared binary response");
        ready = true;
        // Standard child-process pipes are sockets. Keep stdin open while the
        // caller runs, without preventing its event loop from exiting.
        child.unref();
        for (const pipe of [child.stdin, child.stdout, child.stderr]) /** @type {import("node:net").Socket} */ (pipe).unref();
        resolveBuild(result.binaries);
      } catch (error) { release(); reject(error); }
    });
    child.once("error", error => { process.removeListener("exit", release); reject(error); });
    child.once("close", code => {
      process.removeListener("exit", release);
      if (!ready) reject(new Error(`Binary preparation exited ${code}: ${stderr}`));
    });
  });
  built.catch(() => {});
  return built;
}
