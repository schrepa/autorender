/** Bounded source fingerprinting for the native development supervisor.
 * Polling reads O(total input bytes), outside the browser and gameplay loop.
 * Content hashes deliberately do not trust timestamps, editor rename behavior,
 * or platform-specific recursive filesystem notifications.
 */
import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, opendir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { performance } from "node:perf_hooks";

const MAX_FILES = 20_000;
const MAX_BYTES = 128 * 1024 * 1024;
const MAX_ENTRIES = 40_000;
const MAX_DEPTH = 64;
const CHUNK_BYTES = 64 * 1024;
// The patched glam (vendor/glam) compiles into every Wasm build; development
// schemas are generated from crate sources at build time.
const TREES = ["crates", "apps", ".cargo", "vendor/glam"];
const FILES = ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"];
const IGNORED = new Set(["target", "pkg", "pkg-dev", "dist", ".git", "node_modules", ".tools"]);

class SourceWatchError extends Error {
  constructor(code, message, path = "$") {
    super(message);
    this.code = code;
    this.path = path;
  }
}

function limit(value, maximum, name) {
  if (!Number.isSafeInteger(value) || value < 0 || value > maximum) {
    throw new TypeError(`${name} must be an integer between 0 and ${maximum}.`);
  }
  return value;
}

function changed(path) {
  return new SourceWatchError("source_changed", "Source changed during fingerprinting; retry the scan.", path);
}

async function statPath(root, path, optional = false) {
  try {
    const stat = await lstat(join(root, path), { bigint: true });
    if (stat.isSymbolicLink()) {
      throw new SourceWatchError("source_symlink", "Build input symlinks are unsupported; use regular files and directories.", path);
    }
    return stat;
  } catch (error) {
    if (error?.code === "ENOENT" && optional) return null;
    if (error instanceof SourceWatchError) throw error;
    if (error?.code === "ENOENT") throw changed(path);
    throw new SourceWatchError("source_io", String(error?.message ?? error), path);
  }
}

function sameFile(left, right) {
  return left.dev === right.dev && left.ino === right.ino && left.size === right.size
    && left.mtimeNs === right.mtimeNs && left.ctimeNs === right.ctimeNs;
}

/** Hash browser build inputs, including new/deleted files and unchanged mtimes.
 * Limits can be lowered for callers/tests but cannot exceed the hard budgets.
 * Native tool sources and editable scene files are intentionally not inputs:
 * tool changes require supervisor restart; scene files use their reload API.
 */
export async function fingerprintSources(root, options = {}) {
  const maxFiles = limit(options.maxFiles ?? MAX_FILES, MAX_FILES, "maxFiles");
  const maxBytes = limit(options.maxBytes ?? MAX_BYTES, MAX_BYTES, "maxBytes");
  root = resolve(root);
  const rootStat = await statPath(root, "");
  if (!rootStat.isDirectory()) throw new SourceWatchError("source_type", "Expected a workspace directory.", "$");
  const files = [];
  let declaredBytes = 0;
  let entries = 0;

  function addFile(path, stat) {
    if (!stat.isFile()) throw new SourceWatchError("source_type", "Build inputs must be regular files.", path);
    if (files.length >= maxFiles) throw new SourceWatchError("source_budget", `Build inputs exceed ${maxFiles} files.`, path);
    if (stat.size > BigInt(maxBytes - declaredBytes)) {
      throw new SourceWatchError("source_budget", `Build inputs exceed ${maxBytes} bytes.`, path);
    }
    declaredBytes += Number(stat.size);
    files.push({ path, stat });
  }

  // Whether every ancestor directory of `path` exists, refusing symlinked or
  // non-directory ancestors.
  async function ancestorsPresent(path) {
    const parts = path.split("/");
    for (let count = 1; count < parts.length; count++) {
      const parent = parts.slice(0, count).join("/");
      const stat = await statPath(root, parent, true);
      if (!stat) return false;
      if (!stat.isDirectory()) throw new SourceWatchError("source_type", "Expected a build input directory.", parent);
    }
    return true;
  }

  for (const tree of TREES) {
    if (!(await ancestorsPresent(tree))) continue;
    const stat = await statPath(root, tree, true);
    if (!stat) continue;
    if (!stat.isDirectory()) throw new SourceWatchError("source_type", "Expected a build input directory.", tree);
    const pending = [{ path: tree, depth: 1 }];
    while (pending.length) {
      const current = pending.pop();
      // Recheck queued directories so an editor cannot silently replace one
      // with a symlink between discovery and traversal.
      const directoryStat = await statPath(root, current.path);
      if (!directoryStat.isDirectory()) throw changed(current.path);
      let directory;
      try { directory = await opendir(join(root, current.path)); }
      catch (error) { throw new SourceWatchError("source_io", String(error?.message ?? error), current.path); }
      for await (const entry of directory) {
        if (IGNORED.has(entry.name)) continue;
        const path = `${current.path}/${entry.name}`;
        if (++entries > MAX_ENTRIES) throw new SourceWatchError("source_budget", `Build input traversal exceeds ${MAX_ENTRIES} entries.`, path);
        const child = await statPath(root, path);
        if (child.isDirectory()) {
          if (current.depth >= MAX_DEPTH) throw new SourceWatchError("source_budget", `Build input traversal exceeds ${MAX_DEPTH} levels.`, path);
          pending.push({ path, depth: current.depth + 1 });
        } else addFile(path, child);
      }
    }
  }

  for (const path of FILES) {
    if (!(await ancestorsPresent(path))) continue;
    const stat = await statPath(root, path, true);
    if (stat) addFile(path, stat);
  }

  files.sort((left, right) => left.path < right.path ? -1 : left.path > right.path ? 1 : 0);
  const fingerprint = createHash("sha256").update("engine-browser-inputs-v1\n");
  const buffer = Buffer.allocUnsafe(CHUNK_BYTES);
  let actualBytes = 0;
  for (const file of files) {
    let handle;
    try {
      // O_NOFOLLOW is supported by POSIX. Windows still receives the lstat
      // checks before/after reading and the handle identity check below.
      await statPath(root, file.path);
      handle = await open(join(root, file.path), constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
      const before = await handle.stat({ bigint: true });
      if (!before.isFile() || !sameFile(file.stat, before)) throw changed(file.path);
      const content = createHash("sha256");
      let fileBytes = 0;
      for (;;) {
        // Read at most one byte beyond the remaining allowance to detect a
        // file growing during this scan without an unbounded allocation/read.
        const length = Math.min(CHUNK_BYTES, maxBytes - actualBytes + 1);
        const { bytesRead } = await handle.read(buffer, 0, length, null);
        if (!bytesRead) break;
        actualBytes += bytesRead;
        fileBytes += bytesRead;
        if (actualBytes > maxBytes) throw new SourceWatchError("source_budget", `Build inputs exceed ${maxBytes} bytes.`, file.path);
        content.update(buffer.subarray(0, bytesRead));
      }
      const after = await handle.stat({ bigint: true });
      const pathAfter = await statPath(root, file.path);
      if (!sameFile(before, after) || !sameFile(before, pathAfter)) throw changed(file.path);
      fingerprint.update(JSON.stringify([file.path, fileBytes, content.digest("hex")]) + "\n");
    } catch (error) {
      if (error instanceof SourceWatchError) throw error;
      if (error?.code === "ENOENT") throw changed(file.path);
      throw new SourceWatchError("source_io", String(error?.message ?? error), file.path);
    } finally { await handle?.close(); }
  }
  return { fingerprint: fingerprint.digest("hex"), files: files.length };
}

/** Start one serial polling scan at a time and retain only the newest candidate.
 * onChange is a notification: launch/coordinate builds outside this callback.
 * The initial successful scan notifies immediately; later fingerprints must
 * remain unchanged for debounceMs. Recovery from a published scan error also
 * notifies when source is unchanged. Ordinary concurrent-edit races retry
 * silently. Stop/abort prevents all later callbacks.
 */
export function watchSources({ root, onChange, onError = () => {}, intervalMs = 500, debounceMs = 200, signal }) {
  if (typeof onChange !== "function" || typeof onError !== "function") throw new TypeError("Source callbacks must be functions.");
  if (!Number.isFinite(intervalMs) || intervalMs < 1 || intervalMs > 60_000) throw new TypeError("intervalMs must be between 1 and 60000.");
  if (!Number.isFinite(debounceMs) || debounceMs < 0 || debounceMs > 60_000) throw new TypeError("debounceMs must be between 0 and 60000.");
  let stopped = Boolean(signal?.aborted);
  let timer;
  let last;
  let candidate;
  let candidateSince = 0;
  let recoveryPending = false;

  function report(error) {
    if (stopped) return;
    // Error reporters cannot create unhandled promise rejections themselves.
    try { Promise.resolve(onError(error)).catch(() => {}); } catch {}
  }

  function notify(snapshot, initial) {
    recoveryPending = false;
    try { Promise.resolve(onChange({ ...snapshot, initial })).catch(report); }
    catch (error) { report(error); }
  }

  async function poll() {
    if (stopped) return;
    try {
      const snapshot = await fingerprintSources(root);
      if (stopped) return;
      if (!last) {
        last = snapshot.fingerprint;
        notify(snapshot, true);
      } else if (snapshot.fingerprint === last) {
        candidate = undefined;
        if (recoveryPending) notify(snapshot, false);
      } else {
        if (candidate?.fingerprint !== snapshot.fingerprint) {
          candidate = snapshot;
          candidateSince = performance.now();
        }
        if (performance.now() - candidateSince >= debounceMs) {
          last = snapshot.fingerprint;
          candidate = undefined;
          notify(snapshot, false);
        }
      }
    } catch (error) {
      candidate = undefined;
      // Atomic editor saves and writes overlapping a scan are ordinary
      // transient conditions. Keep polling without publishing a fatal-looking
      // watcher error for them. A prior persistent error stays pending until
      // a successful observation reports recovery, even for unchanged bytes.
      if (error?.code !== "source_changed") {
        recoveryPending = true;
        report(error);
      }
    } finally {
      if (!stopped) timer = setTimeout(poll, intervalMs);
    }
  }

  function stop() {
    stopped = true;
    clearTimeout(timer);
    signal?.removeEventListener("abort", stop);
  }

  if (!stopped) {
    signal?.addEventListener("abort", stop, { once: true });
    void poll();
  }
  return stop;
}
