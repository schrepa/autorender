/**
 * The machine's "one heavy run at a time" lock: anything that opens a
 * browser, a native engine window or a GPU host (smokes, measurements,
 * equivalence runs, captures) holds it while it runs, so runs never share
 * the GPU and processors with each other.
 *
 * The lock is the folder `ENGINE_HEAVY_LOCK_DIR`, else
 * `engine-heavy-run.lock` in the system's temporary folder, holding
 * `owner.json` ({pid, label, cwd, command, started}); `cargo xtask` takes
 * the same one (tools/xtask/src/heavy.rs). A lock whose owner process ended
 * is stale and taken over. Those waiting queue in the folder beside it,
 * `engine-heavy-run.lock.queue`, each as `MILLISECONDS-PID.json`, and take
 * it in the order they came: a run that releases it between steps
 * (equivalence, cell by cell) and wants it again queues behind them. After
 * taking it, a holder waits up to 15 minutes
 * while a headless Chrome of another program runs (external programs may
 * not take the lock). A holder's children run with `ENGINE_HEAVY_LOCK_HELD=1`
 * and pass straight through, so runs nest.
 *
 * Smokes hold it for their process's life (`holdHeavyRun`; the harness
 * does). Run as a program, it holds the lock around a command:
 * `node web/scripts/heavy-run.mjs [--label TEXT] -- COMMAND ARGS...`, exiting
 * with the command's exit code.
 */
// @ts-check
import { execFileSync, spawn } from "node:child_process";
import { mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";

/** The environment variable a holder's children see. */
export const HELD = "ENGINE_HEAVY_LOCK_HELD";

const lock = process.env.ENGINE_HEAVY_LOCK_DIR ?? join(tmpdir(), "engine-heavy-run.lock");
const owner = join(lock, "owner.json");
const queue = `${lock}.queue`;
/** How long a holder waits for another program's headless Chrome. */
const FOREIGN_CHROME_WAIT_MS = 15 * 60_000;

/** Whether process `pid` runs. */
function alive(pid) {
  try { process.kill(pid, 0); return true; } catch (error) { return error.code === "EPERM"; }
}

/** The process ids of headless Chrome browsers running now (Windows). */
function headlessChromes() {
  if (process.platform !== "win32") return [];
  try {
    const out = execFileSync("powershell.exe", ["-NoProfile", "-Command",
      "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | Where-Object { $_.CommandLine -match '--headless' -and $_.CommandLine -notmatch '--type=' } | ForEach-Object { $_.ProcessId }"],
    { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
    return out.split(/\s+/).filter(Boolean).map(Number);
  } catch { return []; }
}

/** The lock's owner as its file says, or null while it is written or gone. */
function heldBy() {
  try { return JSON.parse(readFileSync(owner, "utf8")); } catch { return null; }
}

/** The tickets in the queue in the order they came, removing those of
 * processes that ended. */
function queued() {
  let names;
  try { names = readdirSync(queue); } catch { return []; }
  const line = [];
  for (const name of names) {
    const pid = Number(/^\d+-(\d+)\.json$/.exec(name)?.[1]);
    if (!pid) continue;
    if (pid !== process.pid && !alive(pid)) { rmSync(join(queue, name), { force: true }); continue; }
    line.push(name);
  }
  return line.sort();
}

let holding = false;
/** This process's ticket in the queue while it waits.
 * @type {string | null} */
let ticket = null;
/** Leaves the queue. */
function leave() {
  if (ticket) rmSync(join(queue, ticket), { force: true });
  ticket = null;
}

/** Removes the lock where this process holds it. */
function release() {
  if (!holding) return;
  holding = false;
  if (heldBy()?.pid === process.pid) rmSync(lock, { recursive: true, force: true });
}

/** Takes the lock for this process as `label` running `command`, waiting
 * while another holds it, then while another program's headless Chrome
 * runs; it is released when the process exits. Where a parent holds it
 * (`ENGINE_HEAVY_LOCK_HELD=1`) this does nothing. Afterwards children this
 * process starts inherit `ENGINE_HEAVY_LOCK_HELD=1`. */
export async function holdHeavyRun(label, command = process.argv.slice(1)) {
  if (process.env[HELD] === "1" || holding) return;
  let reported = 0;
  process.on("exit", leave);
  for (;;) {
    // Its turn: first in the queue, or nobody queued before it came.
    const line = queued();
    const turn = ticket ? line[0] === ticket : line.length === 0;
    try {
      if (!turn) throw Object.assign(new Error("not its turn"), { code: "EEXIST" });
      mkdirSync(lock);
      writeFileSync(owner, JSON.stringify({ pid: process.pid, label, cwd: process.cwd(), command, started: new Date().toISOString() }, null, 1));
      break;
    } catch (error) {
      if (error.code !== "EEXIST") throw error;
    }
    const held = heldBy();
    if (held && !alive(held.pid)) {
      // Taken over only while the same stale owner still holds it.
      if (heldBy()?.pid === held.pid) {
        console.error(`heavy-run: taking over a stale lock from pid ${held.pid} (${held.label || held.command?.join(" ")})`);
        rmSync(lock, { recursive: true, force: true });
      }
      continue;
    }
    if (!held) {
      // Owner file missing for long: a crashed acquirer between mkdir and write.
      const age = Date.now() - (statSync(lock, { throwIfNoEntry: false })?.mtimeMs ?? Date.now());
      if (age > 60_000) { rmSync(lock, { recursive: true, force: true }); continue; }
    }
    if (!ticket) {
      mkdirSync(queue, { recursive: true });
      ticket = `${String(Date.now()).padStart(13, "0")}-${process.pid}.json`;
      writeFileSync(join(queue, ticket), JSON.stringify({ pid: process.pid, label }));
    }
    if (Date.now() - reported > 60_000) {
      reported = Date.now();
      if (held) console.error(`heavy-run: waiting for the heavy-run lock held by pid ${held.pid} (${held.label || held.command?.join(" ")}) since ${held.started}`);
      else console.error(`heavy-run: waiting for its turn after ${queued().indexOf(ticket)} queued before it`);
    }
    await delay(3000);
  }
  leave();
  holding = true;
  process.env[HELD] = "1";
  process.on("exit", release);
  for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) process.on(signal, () => { release(); process.exit(130); });
  const started = Date.now();
  for (let reportedChrome = 0; ;) {
    const foreign = headlessChromes();
    if (!foreign.length) break;
    if (Date.now() - started > FOREIGN_CHROME_WAIT_MS) {
      console.error(`heavy-run: foreign headless Chrome ${foreign.join(",")} still running after 15 min; going ahead`);
      break;
    }
    if (Date.now() - reportedChrome > 60_000) {
      reportedChrome = Date.now();
      console.error(`heavy-run: waiting for foreign headless Chrome ${foreign.join(",")} (factory?)`);
    }
    await delay(5000);
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  let label = "";
  while (args.length && args[0] !== "--") {
    const option = args.shift();
    if (option === "--label") label = args.shift() ?? "";
    else { console.error(`heavy-run: unknown option ${option}`); process.exit(1); }
  }
  if (args[0] !== "--" || args.length < 2) {
    console.error("usage: node web/scripts/heavy-run.mjs [--label TEXT] -- COMMAND ARGS...");
    process.exit(1);
  }
  const command = args.slice(1);
  await holdHeavyRun(label, command);
  const child = spawn(command[0], command.slice(1), { stdio: "inherit" });
  child.on("error", error => { console.error(`heavy-run: cannot run ${command[0]}: ${error.message}`); process.exit(1); });
  child.on("exit", (code, signal) => process.exit(code ?? (signal ? 1 : 0)));
}
