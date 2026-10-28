// @ts-check
import { execFileSync, spawn } from "node:child_process";
import { mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";

export const HELD = "ENGINE_HEAVY_LOCK_HELD";

const lock = process.env.ENGINE_HEAVY_LOCK_DIR ?? join(tmpdir(), "engine-heavy-run.lock");
const owner = join(lock, "owner.json");
const queue = `${lock}.queue`;
const FOREIGN_CHROME_WAIT_MS = 15 * 60_000;

function alive(pid) {
  try { process.kill(pid, 0); return true; } catch (error) { return error.code === "EPERM"; }
}

function headlessChromes() {
  if (process.platform !== "win32") return [];
  try {
    const out = execFileSync("powershell.exe", ["-NoProfile", "-Command",
      "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | Where-Object { $_.CommandLine -match '--headless' -and $_.CommandLine -notmatch '--type=' } | ForEach-Object { $_.ProcessId }"],
    { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
    return out.split(/\s+/).filter(Boolean).map(Number);
  } catch { return []; }
}

function heldBy() {
  try { return JSON.parse(readFileSync(owner, "utf8")); } catch { return null; }
}

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
/** @type {string | null} */
let ticket = null;
function leave() {
  if (ticket) rmSync(join(queue, ticket), { force: true });
  ticket = null;
}

function release() {
  if (!holding) return;
  holding = false;
  if (heldBy()?.pid === process.pid) rmSync(lock, { recursive: true, force: true });
}

export async function holdHeavyRun(label, command = process.argv.slice(1)) {
  if (process.env[HELD] === "1" || holding) return;
  let reported = 0;
  process.on("exit", leave);
  for (;;) {
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
      if (heldBy()?.pid === held.pid) {
        console.error(`heavy-run: taking over a stale lock from pid ${held.pid} (${held.label || held.command?.join(" ")})`);
        rmSync(lock, { recursive: true, force: true });
      }
      continue;
    }
    if (!held) {
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
