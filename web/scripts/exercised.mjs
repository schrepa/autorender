// @ts-check
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const folder = process.env.ENGINE_SMOKE_SENT;
const ops = new Set();
const sections = new Set();
const inspectors = new Set();

export function recordSent(request) {
  if (!folder || typeof request?.verb !== "string") return;
  const named = ["query", "control"].includes(request.verb);
  if (named && typeof request.name !== "string") return;
  ops.add(named ? request.name : request.verb);
  if (request.verb === "control" && request.name === "environment") {
    for (const key of Object.keys(request.args ?? {})) if (key !== "merge") sections.add(key);
  }
  if (request.verb === "inspect" && typeof request.component === "string") inspectors.add(request.component);
  if (request.verb === "watch") {
    for (const watched of request.entities ?? []) if (typeof watched?.component === "string") inspectors.add(watched.component);
  }
}

const disabled = new Set();
let checking = false;
let expected = false;

export function failOnDisabledFeatures() {
  if (checking) return;
  checking = true;
  process.on("exit", code => {
    if (disabled.size === 0) return;
    console.error(`The renderer disabled ${[...disabled].join(", ")} during the check: a GPU error named it (disabled_features).`);
    if (code === 0) process.exitCode = 3;
  });
}

export function expectDisabledFeatures() {
  expected = true;
}

export function recordReceived(reply) {
  const names = reply?.result?.rendering?.disabled_features;
  if (!checking || expected || !Array.isArray(names)) return;
  for (const name of names) disabled.add(String(name));
}

if (folder) {
  process.on("exit", () => {
    if (ops.size === 0) return;
    mkdirSync(folder, { recursive: true });
    const sorted = set => [...set].sort();
    writeFileSync(join(folder, `${process.pid}.json`),
      JSON.stringify({ ops: sorted(ops), sections: sorted(sections), inspectors: sorted(inspectors) }));
  });
}
