/**
 * What a smoke check sends the engine, recorded while `ENGINE_SMOKE_SENT`
 * names a folder (`cargo xtask smoke` names one per check): each op, the
 * environment sections an environment control set, and the components and
 * resources `inspect` read. The static claims of
 * web/src/generated/coverage.json say which checks name each; this says
 * which sent it. Every process of a check records alike (the agent bridge
 * it starts inherits the variable), each into its own `PID.json` as it
 * exits, and the runner merges them.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const folder = process.env.ENGINE_SMOKE_SENT;
const ops = new Set();
const sections = new Set();
const inspectors = new Set();
/** The environment control's own fields, which name no section. */
const CONTROL_FIELDS = new Set(["op", "version", "merge"]);

/** Records `request`, a development command a check sends. */
export function recordSent(request) {
  if (!folder || typeof request?.op !== "string") return;
  ops.add(request.op);
  if (request.op === "environment") {
    for (const key of Object.keys(request)) if (!CONTROL_FIELDS.has(key)) sections.add(key);
  }
  if (request.op === "inspect" && typeof request.component === "string") inspectors.add(request.component);
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
