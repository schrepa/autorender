/**
 * The workspace scripts work in: the checkout the development server
 * serves, whose programs they build and run (the native host, tools,
 * `cargo xtask`, the agent bridge) and whose files they write (fixture
 * scenes, captures). `ENGINE_WORKSPACE` names it when the scripts come from
 * another commit than the served one (`cargo xtask checks` runs one commit's
 * checks against another's server; `cargo xtask smoke` names its own); by
 * default it is the checkout holding these scripts. What the scripts read
 * as their own inputs (their fixtures, the protocol they speak) stays
 * beside them.
 */
// @ts-check
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

/** The workspace's folder. */
export const workspace = resolve(process.env.ENGINE_WORKSPACE ?? fileURLToPath(new URL("../../", import.meta.url)));
if (!existsSync(join(workspace, "Cargo.toml"))) {
  throw new Error(`ENGINE_WORKSPACE ${workspace} is not a checkout of the engine: it has no Cargo.toml`);
}

/** The path of `parts` (relative, `/`-separated or not) in the workspace. */
export function workspacePath(...parts) {
  return join(workspace, ...parts);
}
