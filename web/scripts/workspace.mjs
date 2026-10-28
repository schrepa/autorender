// @ts-check
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const workspace = resolve(process.env.ENGINE_WORKSPACE ?? fileURLToPath(new URL("../../", import.meta.url)));
if (!existsSync(join(workspace, "Cargo.toml"))) {
  throw new Error(`ENGINE_WORKSPACE ${workspace} is not a checkout of the engine: it has no Cargo.toml`);
}

export function workspacePath(...parts) {
  return join(workspace, ...parts);
}
