// @ts-check
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const workspace = resolve(fileURLToPath(new URL("../../", import.meta.url)));
