import { existsSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

const FOLDER = "highlands/rocks";
const folder = resolve(import.meta.dirname, "../../public", FOLDER);
const rocks = existsSync(folder)
  ? readdirSync(folder, { recursive: true, encoding: "utf8" })
    .map(path => `${FOLDER}/${path.replaceAll("\\", "/")}`).filter(path => path.endsWith(".ktx2")).sort()
  : [];

export default [{
  name: "highlands-rocks",
  config: () => ({ define: { __HIGHLANDS_ROCKS__: JSON.stringify(rocks) } }),
}];
