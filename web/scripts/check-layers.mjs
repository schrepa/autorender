/** The web host's layers: `src/engine/` is the engine's browser host and
 * imports only itself, `src/generated/`, the engine's worker bindings and
 * packages, never an app such as `src/stress/`. */
import { readdir, readFile } from "node:fs/promises";
import { dirname, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const web = fileURLToPath(new URL("../", import.meta.url));
const engine = resolve(web, "src/engine");
const allowed = [engine, resolve(web, "src/generated"), resolve(web, "pkg/engine_worker"), resolve(web, "pkg/engine_worker_bg.wasm")];
const specifier = /(?:from|import)\s*\(?\s*["'](\.{1,2}\/[^"']+)["']|new URL\(\s*["'](\.{1,2}\/[^"']+)["']/g;
const violations = [];
for (const name of await readdir(engine)) {
  if (!/\.(ts|css)$/.test(name)) continue;
  const path = resolve(engine, name);
  for (const match of (await readFile(path, "utf8")).matchAll(specifier)) {
    // Vite queries such as `?url` name the same file.
    const target = resolve(dirname(path), (match[1] ?? match[2]).replace(/\?.*$/, ""));
    if (!allowed.some(root => target === root || target.startsWith(root + sep) || target.startsWith(root + "."))) {
      violations.push(`${relative(web, path)} imports ${match[1] ?? match[2]}`);
    }
  }
}
if (violations.length) {
  console.error(`src/engine/ must not import apps or other folders:\n${violations.join("\n")}`);
  process.exit(1);
}
console.log("src/engine/ imports only the engine's own modules.");
