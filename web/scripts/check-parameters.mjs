// @ts-check
import { readdir, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const web = fileURLToPath(new URL("../", import.meta.url));
function declared(source, name) {
  const body = source.slice(source.indexOf(`const ${name}`));
  return new Set([...body.slice(0, body.indexOf("};")).matchAll(/^\s+(\w+):/gm)].map(match => match[1]));
}
const host = declared(await readFile(`${web}src/engine/parameters.ts`, "utf8"), "HOST_PARAMETERS");
const read = /(?:URLSearchParams\([^)]*\)|params|searchParams)\.(?:get|has)\(\s*["'](\w+)["']|^\s*\["(\w+)", \["/gm;
const unknown = [];
let own = 0;
const folders = (await readdir(`${web}src`, { withFileTypes: true }))
  .filter(entry => entry.isDirectory() && entry.name !== "generated").map(entry => `src/${entry.name}`);
for (const folder of folders) {
  const names = new Set(host);
  if (folder !== "src/engine") {
    const source = await readFile(`${web}${folder}/parameters.ts`, "utf8").catch(() => "");
    for (const [, list] of source.matchAll(/const (\w+_PARAMETERS)\b/g)) {
      for (const name of declared(source, list)) { names.add(name); own++; }
    }
  }
  for (const file of (await readdir(`${web}${folder}`)).filter(name => name.endsWith(".ts"))) {
    for (const match of (await readFile(`${web}${folder}/${file}`, "utf8")).matchAll(read)) {
      const name = match[1] ?? match[2];
      if (!names.has(name)) unknown.push(`${folder}/${file}: ${name}`);
    }
  }
}
if (unknown.length) {
  console.error(`Page parameters read but not declared with what they do:\n${unknown.join("\n")}`);
  process.exit(1);
}
console.log(`Every page parameter read is declared (${host.size} host, ${own} of the games).`);
