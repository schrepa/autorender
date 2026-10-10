/** Every page parameter the web host and the Stress Lab read is one they
 * declare with what it does (src/engine/parameters.ts, src/stress/
 * parameters.ts), so agents find them all in describe's PageParameters. */
import { readdir, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const web = fileURLToPath(new URL("../", import.meta.url));
/** The keys of the record constant `name` in `source`. */
function declared(source, name) {
  const body = source.slice(source.indexOf(`const ${name}`));
  return new Set([...body.slice(0, body.indexOf("};")).matchAll(/^\s+(\w+):/gm)].map(match => match[1]));
}
const host = declared(await readFile(`${web}src/engine/parameters.ts`, "utf8"), "HOST_PARAMETERS");
const stress = declared(await readFile(`${web}src/stress/parameters.ts`, "utf8"), "STRESS_PARAMETERS");
// A parameter read by name, or a host switch named in its table.
const read = /(?:URLSearchParams\([^)]*\)|params|searchParams)\.(?:get|has)\(\s*["'](\w+)["']|^\s*\["(\w+)", \["/gm;
const unknown = [];
for (const [folder, names] of [["src/engine", host], ["src/stress", new Set([...host, ...stress])]]) {
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
console.log(`Every page parameter read is declared (${host.size} host, ${stress.size} Stress Lab).`);
