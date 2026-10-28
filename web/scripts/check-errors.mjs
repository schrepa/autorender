// @ts-check
import { readdir, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const web = fileURLToPath(new URL("../", import.meta.url));
const root = fileURLToPath(new URL("../../", import.meta.url));
const { errors } = JSON.parse(await readFile(`${web}src/generated/protocol.json`, "utf8"));
const code = String.raw`["']([a-z]+(?:_[a-z0-9]+)*)["']`;
const made = [String.raw`\b(?:\w+Error|error|failure|fail)\s*\(\s*${code}`, String.raw`\bcode\s*(?::|\?\?)\s*${code}`];
const expected = [String.raw`\bcode\s*(?:===|!==|,)\s*${code}`];
const written = [String.raw`"code"\s*:\s*${code}`];

async function files(folder, suffix) {
  const names = await readdir(`${root}${folder}`, { recursive: true });
  return names.filter(name => name.endsWith(suffix)).map(name => `${folder}/${name.replaceAll("\\", "/")}`);
}

const scripts = (await readdir(`${web}scripts`)).filter(name => name.endsWith(".mjs") && !name.startsWith("zz-"));
const checks = [
  [made, (await files("web/src", ".ts")).filter(path => !path.startsWith("web/src/generated/"))],
  [made, scripts.filter(name => !name.endsWith("-smoke.mjs")).map(name => `web/scripts/${name}`)],
  [expected, scripts.filter(name => name.endsWith("-smoke.mjs")).map(name => `web/scripts/${name}`)],
  [written, [...await files("crates", ".rs"), ...await files("apps", ".rs")]],
];
const unknown = [];
let read = 0;
for (const [patterns, paths] of checks) {
  for (const path of paths) {
    const text = await readFile(`${root}${path}`, "utf8");
    read++;
    for (const pattern of patterns) {
      for (const match of text.matchAll(new RegExp(pattern, "g"))) {
        if (!errors.includes(match[1])) unknown.push(`${path}: ${match[1]}`);
      }
    }
  }
}
if (unknown.length) {
  console.error(`Error codes missing from the engine's catalog (crates/engine_settings/src/codes.rs):\n${unknown.join("\n")}`);
  process.exit(1);
}
console.log(`Every error code ${read} files name is in the catalog (${errors.length} codes).`);
