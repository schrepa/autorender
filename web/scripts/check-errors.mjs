/** Every error code the web host and the agent tools answer with is one the
 * engine's catalog (`ErrorCode`, generated into protocol.json) describes,
 * so agents look each up in one place. */
import { readdir, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const web = fileURLToPath(new URL("../", import.meta.url));
const { errors } = JSON.parse(await readFile(`${web}src/generated/protocol.json`, "utf8"));
// A code is the first argument of an error constructor or reply helper.
// (The built-in Error takes a message, not a code.)
const code = /\b(?:\w+Error|error|failure|fail)\s*\(\s*["']([a-z]+(?:_[a-z0-9]+)*)["']/g;
const sources = [
  ...(await readdir(`${web}src/engine`)).filter(name => name.endsWith(".ts")).map(name => `src/engine/${name}`),
  ...(await readdir(`${web}scripts`)).filter(name => name.endsWith(".mjs") && !name.endsWith("-smoke.mjs") && !name.startsWith("zz-")).map(name => `scripts/${name}`),
];
const unknown = [];
for (const source of sources) {
  for (const match of (await readFile(`${web}${source}`, "utf8")).matchAll(code)) {
    if (!Object.hasOwn(errors, match[1])) unknown.push(`${source}: ${match[1]}`);
  }
}
if (unknown.length) {
  console.error(`Error codes missing from the engine's catalog (crates/engine_settings/src/codes.rs):\n${unknown.join("\n")}`);
  process.exit(1);
}
console.log(`Every error code the web side answers with is in the catalog (${Object.keys(errors).length} codes).`);
