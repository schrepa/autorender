// @ts-check
import "./exit-codes.mjs";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { initSync, PackDirectory } from "../pkg-dev/engine_player.js";

const [path, listed] = process.argv.slice(2);
if (!path || !listed) {
  console.error("usage: node web/scripts/pack-wasm.mjs PACK RECORDS.json");
  process.exit(1);
}
initSync({ module: readFileSync(new URL("../pkg-dev/engine_player_bg.wasm", import.meta.url)) });
const pack = readFileSync(path);
const header = pack.subarray(0, 32);
const [at, stored] = PackDirectory.locate(header);
const directory = new PackDirectory(header, pack.subarray(at, at + stored));
const hash = bytes => createHash("sha256").update(bytes).digest("hex").slice(0, 16);
const records = JSON.parse(readFileSync(listed, "utf8"));
const mismatches = [];
for (const { kind, key, hash: native } of records) {
  const entry = directory.find(key);
  let wasm = "missing";
  if (entry) {
    const [offset, length] = entry;
    try {
      wasm = hash(directory.unpack(key, pack.subarray(offset, offset + length)));
    } catch (error) {
      wasm = `unreadable: ${String(error)}`;
    }
  }
  if (wasm !== native) mismatches.push({ kind, key, native, wasm });
}
console.log(JSON.stringify({ records: records.length, packed: directory.records(), mismatches: mismatches.slice(0, 64), mismatched: mismatches.length }));
process.exit(mismatches.length ? 3 : 0);
