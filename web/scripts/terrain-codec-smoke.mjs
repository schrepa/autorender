/** Exact worker-wire fixtures, executed by native Rust and the same Rust
 * compiled to Wasm. Builds only the standalone diagnostic, never a game.
 * No JavaScript float crosses either interface: inputs and outputs are u32.
 */
import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { once } from "node:events";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { artifacts } from "./harness.mjs";

const root = fileURLToPath(new URL("../../", import.meta.url));
const output = await artifacts("terrain-codec-smoke");
const generated = join(root, "target/terrain-codec-smoke");
const goldenPath = join(root, "tools/terrain_codec/zfp-golden.json");
const report = { source: null, golden_blocks: 0, fixtures: [], malformed: [], production: null, native_codec_context: [] };
let native;
try {
  await mkdir(generated, { recursive: true });
  const command = (program, args) => execFileSync(program, args, { cwd: root, stdio: "inherit" });
  command("cargo", ["build", "-p", "engine_terrain_codec", "--bin", "terrain_codec_fixture", "--locked"]);
  command("cargo", ["build", "-p", "engine_terrain_codec", "--lib", "--target", "wasm32-unknown-unknown", "--locked"]);
  command("wasm-bindgen", [join(root, "target/wasm32-unknown-unknown/debug/engine_terrain_codec.wasm"),
    "--target", "nodejs", "--out-dir", generated, "--out-name", "terrain_codec"]);
  const wasm = createRequire(import.meta.url)(join(generated, "terrain_codec.js"));
  native = spawn(join(root, "target/debug/terrain_codec_fixture") + (process.platform === "win32" ? ".exe" : ""), [],
    { cwd: root, stdio: ["pipe", "pipe", "inherit"], windowsHide: true });
  const waiting = [];
  createInterface({ input: native.stdout }).on("line", line => {
    const pending = waiting.shift();
    if (!pending) return;
    try { pending.resolve(JSON.parse(line)); } catch (error) { pending.reject(error); }
  });
  native.on("error", error => { for (const pending of waiting.splice(0)) pending.reject(error); });
  native.on("exit", code => { for (const pending of waiting.splice(0)) pending.reject(new Error(`fixture exited ${code}`)); });
  const nativeRun = request => new Promise((resolve, reject) => {
    waiting.push({ resolve, reject });
    native.stdin.write(`${JSON.stringify(request)}\n`);
  });
  const wasmRun = request => JSON.parse(wasm.run(JSON.stringify(request)));
  const both = async request => [await nativeRun(request), wasmRun(request)];
  const successful = result => { assert.equal(result.error, undefined, JSON.stringify(result)); return result; };
  const hash = values => createHash("sha256").update(Buffer.from(values)).digest("hex");
  const envelope = (count, mode, body) => {
    const bytes = Buffer.alloc(21 + body.length);
    bytes.writeBigUInt64LE(BigInt(count)); bytes.write("TFZ1", 8); bytes[12] = mode;
    bytes.writeBigUInt64LE(BigInt(body.length), 13); Buffer.from(body).copy(bytes, 21);
    return [...bytes];
  };
  const inspect = bytes => {
    const buffer = Buffer.from(bytes), count = Number(buffer.readBigUInt64LE()), mode = bytes[12];
    assert.equal(buffer.toString("ascii", 8, 12), "TFZ1");
    assert.equal(Number(buffer.readBigUInt64LE(13)), bytes.length - 21);
    if (mode === 0) { assert.equal(bytes.length, 21 + 4 * count); return { mode, coded: 0, raw: 0 }; }
    assert.equal(mode, 1); assert.ok(bytes.length - 21 < count * 4);
    let at = 21, coded = 0, raw = 0;
    for (let left = count; left > 0; left -= 4) {
      const descriptor = bytes[at++];
      if (descriptor === 0) { raw++; at += 4 * Math.min(left, 4); }
      else { assert.ok(descriptor <= 120 && left >= 4); coded++; at += Math.ceil(descriptor / 8); }
    }
    assert.equal(at, bytes.length);
    return { mode, coded, raw };
  };
  const golden = JSON.parse(await readFile(goldenPath, "utf8"));
  report.source = golden.source;
  assert.equal(golden.source.commit, "f40868a6a1c190c802e7d8b5987064f044bf7812");
  const source = (await readFile(join(root, "tools/terrain_codec/zfp-golden.c"), "utf8")).replaceAll("\r\n", "\n");
  assert.equal(hash(Buffer.from(source)), golden.source.generator_sha256);
  await writeFile(join(output, "zfp-golden.json"), JSON.stringify(golden, null, 2));
  for (const block of golden.blocks) {
    assert.deepEqual(block.decoded, block.input, "upstream self round trip");
    const bits = [...block.input, 0, 0, 0, 0];
    const encoded = (await both({ op: "encode", container: "vec", bits })).map(successful);
    assert.deepEqual(encoded[0], encoded[1], "native/Wasm golden bytes");
    const bytes = encoded[0].bytes;
    assert.equal(bytes[12], 1);
    if (block.bits <= 120) {
      assert.equal(bytes[21], block.bits);
      assert.deepEqual(bytes.slice(22, 22 + block.bytes.length), block.bytes, "Rust encode matches upstream decode input");
      const upstream = envelope(8, 1, [block.bits, ...block.bytes, 1, 0]);
      for (const result of await both({ op: "decode", container: "vec", bytes: upstream }))
        assert.deepEqual(successful(result).bits, bits, "Rust decodes upstream block");
    } else assert.equal(bytes[21], 0, "complete block above 120 bits escapes raw");
    for (const result of await both({ op: "decode", container: "vec", bytes }))
      assert.deepEqual(successful(result).bits, bits);
    report.golden_blocks++;
  }

  const word = new Uint32Array(1), number = new Float32Array(word.buffer);
  const floatBits = value => { number[0] = value; return word[0]; };
  const specials = [0, 0x80000000, 1, 0x80000001, 0x7fffff, 0x807fffff, 0x800000, 0x80800000,
    0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0x7fc01234, 0x7f801234, 0xffc01234, 0xff801234];
  const patterns = {
    zero: () => 0,
    constant: () => floatBits(132.5),
    smooth: i => floatBits(100 + i / 256),
    monotone: i => floatBits(i - 400),
    alternating: i => floatBits((i & 1 ? -1 : 1) * (2 + i / 1024)),
    signed_zero: i => (i & 1 ? 0x80000000 : 0),
    exceptional: i => specials[i % specials.length],
    mixed_escape: i => i % 16 < 12 ? 0 : specials[(i * 3) % specials.length],
    random: () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return seed >>> 0; },
  };
  let seed = 0x315972ae;
  let coded = 0, raw = 0, fallback = 0;
  const lengths = [...Array.from({ length: 10 }, (_, i) => i), 257, 258, 259, 260, 259 * 5, 259 * 17, 259 * 259];
  for (const [pattern, make] of Object.entries(patterns)) for (const length of lengths) {
    const bits = Array.from({ length }, (_, i) => make(i));
    let expected;
    const started = performance.now();
    for (const container of ["vec", "box", "arc"]) {
      const encoded = (await both({ op: "encode", container, bits })).map(successful);
      assert.deepEqual(encoded[0], encoded[1], `${pattern}/${length} native/Wasm`);
      expected ??= encoded[0].bytes;
      assert.deepEqual(encoded[0].bytes, expected, "container-independent bytes");
      // Each target consumes the other target's bytes.
      assert.deepEqual(successful(await nativeRun({ op: "decode", container, bytes: encoded[1].bytes })).bits, bits);
      assert.deepEqual(successful(wasmRun({ op: "decode", container, bytes: encoded[0].bytes })).bits, bits);
    }
    const stats = inspect(expected);
    coded += stats.coded; raw += stats.raw; fallback += Number(stats.mode === 0 && length > 0);
    report.fixtures.push({ pattern, length, old_bytes: 8 + length * 4, bytes: expected.length,
      sha256: hash(expected), ...stats, native_wasm_exchange_ms: performance.now() - started });
    if (length === 259 * 259 && ["constant", "smooth", "random"].includes(pattern))
      report.native_codec_context.push({ pattern, ...successful(await nativeRun({ op: "profile", bits })) });
  }
  assert.ok(coded > 0 && raw > 0 && fallback > 0, "all representation paths executed");

  const bad = [];
  const rawBytes = envelope(3, 0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  const codedBytes = envelope(8, 1, [1, 0, 1, 0]);
  const mixedBytes = envelope(13, 1, [1, 0, 0, ...Array.from({ length: 16 }, (_, i) => i), 1, 0, 0, 0, 0, 0, 128]);
  for (const [name, bytes] of [["raw", rawBytes], ["coded", codedBytes], ["mixed-tail", mixedBytes]]) {
    for (const result of await both({ op: "decode", container: "vec", bytes })) successful(result);
    for (let end = 0; end < bytes.length; end++) bad.push([`${name}-truncated-${end}`, bytes.slice(0, end)]);
    bad.push([`${name}-object-trailing`, [...bytes, 0]]);
  }
  const changed = (bytes, at, data) => { const copy = [...bytes]; copy.splice(at, data.length, ...data); return copy; };
  const countBytes = value => { const bytes = Buffer.alloc(8); bytes.writeBigUInt64LE(value); return [...bytes]; };
  bad.push(["old-untagged", [4, 0, 0, 0, 0, 0, 0, 0, ...Array(16).fill(0)]]);
  bad.push(["unknown-tag", changed(codedBytes, 8, [84, 70, 90, 50])]);
  for (const mode of [2, 3, 127, 255]) bad.push([`mode-${mode}`, changed(codedBytes, 12, [mode])]);
  for (let descriptor = 121; descriptor <= 255; descriptor++)
    bad.push([`descriptor-${descriptor}`, changed(codedBytes, 21, [descriptor])]);
  for (const count of [0x1_0000_0000n, 0x4000_0000_0000_0000n, 0xffff_ffff_ffff_ffffn]) {
    bad.push([`count-${count}`, changed(codedBytes, 0, countBytes(count))]);
    bad.push([`length-${count}`, changed(codedBytes, 13, countBytes(count))]);
  }
  bad.push(["empty-blocked", envelope(0, 1, [])], ["raw-wrong-length", envelope(4, 0, [0])],
    ["blocked-not-smaller", envelope(1, 1, [0, 0, 0, 0, 0])],
    ["missing-group", envelope(8, 1, [1, 0])], ["extra-group", envelope(4, 1, [1, 0, 1, 0])],
    ["coded-tail", envelope(5, 1, [1, 0, 1, 0])], ["unfinished-bits", envelope(4, 1, [1, 1])],
    ["nonzero-padding", envelope(4, 1, [1, 128])], ["extra-coded-bits", envelope(4, 1, [2, 0])]);
  // Truncate inside every field and bit plane of complete exponent and
  // reinterpretation blocks, retaining a well-formed outer envelope.
  for (const reinterpret of [false, true]) {
    const block = golden.blocks.find(block => block.bits > 40 && block.bits < 113
      && !!(block.bytes[0] & 2) === reinterpret);
    assert.ok(block);
    for (let bits = 1; bits < block.bits; bits++) {
      const payload = block.bytes.slice(0, Math.ceil(bits / 8));
      if (bits % 8) payload[payload.length - 1] &= (1 << (bits % 8)) - 1;
      bad.push([`bit-truncation-${reinterpret}-${bits}`, envelope(8, 1, [bits, ...payload, 1, 0])]);
    }
  }
  for (const [name, bytes] of bad) {
    for (const container of ["vec", "box", "arc"]) for (const result of await both({ op: "decode", container, bytes }))
      assert.equal(typeof result.error, "string", `${name}/${container} must fail`);
    report.malformed.push(name);
  }
  const production = (await both({ op: "production" })).map(successful);
  assert.deepEqual(production[0], production[1], "native/Wasm production jobs and every fixture byte agree");
  report.production = production[0];
  console.log(JSON.stringify({ golden_blocks: report.golden_blocks, arrays: report.fixtures.length,
    malformed: report.malformed.length, coded, raw, fallback, production: report.production }));
  await writeFile(join(output, "report.json"), JSON.stringify(report, null, 2));
} catch (error) {
  await writeFile(join(output, "failure.json"), JSON.stringify({ message: String(error), stack: error.stack, report }, null, 2));
  console.error(error);
  process.exitCode = error instanceof assert.AssertionError ? 3 : 1;
} finally {
  if (native && native.exitCode === null) {
    const finished = once(native, "exit");
    native.stdin.end();
    await finished;
  }
}
