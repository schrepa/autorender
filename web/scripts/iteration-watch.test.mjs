import assert from "node:assert/strict";
import fsPromises from "node:fs/promises";
import { mkdtemp, mkdir, readFile, rename, rm, stat, symlink, utimes, writeFile } from "node:fs/promises";
import { syncBuiltinESMExports } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import test from "node:test";
import { fingerprintSources, watchSources } from "./iteration-watch.mjs";

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), "engine-iteration-watch-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const put = async (path, content = "a") => {
    const full = join(root, path);
    await mkdir(dirname(full), { recursive: true });
    await writeFile(full, content);
  };
  return { root, put };
}

async function until(condition, message = "condition was not reached") {
  const deadline = Date.now() + 4000;
  while (!condition()) {
    if (Date.now() >= deadline) throw new Error(message);
    await delay(10);
  }
}

test("fingerprints include all browser compile input roots and stable path ordering", async t => {
  const first = await fixture(t);
  const second = await fixture(t);
  const files = [
    ["crates/renderer/src/meshes.wgsl", "shader"],
    ["crates/engine/build.rs", "build"],
    ["examples/game/Cargo.toml", "example"],
    ["examples/game/src/lib.rs", "game"],
    ["Cargo.toml", "workspace"],
    ["Cargo.lock", "dependencies"],
    ["rust-toolchain.toml", "toolchain"],
    [".cargo/config.toml", "flags"],
    ["vendor/glam/src/f32/math.rs", "patched glam"],
  ];
  for (const [path, content] of files) await first.put(path, content);
  for (const [path, content] of files.toReversed()) await second.put(path, content);
  const initial = await fingerprintSources(first.root);
  assert.deepEqual(initial, await fingerprintSources(second.root));
  assert.match(initial.fingerprint, /^[a-f0-9]{64}$/);
  assert.equal(initial.files, files.length);
  for (const [path, content] of files) {
    await first.put(path, `${content}!`);
    assert.notEqual((await fingerprintSources(first.root)).fingerprint, initial.fingerprint, path);
    await first.put(path, content);
  }
});

test("content changes are detected with the same size and restored timestamp", async t => {
  const { root, put } = await fixture(t);
  const path = "crates/game/src/lib.rs";
  await put(path, "first");
  const before = await fingerprintSources(root);
  const times = await stat(join(root, path));
  await put(path, "other");
  await utimes(join(root, path), times.atime, times.mtime);
  assert.equal((await stat(join(root, path))).size, times.size);
  assert.notEqual((await fingerprintSources(root)).fingerprint, before.fingerprint);
});

test("new, deleted, and renamed files change the fingerprint including empty inputs", async t => {
  const { root, put } = await fixture(t);
  const empty = await fingerprintSources(root);
  await put("crates/game/new.rs", "");
  const added = await fingerprintSources(root);
  assert.equal(added.files, 1);
  assert.notEqual(added.fingerprint, empty.fingerprint);
  await rename(join(root, "crates/game/new.rs"), join(root, "crates/game/renamed.rs"));
  assert.notEqual((await fingerprintSources(root)).fingerprint, added.fingerprint);
  await rm(join(root, "crates/game/renamed.rs"));
  assert.deepEqual(await fingerprintSources(root), empty);
});

test("generated directories, native tooling, and reloadable scenes are excluded", async t => {
  const { root, put } = await fixture(t);
  await put("crates/game/src/lib.rs", "source");
  const before = await fingerprintSources(root);
  for (const name of ["target", "pkg", "pkg-dev", "dist", ".git", "node_modules", ".tools"]) {
    await put(`crates/game/${name}/generated.bin`, name);
    await put(`examples/game/${name}/generated.bin`, name);
  }
  await put("tools/xtask/src/main.rs", "native supervisor");
  await put("web/public/scenes/playground.scene.json", "reloadable scene");
  await put("docs/architecture.md", "documentation");
  await put("web/pkg-dev/game_bg.wasm", "generated wasm");
  assert.deepEqual(await fingerprintSources(root), before);
});

test("source budgets reject excessive file counts and bytes while exact bounds pass", async t => {
  const { root, put } = await fixture(t);
  await put("crates/a.rs", "1234");
  await put("crates/b.rs", "5678");
  assert.equal((await fingerprintSources(root, { maxFiles: 2, maxBytes: 8 })).files, 2);
  await assert.rejects(fingerprintSources(root, { maxFiles: 1 }), { code: "source_budget" });
  await assert.rejects(fingerprintSources(root, { maxBytes: 7 }), { code: "source_budget" });
  await assert.rejects(fingerprintSources(root, { maxFiles: 20_001 }), TypeError);
  await assert.rejects(fingerprintSources(root, { maxBytes: 128 * 1024 * 1024 + 1 }), TypeError);
});

test("directory depth is bounded even for trees with no regular files", async t => {
  const { root } = await fixture(t);
  await mkdir(join(root, "crates", ...Array(64).fill("d")), { recursive: true });
  await assert.rejects(fingerprintSources(root), { code: "source_budget" });
});

test("symlinked input directories and input ancestors fail clearly", async t => {
  const { root, put } = await fixture(t);
  await put("external/file.rs", "outside scope");
  await mkdir(join(root, "crates"));
  await symlink(join(root, "external"), join(root, "crates", "linked"), "junction");
  await assert.rejects(fingerprintSources(root), { code: "source_symlink", path: "crates/linked" });
  await rm(join(root, "crates", "linked"));
  await symlink(join(root, "external"), join(root, "vendor"), "junction");
  await assert.rejects(fingerprintSources(root), { code: "source_symlink", path: "vendor" });
  assert.equal(await readFile(join(root, "external/file.rs"), "utf8"), "outside scope");
});

test("watcher coalesces rapid edits to the latest stable fingerprint", async t => {
  const { root, put } = await fixture(t);
  await put("crates/game.rs", "initial");
  const observations = [];
  const errors = [];
  const stop = watchSources({ root, intervalMs: 15, debounceMs: 120, onChange: value => observations.push(value), onError: error => errors.push(error) });
  t.after(stop);
  await until(() => observations.length === 1);
  assert.equal(observations[0].initial, true);
  await put("crates/game.rs", "edit one");
  await delay(35);
  await put("crates/game.rs", "edit two");
  await delay(35);
  await put("crates/game.rs", "edit final");
  await until(() => observations.length === 2);
  assert.equal(observations[1].initial, false);
  assert.equal(observations[1].fingerprint, (await fingerprintSources(root)).fingerprint);
  await delay(160);
  assert.equal(observations.length, 2);
  assert.ok(errors.every(error => error.code === "source_changed"), JSON.stringify(errors));
});

test("watcher cancels a pending edit when source returns to its notified state", async t => {
  const { root, put } = await fixture(t);
  await put("crates/game.rs", "original");
  const observations = [];
  const stop = watchSources({ root, intervalMs: 10, debounceMs: 120, onChange: value => observations.push(value) });
  t.after(stop);
  await until(() => observations.length === 1);
  await put("crates/game.rs", "temporary");
  await delay(40);
  await put("crates/game.rs", "original");
  await delay(180);
  assert.equal(observations.length, 1);
});

test("asynchronous build coordination does not block source scans", async t => {
  const { root, put } = await fixture(t);
  await put("crates/game.rs", "original");
  const observations = [];
  const stop = watchSources({ root, intervalMs: 10, debounceMs: 10, onChange: value => {
    observations.push(value);
    return new Promise(() => {});
  } });
  t.after(stop);
  await until(() => observations.length === 1);
  await put("crates/game.rs", "changed");
  await until(() => observations.length === 2);
  assert.equal(observations[1].fingerprint, (await fingerprintSources(root)).fingerprint);
});

test("watcher reports invalid sources and recovers after they are repaired", async t => {
  const { root, put } = await fixture(t);
  await mkdir(join(root, "Cargo.toml"));
  const observations = [];
  const errors = [];
  const stop = watchSources({ root, intervalMs: 10, debounceMs: 10, onChange: value => observations.push(value), onError: error => errors.push(error) });
  t.after(stop);
  await until(() => errors.length > 0);
  assert.equal(errors[0].code, "source_type");
  assert.equal(observations.length, 0);
  await rm(join(root, "Cargo.toml"), { recursive: true });
  await put("Cargo.toml", "fixed");
  await until(() => observations.length === 1);
  assert.equal(observations[0].initial, true);
});

test("repairing an error to the same source fingerprint publishes one recovery notification", async t => {
  const { root, put } = await fixture(t);
  await put("Cargo.toml", "original");
  const observations = [];
  const errors = [];
  const stop = watchSources({ root, intervalMs: 10, debounceMs: 10, onChange: value => observations.push(value), onError: error => errors.push(error) });
  t.after(stop);
  await until(() => observations.length === 1);
  await rm(join(root, "Cargo.toml"));
  await mkdir(join(root, "Cargo.toml"));
  await until(() => errors.length > 0);
  assert.equal(errors[0].code, "source_type");
  await rm(join(root, "Cargo.toml"), { recursive: true });
  await put("Cargo.toml", "original");
  await until(() => observations.length === 2);
  assert.equal(observations[1].initial, false);
  assert.equal(observations[1].fingerprint, observations[0].fingerprint);
  await delay(40);
  assert.equal(observations.length, 2);
});

test("a file disappearing during a scan retries silently instead of publishing a watcher error", async t => {
  const { root, put } = await fixture(t);
  await put("crates/game.rs", "original");
  const observations = [];
  const errors = [];
  const stop = watchSources({ root, intervalMs: 10, debounceMs: 10, onChange: value => observations.push(value), onError: error => errors.push(error) });
  t.after(stop);
  await until(() => observations.length === 1);

  // Inject the filesystem boundary of an editor rename once, after directory
  // discovery and before its child's stat. Production APIs need no test hook.
  const originalStat = fsPromises.lstat;
  const target = join(root, "crates/game.rs");
  let raced = false;
  const restore = () => { fsPromises.lstat = originalStat; syncBuiltinESMExports(); };
  t.after(restore);
  fsPromises.lstat = async (path, options) => {
    if (path === target && !raced) {
      raced = true;
      throw Object.assign(new Error("editor replaced the file"), { code: "ENOENT" });
    }
    return originalStat(path, options);
  };
  syncBuiltinESMExports();
  try {
    await until(() => raced);
    await put("crates/game.rs", "changed");
    await until(() => observations.length === 2);
    assert.equal(observations[1].fingerprint, (await fingerprintSources(root)).fingerprint);
    assert.deepEqual(errors, []);
  } finally { restore(); stop(); }
});

test("stop and abort suppress later callbacks, including initial scans", async t => {
  const { root, put } = await fixture(t);
  await put("crates/game.rs", "source");
  const observations = [];
  const lifecycle = new AbortController();
  const stop = watchSources({ root, intervalMs: 10, debounceMs: 10, signal: lifecycle.signal, onChange: value => observations.push(value) });
  t.after(stop);
  await until(() => observations.length === 1);
  lifecycle.abort();
  stop();
  stop();
  await put("crates/game.rs", "changed");
  watchSources({ root, intervalMs: 10, signal: lifecycle.signal, onChange: value => observations.push(value) });
  const stopInitial = watchSources({ root, intervalMs: 10, onChange: value => observations.push(value) });
  stopInitial();
  await delay(80);
  assert.equal(observations.length, 1);
});

test("callback failures are reported and error reporter failures remain contained", async t => {
  const { root } = await fixture(t);
  const errors = [];
  const stop = watchSources({ root, intervalMs: 10, onChange: async () => { throw new Error("callback failed"); }, onError: async error => {
    errors.push(error);
    throw new Error("report failed");
  } });
  t.after(stop);
  await until(() => errors.length === 1);
  assert.equal(errors[0].message, "callback failed");
});
