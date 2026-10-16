/** Source identities and the arithmetic/binding contracts probed on the GPU. */
// @ts-check
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { workspacePath } from "./workspace.mjs";

export async function auditSources() {
  const paths = ["bindings.rs", "lights.rs", "light_depth.rs", "light_depth_interval.wgsl", "light_depth_atomic.wgsl", "meshes/light_depth.wgsl",
    "meshes/shading.wgsl", "renderer/meshes.rs", "renderer/light_depth.rs", "renderer/frame/light_depth.rs", "renderer/frame/opaque.rs",
    "renderer/frame/encode.rs", "renderer/terrain_draws.rs", "renderer/scatter_draws.rs", "renderer/probe_captures.rs"];
  const sources = Object.fromEntries(await Promise.all(paths.map(async path => [path, await readFile(workspacePath(`crates/engine_render/src/${path}`), "utf8")])));
  assert.ok(sources["meshes/shading.wgsl"].includes("let d2 = dot(to_light, to_light);") && sources["meshes/shading.wgsl"].includes("let r2 = light.range * light.range;"));
  assert.ok(sources["meshes/shading.wgsl"].includes("if d2 >= r2"));
  assert.ok(sources["light_depth_interval.wgsl"].includes("bitcast<f32>(bits + 1u)") && sources["light_depth_interval.wgsl"].includes("bitcast<f32>(bits - 1u)"));
  assert.ok(sources["meshes/light_depth.wgsl"].includes("override DEPTH_OCCUPANCY: bool = false;"));
  assert.ok(sources["renderer/meshes.rs"].includes("self.create_shading_variant(kind, variant, false)"));
  const binning = sources["bindings.rs"].split("const LIGHT_BINNING: Table")[1].split("};")[0];
  assert.equal((binning.match(/\b(?:read|write)\(/g) ?? []).length, 8, "the original binner still uses eight storage bindings");
  for (const path of ["renderer/frame/encode.rs", "renderer/terrain_draws.rs", "renderer/scatter_draws.rs", "renderer/frame/opaque.rs"]) {
    assert.ok(sources[path].includes("lights.fragment_group()"), `${path}: full-list group remains bound`);
  }
  assert.ok(sources["renderer/probe_captures.rs"].includes("lights.probe_fragment_group(face)"));
  return {
    files: paths.map(path => ({ path: `crates/engine_render/src/${path}`, sha256: createHash("sha256").update(sources[path]).digest("hex") })),
    widening: sources["light_depth_interval.wgsl"],
    original_range_test: sources["meshes/shading.wgsl"].slice(sources["meshes/shading.wgsl"].indexOf("let to_light = light.position - point.world;"), sources["meshes/shading.wgsl"].indexOf("let to_light = light.position - point.world;") + 300),
    routing: "Full variants retain the false override; full uniforms keep zero tags. Production helper GPU cases separately exercise every named full-route contract. Only glass and decals are asserted to draw in the variant fixture; other named consumers are covered by routing-kernel cases.",
    timestamps: "The established opaque/light-bin boundaries are unchanged. Each phase has its own collection/filter span, including reset and copy costs; phase-two work is also inside the original multi-phase opaque span.",
  };
}
