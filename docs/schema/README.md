# Generated authoring schemas

`scene-v6.schema.json` is generated from the versioned Rust types in
`engine_scene`, using its host-only `schema` feature. Regenerate deliberately:

```sh
cargo xtask schema
```

The versioned artifact supports editor discovery and reviewable format changes.
`cargo xtask lint` rejects stale output without rewriting it. Associate the file
with `web/public/scenes/*.scene.json` in your editor's JSON schema settings.

The schema describes structure, defaults, supported recipes, ID patterns and
collection limits. Runtime validation additionally checks unique IDs, references,
geometry, numeric invariants and authoring bounds. Use `cargo xtask assets` for
the same validation used by the browser scene loader. Schemas do not encode or
persist runtime entity IDs or asset handles.

Version 6 adds material `opacity` (below 1, transparent), entity `light`
(point or spot) and `particles` (an emitter), and the `local_lights`,
`sun_shafts`, `reflections` and `particles` environment sections (decisions
[023](../adr-023-local-lights.md) to
[026](../adr-026-transparency-and-particles.md)), and the atmosphere's
`contrast`, `vibrance`, `shadow_tint` and `highlight_tint`
([decision 027](../adr-027-stylized-grading.md)), and `shadows.cascades`
([decision 028](../adr-028-shadow-cascades-and-receiver-planes.md)). An environment that omits
`sun_shafts` or `reflections` turns them off. Versions 1–5 keep their
meaning and schema artifacts; atmosphere presets are engine looks, so their
values follow the engine.

Version 5 adds an optional material `surface`: roughness, metalness,
emission and ripples, matte when omitted
([decision 019](../adr-019-surface-materials-and-reflections.md)). Authored
textures stay opaque, so a textured material emits over its whole surface.
It also adds an optional `environment`: an `atmosphere` preset with
overrides, `shadows`, `ambient_occlusion` and `bloom`
([decision 022](../adr-022-scene-environments-and-agent-controls.md)). A
document with an environment sets all four on instantiation, omitted
sections taking the defaults; one without leaves the look to the game.
Versions 1–4 keep their meaning and schema artifacts.

Version 4 adds optional authored cuboid/sphere colliders, sensor flags and
unsigned membership/filter masks independent of rendering. Local shape bounds
and composed collision transforms are checked by the shared collision crate
before scene mutation. Versions 1–3 do not acquire implicit colliders and their
schema artifacts remain unchanged.

Version 3 adds parent transforms, transform-only nodes and prefab definitions
with stable expanded IDs. Versions 1 and 2 retain their original semantics;
their schemas remain unchanged. Textures/UVs require version 2 or later, while
hierarchy/prefab fields require version 3. Cycle, depth, expanded count and ID
collision checks run in Rust in addition to structural schema validation.
