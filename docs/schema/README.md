# Generated authoring schemas

`scene-v4.schema.json` is generated from the versioned Rust types in
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
