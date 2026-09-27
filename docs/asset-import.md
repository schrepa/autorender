# Import and replace textured scenery

The supported path is **GLB source → native import → validated scene JSON → guarded save → browser reload → inspect/capture**. The importer and runtime share `engine_scene` validation. Importing does not run a browser, and reloading does not save source.

## Import

Activate the repository tools in PowerShell with `. ./scripts/activate.ps1`. Create a separate, reviewable model scene:

```sh
cargo xtask asset import --file assets/source/garden-prop.glb --id garden --output artifacts/garden.scene.json
```

The output path must be new. The result is versioned JSON with counts, saved path/revision, SHA-256 of the exact input, importer version, namespace, and import settings. Save this report with build artifacts when provenance is needed. The source GLB remains separate from the cooked scene data. A model scene contains scenery only; the Crystal Garden client additionally requires its player/game setup.

To place a model in an existing game, import into its scene using the current saved revision:

```sh
cargo xtask asset import --file assets/source/garden-prop.glb --id scenery --into web/public/scenes/playground.scene.json --expected-revision N
```

This validates the entire candidate before the shared revision-checked writer commits it. It increments the revision and upgrades the document to scene version 6. Existing camera and gameplay records remain. Change the new entities' `transform` fields through the normal scene authoring workflow to place them. Asset names are stable within the import namespace and source node/primitive indices. No runtime handles are persisted.

Importing the same namespace again fails on duplicate IDs unless `--replace` is supplied:

```sh
cargo xtask asset import --file assets/source/garden-prop.glb --id scenery --into web/public/scenes/playground.scene.json --expected-revision N --replace
```

Replacement owns the generated IDs in that namespace: `NAMESPACE.imageN`, `.materialN`, `.nodeN.pN`, and `.nodeN.pN.mesh`. Reserve these names for the importer. Unrelated IDs are retained. Matching flat entity IDs retain authored placement, parent, names, game tags, and authored colliders. Source node reordering changes identity; review the result when editing model structure. References broken by removal are rejected before saving, leaving the previous source untouched. A stale revision or malformed import also leaves the previous source intact.

## Reusable prefab import

Add `--as-prefab` to create one definition named by `--id`, referencing its shared
scene assets. This creates no placement entities:

```sh
cargo xtask asset import --file assets/source/garden-prop.glb --id scenery --as-prefab --into web/public/scenes/playground.scene.json --expected-revision N
```

Then author placement roots through the normal scene save path:

```json
{"id":"west-scenery","prefab":"scenery","transform":{"translation":[-8,0,-5]}}
```

One root expands into stable children such as `west-scenery.node1.p0`. Multiple
placements share the scene's mesh, material and texture handles. Reimport with
both `--replace --as-prefab` to replace generated assets and the definition while
preserving placement records and their transforms/tags. Removed children/assets
referenced elsewhere cause validation failure before save. Flat and prefab
namespaces cannot silently replace each other; migrate source explicitly first.
The hand-authored `garden-lantern` definition in Crystal Garden demonstrates the
same runtime format with readable local names `planter` and `lantern`.

## Observe the result

In the paused development session, use the saved scene reload command with the previous runtime revision and newly saved revision. Then request `assets` and inspect an imported entity's `Mesh3d` projection. See [agent workflow](agent-workflow.md) for command envelopes.

`assets` reports CPU counts/bytes, authored texture IDs, dimensions, `cpu_ready` or `missing` state, and `gpu_resident` in a browser. GPU residency means an upload has been queued and cached at a valid render boundary; it does not claim GPU completion. Unreferenced authored textures can be CPU-ready without being GPU-resident. `Mesh3d` reports texture dependency readiness without returning pixels or runtime handles.

Rendering observations include `resident_textures`, `resident_texture_bytes`, and `texture_upload_bytes`. Repeated frames with unchanged textures should upload zero texture bytes. A capture is the visual check that imported UVs and pixels match expectations. Failed scene validation/admission preserves the active game. Successful reload replaces scene-owned resources and resets the sample's gameplay, following the existing reload contract.

## Supported subset and budgets

- One GLB 2 scene with one embedded BIN buffer; indexed triangle primitives with FLOAT positions, normals, and optional FLOAT `TEXCOORD_0`. Textured primitives require UVs.
- Static node transforms are baked into geometry, including normal transforms and mirrored triangle winding. Prefab mode groups those baked parts; it does not reconstruct the GLB node hierarchy or add animation.
- Embedded 8-bit RGB/RGBA PNG base-color textures, opaque pixels only. Colors are sRGB texels multiplied by linear material color and the engine's existing diffuse lighting.
- Explicit clamp-to-edge samplers with linear filtering; minification may also be `LINEAR_MIPMAP_LINEAR`. Scenes store one level: the renderer generates mip chains on the GPU and samples every texture trilinearly. Materials require opaque single-sided rendering, metallic factor 0 and roughness factor 1. This renderer does not implement glTF PBR.
- External/data URIs, skins, animation, morphs, cameras, extensions/compression, extra vertex attributes, transparency, normal/emissive/occlusion/metallic textures, and unsupported samplers are rejected with source paths.

The native input is bounded to 16 MiB. Output remains the engine's self-contained 1 MiB scene format: indexed vertices plus base64 RGBA8 pixels. The importer decodes PNG offline; the browser decodes base64 only at load/validation boundaries. This intentionally supports small static game assets. There is no binary streaming package or compressed GPU texture format yet (mips are generated on the GPU at upload); the source limit is often reached before the dimension limit.

Textures are at most 2048×2048. CPU texture data shares the library's 32 MiB payload budget, with at most 64 textures by default. The renderer independently limits textures to 64 / 32 MiB. Reload admission needs room for both old and new CPU assets until replacement commits. GPU objects remain renderer-owned; old snapshots can temporarily retain CPU pixels. Neither import libraries nor image decoders are browser runtime dependencies.

## Verification

Native tests cover malformed input, unsupported features, accessor bounds, transforms, deterministic output, decoded budgets, explicit namespace replacement, and guarded source writes. `node web/scripts/texture-smoke.mjs` checks real Chrome rendering through the development interface; it routes disposable candidate responses and never edits scene files. Use `DEMO_URL` to select the development server. The test verifies texture readiness, zero repeat uploads, failed replacement preservation, visible pixel changes, stable repeated replacement, and GPU retirement.

Version 4 reimports preserve authored colliders on surviving deterministic flat
and prefab-template IDs. Removing a generated collider-bearing node is rejected
with `collider_reimport_conflict`; explicitly migrate or remove its collider in
source first. Placement-root colliders remain with their placements. Importing
never infers collider geometry from render meshes. This behavior is recorded by
importer provenance version `static-glb-3`.
