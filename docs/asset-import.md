# Import and replace textured scenery

The supported path is **GLB source → native import → validated scene JSON and its files → guarded save → browser reload → inspect/capture**. The importer and runtime share `engine_scene` validation. Importing does not run a browser, and reloading does not save source. Materials import as glTF 2.0 metallic-roughness materials ([decision 031](adr-031-gltf-materials.md)); images and meshes become files beside the scene ([decision 030](adr-030-scene-files.md)).

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

This validates the entire candidate before the shared revision-checked writer commits it. It increments the revision and upgrades the document to the current scene version with `SceneDocument::upgrade`, which keeps its existing textures clamped. Existing camera and gameplay records remain. Change the new entities' `transform` fields through the normal scene authoring workflow to place them. Asset names are stable within the import namespace and source node/primitive indices. No runtime handles are persisted.

Importing the same namespace again fails on duplicate IDs unless `--replace` is supplied:

```sh
cargo xtask asset import --file assets/source/garden-prop.glb --id scenery --into web/public/scenes/playground.scene.json --expected-revision N --replace
```

Replacement owns the generated IDs in that namespace: `NAMESPACE.imageN`, `.materialN`, `.nodeN.pN`, and `.nodeN.pN.mesh`. Reserve these names for the importer. Files are written to the folder `NAMESPACE/` beside the scene, named by their content (`imageN.HASH.jpg`, `nodeN.pM.HASH.bin`), so a reimport never overwrites a file the saved scene reads; files a replacement no longer references stay until removed. Unrelated IDs are retained. Matching flat entity IDs retain authored placement, parent, names, game tags, and authored colliders. Source node reordering changes identity; review the result when editing model structure. References broken by removal are rejected before saving, leaving the previous source untouched. A stale revision or malformed import also leaves the previous source intact.

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

## Supported subset

- One GLB 2 scene with one embedded BIN buffer; indexed or unindexed triangle primitives with FLOAT positions, optional normals, optional `TEXCOORD_0` and `TEXCOORD_1`, optional `COLOR_0` in any glTF format (decision 034) and, on skinned nodes, `JOINTS_0` and `WEIGHTS_0` (decision 036); primitives with any of these write `EMB2` buffers. Primitives without normals get flat normals, as glTF requires; `TANGENT` and further coordinate or color sets are ignored (normal maps derive their tangent frames). Textured primitives require UVs.
- Static models bake node transforms into geometry, including normal transforms and mirrored triangle winding. Models with animations or skins keep their nodes instead (decisions 035 and 036): entities `ID.nodeN` with local transforms and parents below a root `ID.root` that plays the first animation, primitives as children `ID.nodeN.pM`, clips `ID.animN` in `EAN1` files and skins `ID.skinN`; their rest poses must not be mirrored, zero-scaled or sheared. Prefab mode keeps the same entities as a definition whose animation targets and skin joints name local IDs. Cameras are not imported.
- glTF metallic-roughness materials map one to one: base color, metallic and roughness factors; base color, metallic-roughness, normal (with scale), occlusion (with strength) and emissive textures; the emissive factor times `KHR_materials_emissive_strength`; `OPAQUE`, `MASK` (with its cutoff) and `BLEND` alpha modes; double-sided materials; every sampler wrap mode and filter; and `KHR_texture_transform` offsets, rotations and scales. Maps read `TEXCOORD_0` or `TEXCOORD_1`. Glass takes `KHR_materials_transmission`'s factor, `KHR_materials_ior` and `KHR_materials_volume`'s thickness factor as world units (decision 038); transmission textures are not read.
- Embedded PNG and JPEG images are written unchanged; the browser decodes them at load. KTX2 images of `KHR_texture_basisu` textures replace the textures' own sources and are written unchanged too; the browser transcodes them to a compressed GPU format at load (decision 037). Meshes are written as `EMB1` or `EMB2` buffers.
- External/data URIs, morph targets and weights, sparse accessors, more than four joints per vertex and required extensions other than `KHR_texture_transform`, `KHR_materials_emissive_strength`, `KHR_texture_basisu`, `KHR_materials_transmission`, `KHR_materials_ior` and `KHR_materials_volume` are rejected with source paths. Other optional extensions are ignored and listed in the result's `ignored_extensions`.

The input is bounded only by the GLB format's 4 GiB length field; there is no count, vertex or decoded-size budget (decision 029). PNG and JPEG textures upload as RGBA8 with mips generated on the GPU; KTX2 textures (Basis Universal ETC1S or UASTC, as `gltf-transform` with KTX-Software writes them) upload their own levels block-compressed (decision 037).

Textures are at most 8192×8192. The asset library and renderer set no texture count or byte budget by default (decision 029); a game may set an asset budget. Reload admission needs room for both old and new CPU assets until replacement commits. GPU objects remain renderer-owned; old snapshots can temporarily retain CPU pixels. Neither import libraries nor image decoders are browser runtime dependencies.

## Verification

Native tests cover malformed input, unsupported features, accessor bounds, transforms, flat normals, one-to-one material mapping, deterministic content-named files, explicit namespace replacement, and guarded source writes. `node web/scripts/texture-smoke.mjs` checks real Chrome rendering through the development interface; it routes disposable candidate responses and never edits scene files. Use `DEMO_URL` to select the development server. The test verifies texture readiness, zero repeat uploads, failed replacement preservation, visible pixel changes, stable repeated replacement, and GPU retirement.

Version 4 reimports preserve authored colliders on surviving deterministic flat
and prefab-template IDs. Removing a generated collider-bearing node is rejected
with `collider_reimport_conflict`; explicitly migrate or remove its collider in
source first. Placement-root colliders remain with their placements. Importing
never infers collider geometry from render meshes. This behavior is recorded by
importer provenance version `gltf-pbr-7`.
