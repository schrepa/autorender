# Static GLB source importer

Owner: native asset tooling. Execution is synchronous in the calling host tool;
temporary data belongs to the import call and the returned document and files
to its caller. Runtime assets and GPU resources have separate owners.

`engine_assetc` is a native tool dependency. `import_glb(bytes, namespace)` returns an `Imported`: a validated flat `SceneDocument`, the image and mesh-buffer files it references (to write beside the scene) and the optional extensions it ignored; `import_glb_prefab(bytes, namespace)` returns one reusable prefab definition and its global assets with no placements. Both return a structured `{code,path,message}` error without filesystem access. Games load cooked scene data; they never parse glTF and do not link this crate; browsers decode the image files. Importer version `gltf-pbr-5` writes scene version 8 with glTF 2.0 metallic-roughness materials ([decision 031](../../docs/adr-031-gltf-materials.md)), animation clips and skins ([decisions 035](../../docs/adr-035-animation-clips-and-players.md) and [036](../../docs/adr-036-skinning.md)). Imports do not infer colliders from visual geometry.

Reimport with `--replace` preserves authored colliders on surviving deterministic
flat entity IDs and prefab template local IDs. Placement-root colliders remain
part of the preserved placement records. Removing an entity that has
an authored collider returns `collider_reimport_conflict`; explicitly remove or
relocate that collider in source first. Changed geometry may require adjusting
the preserved collider manually. The complete merged scene still passes v4 local
and composed-shape validation before a revision-checked save.

```sh
cargo xtask asset import --file assets/source/garden-prop.glb --id garden --output artifacts/garden.scene.json
cargo xtask asset import --file assets/source/garden-prop.glb --id scenery --into web/public/scenes/playground.scene.json --expected-revision N
```

The first command creates a new file and refuses to overwrite. Parent directories must already exist. The second appends into an existing scene through the same validated revision check and atomic save used by `cargo xtask scene save`; it retains the camera and other game objects, increments the revision, and upgrades legacy scenes to the current format. Importing alone does not reload a running game.

To replace a previously imported namespace, supply **`--replace`** together with `--into` and the current `--expected-revision`. This explicitly authorizes replacing generated `namespace.imageN`, `namespace.materialN`, `namespace.nodeN.pN.mesh`, and (in flat mode) `namespace.nodeN.pN` entries. Stable flat entity IDs retain their authored transform, parent, name, and tags. Other entries remain unchanged. Namespace collisions without this flag fail validation. A replacement that leaves dangling references also fails without saving. Keep a namespace reserved for one source model; IDs follow source node/primitive indices, so reorganizing an exporter can change them. This is explicit namespace replacement, not dependency tracking or a persistent import manifest.

Every CLI response uses a versioned JSON envelope. Success reports imported counts, saved path/revision/bytes and provenance: the exact input SHA-256, importer version, namespace, source path and fixed settings, including `authoring: "flat"` or `"prefab"`. Counts distinguish top-level `entities`, `prefabs`, and `prefab_entities`. Capture this response in your build records when provenance must be retained. Source provenance is not embedded in the scene and is not an authenticity check.

## Import once and place several instances

```sh
cargo xtask asset import --file assets/source/garden-prop.glb --id shrine --as-prefab --into web/public/scenes/playground.scene.json --expected-revision N
```

`--as-prefab` creates a definition named `shrine`, with local children such as `node1.p0`, while retaining global asset IDs such as `shrine.node1.p0.mesh`. It adds **no placement entity**. Add a top-level scene entry for each instance, then validate/save/reload through the usual authoring commands:

```json
{"id":"west","prefab":"shrine","transform":{"translation":[-5,0,-4]}}
```

The placement root groups its children, which receive qualified IDs such as `west.node1.p0`; another root uses the same mesh, material and texture assets. Static glTF node transforms still bake into vertices. This feature groups the imported parts for placement; it does not preserve the glTF hierarchy or enable animated source assets.

Use `--as-prefab --replace --into TARGET --expected-revision N` to update the definition and generated assets. All placement roots referencing the definition retain their transforms, parents, tags and names because the importer does not rewrite them. Template children are replaced from the new source. The complete merged scene, including prefab expansion and external references, must validate before saving. A removed primitive still referenced by another entity causes rejection and leaves the file unchanged.

Flat and prefab namespaces are distinct authoring modes. Replacing a flat namespace as a prefab, replacing a prefab as flat, or taking over generated assets with no recognizable definition/flat entities returns `namespace_mode_conflict`. `--replace` does not authorize those conversions: use a new namespace or prepare an explicit validated scene migration. A similarly prefixed namespace such as `shrine2` remains independent. Prefabs cannot contain nested prefab instances.

## Supported source profile

A **subset of GLB 2.0 with glTF's metallic-roughness materials, animations and skins**:

- Exactly one JSON chunk followed by one BIN chunk, one embedded buffer, and one scene. JSON `asset.version` and any `minVersion` must be `2.0`. No external/data URIs, morph targets, disconnected nodes or shared/cyclic children. Cameras are ignored.
- Indexed or unindexed triangles with unnormalized float32 VEC3 `POSITION` and optional `NORMAL`; primitives without normals get flat normals, as glTF requires. Textured primitives require float32 VEC2 `TEXCOORD_0`; `TEXCOORD_1` and `COLOR_0` (any glTF format) are read into `EMB2` buffers (decision 034), and so are `JOINTS_0` and `WEIGHTS_0` of skinned nodes (decision 036). `TANGENT` and further coordinate or color sets are ignored (normal maps derive their tangent frames); further joint sets and other attributes are rejected. Indices may be tightly packed u8/u16/u32. Dense, checked interleaved attributes are supported; sparse, MAT2 and MAT3 accessors are rejected. Normals must be nonzero and are normalized after transformation.
- Finite affine, nonsingular node matrices or TRS; matrix and TRS cannot coexist, and rotations must be unit quaternions. Static models bake hierarchies into positions and inverse-transpose normals; reflections reverse winding, and world-transform determinant magnitude must be at least `1e-8`. Models with animations or skins keep their nodes as entities with local transforms (decision 035), which must be positive-scale rotations, scales and translations; skinned vertices stay in bind space. Coordinates remain right handed, Y up; there is no unit or axis conversion.
- Animations become clips in `EAN1` files: translation, rotation and scale channels with step, linear or cubic spline samplers (float times; float or normalized rotations); repeated key times are dropped. Skins keep their joints and inverse bind matrices.
- Materials map one to one: base color factor and texture, metallic and roughness factors and texture, normal texture and scale, occlusion texture and strength, emissive factor (times `KHR_materials_emissive_strength`) and texture, `OPAQUE`/`MASK`/`BLEND` alpha modes with the cutoff, and double-sided materials. Every sampler wrap mode and filter maps to the texture reference; `KHR_texture_transform` offsets, rotations and scales too. Maps read `TEXCOORD_0` or `TEXCOORD_1`.
- Embedded PNG and JPEG images are written unchanged, named by content hash under the namespace's folder; the browser decodes them. Meshes are written as `EMB1` or `EMB2` buffers and clips as `EAN1` buffers the same way.
- Required extensions other than `KHR_texture_transform` and `KHR_materials_emissive_strength` are rejected; other optional extensions are ignored and reported.

The importer uses [Khronos's glTF specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) for transforms, materials, color encoding and sampler semantics. [gltf 1.4.1](https://docs.rs/gltf/1.4.1/gltf/) handles validated format access.

## Bounds and output

Input is bounded by the GLB length field (4 GiB); hierarchy depth is 64, and all buffer ranges, alignment and strides are checked before reader access. There is no count, vertex or decoded-size budget (decision 029).

Output is a canonical scene referencing its files (decision 030): textures by `uri`, meshes as `buffer` sources. Static mesh reuse across nodes is baked into distinct meshes.

## Original sample source

`assets/source/garden-prop.glb` contains original procedural geometry and a 64×64 texture atlas: stone/moss planter, carved wood lantern and teal glazed roof. It uses 156 vertices in two primitives, sharing one texture. No third-party source art is included.

```sh
node assets/source/generate-garden-prop.mjs
```

The generator uses only Node built-ins and reproduces the source GLB. Its atlas and geometry are code-native source art. The GLB is retained separately from cooked scene placement so agents can reproduce imports, test replacement, and inspect the actual source format.
