# Static GLB source importer

Owner: native asset tooling. Execution is synchronous in the calling host tool;
decoded temporary data belongs to the import call and the returned document to
its caller. Runtime assets and GPU resources have separate owners.

`engine_assetc` is a native tool dependency. `import_glb(bytes, namespace)` returns a validated flat `SceneDocument`; `import_glb_prefab(bytes, namespace)` returns one reusable prefab definition and its global assets with no placements. Both return a structured `{code,path,message}` error without filesystem access. Games load cooked scene data; they never parse glTF or PNG and do not link this crate. Importer version `static-glb-3` writes scene version 4; the static geometry conversion is unchanged. Imports do not infer colliders from visual geometry.

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

This is a deliberately bounded **static diffuse subset of GLB 2.0**, not a general glTF renderer or PBR conversion:

- Exactly one JSON chunk followed by one BIN chunk, one embedded buffer, and one scene. JSON `asset.version` and any `minVersion` must be `2.0`. No external/data URIs, extensions, compression, animations, skins, morph targets, cameras, disconnected nodes or shared/cyclic children.
- Indexed triangles with `POSITION` and `NORMAL` as unnormalized float32 VEC3. Textured primitives require float32 VEC2 `TEXCOORD_0`; other attributes are rejected. Indices may be tightly packed u8/u16/u32. Dense, checked interleaved attributes are supported; sparse/matrix accessors are rejected. Normals must be nonzero and are normalized after transformation.
- Finite affine, nonsingular node matrices or TRS; matrix and TRS cannot coexist, and rotations must be unit quaternions. Hierarchies are baked into positions and inverse-transpose normals. Reflections reverse winding. World-transform determinant magnitude must be at least `1e-8`. Coordinates remain right handed, Y up; there is no unit or axis conversion.
- Explicit opaque, single-sided materials with base color factor alpha 1, **metallicFactor 0** and **roughnessFactor 1**. No emissive factor or textures beyond base color. These restrictions select diffuse input; the engine applies its own ambient/directional lighting and fog, not glTF's PBR BRDF. glTF's omitted metallic factor defaults to 1 and is therefore rejected.
- Embedded 8-bit RGB/RGBA PNG only, dimensions 1–2048, every texel opaque. No palette/grayscale/16-bit PNG, animation or transparency. Base color pixels retain sRGB encoding; material RGB factors remain linear. Embedded PNG color profiles do not alter this transfer function. Image rows and UV `(0,0)` start at the top left.
- Samplers must explicitly specify `CLAMP_TO_EDGE` on both axes. Only `LINEAR` or omitted filters are accepted; the engine uses one mip level. Repeat/nearest/mipmap modes fail explicitly.

The importer uses [Khronos's glTF specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) for transforms, base color encoding and sampler semantics. [gltf 1.4.1](https://docs.rs/gltf/1.4.1/gltf/) handles validated format access; [png 0.17.16](https://docs.rs/png/0.17.16/png/) decodes bounded PNG data. The profile above states additional engine restrictions rather than claiming full standard support.

## Bounds and output

Input is capped at 16 MiB before parsing. JSON/accessor/view/node counts are capped, hierarchy depth is 64, and all buffer ranges, alignment and strides are checked before reader access. At most 1024 nodes, 128 source meshes, 128 baked primitives, 256 materials and 64 images/textures are admitted. Each primitive has at most 32,768 vertices and 98,304 indices. Decoded textures and mesh data share a 32 MiB budget; the PNG decoder also has an explicit internal allocation limit.

Output is a self-contained canonical scene, with embedded base64 RGBA8 texture pixels and inline geometry. The runtime's scene validation and **1 MiB canonical text limit** apply before an import succeeds; that limit usually admits much smaller models than the binary input budget. JSON is an inspectable first cooked representation, not a compressed streaming container. Static mesh reuse across nodes is baked into distinct meshes. Later binary cooking or mip/format support should follow an actual workload.

## Original sample source

`assets/source/garden-prop.glb` contains original procedural geometry and a 64×64 texture atlas: stone/moss planter, carved wood lantern and teal glazed roof. It uses 156 vertices in two primitives, sharing one texture. No third-party source art is included.

```sh
node assets/source/generate-garden-prop.mjs
```

The generator uses only Node built-ins and reproduces the source GLB. Its atlas and geometry are code-native source art. The GLB is retained separately from cooked scene placement so agents can reproduce imports, test replacement, and inspect the actual source format.
