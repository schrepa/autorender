# Scene authoring and runtime ownership

Owner: scene format and lifecycle subsystem. Public entry points are
`SceneDocument::{from_json, validate, to_json, instantiate, expanded_entities}` and
`SceneInstance::{entity, entities, despawn}`. The `engine` façade reexports these APIs.
This crate depends on core gameplay/CPU asset contracts and serialization, with
no renderer, browser, filesystem, network or host tooling dependencies.

Version 4 sources contain a stable scene ID, an explicit revision, a camera,
named opaque materials, meshes, textures, reusable prefab definitions, and
entities referencing those assets, and optional explicit colliders. Versions 1–3
remain accepted without colliders; versions 1 and 2 keep their previous flat
renderable semantics. Textures and UV data require version 2 or
later; hierarchies, prefab definitions/instances, and transform-only nodes
require version 3. Advance the source revision when upgrading saved data.
Colliders require version 4. The generated schema is `docs/schema/scene-v4.schema.json`;
older versioned schema artifacts remain unchanged.
Geometry recipes include cube, plane, octahedron and inline indexed triangles.
IDs contain 1–64 ASCII letters, digits, underscores, hyphens or periods. Runtime
entity indices and asset handles never appear in persisted sources. Entity tags
are game-defined names for adding game components once after loading.

`from_json` rejects sources over 1 MiB, unknown fields, unsupported versions,
duplicate IDs, missing references and invalid geometry/transforms/cameras.
Limits are 10,000 expanded entities, 128 prefab definitions, 10,000 total template
entities, 128 meshes, 256 materials, 64 textures and 16 tags per entity.
Names are at most 128 UTF-8 bytes. Translation is bounded to ±1,000,000, Euler
XYZ angles to ±360,000 degrees, and positive scales to `[0.0001, 10000]`.
Inline meshes allow up to 32,768 vertices and 98,304 triangle indices, with
vertex positions bounded to ±1,000,000, finite nonzero normals, and finite UVs.
Opaque materials require alpha 1. Camera FOV is `[1, 179]` degrees; near is at
least 0.001, far is at most 1,000,000 and at least 0.001 beyond near. Position and
target must differ by at least 0.001 units.

An entity either has both `mesh` and `material`, or neither. A transform-only
entity may be a parent, attachment point, or prefab instance root. `parent`
references an entity ID; missing parents, cycles, and more than 64 parent edges
are rejected before world mutation. Local transforms compose as full matrices,
preserving shear from nonuniform parent scales and rotated children. Every
composed matrix must be finite and invertible, with a finite normal transform
and inverse. These checks share the core runtime matrix predicate.

An optional `collider` is independent of mesh/material presence and can appear
on transform-only nodes, prefab template nodes, or prefab placement roots:

```json
{"id":"trigger","collider":{"shape":{"kind":"cuboid","half_extents":[1,0.5,2]},"memberships":2,"filter":1,"sensor":true}}
```

The other shape is `{"kind":"sphere","radius":0.5}`. Local dimensions must
be finite and in `[0.0001, 10000]`. Membership/filter masks are unsigned 32-bit
integers, defaulting to `1` / `4294967295`; sensor defaults to false. Zero masks
are valid and disable matching pairs. Both participants must admit each other's
membership bits. There is no implicit collider derived from rendering geometry,
and no separate offset field: put an offset collider on a child transform.

Full authoritative affine matrices transform the shape: shear produces a
parallelepiped from a cuboid, and nonuniform scale produces an ellipsoid from a
sphere. Collision never uses an interpolated display pose. The collision crate's
shared numerical validation additionally requires center coordinates within
±1,000,000, shape-scaled basis lengths in `[0.0001, 10000]`, and a normalized
basis inverse infinity norm at most 1000. This collision envelope is narrower
than rendering's; valid render geometry can therefore have an invalid collider.
Unused definitions and every placed expanded shape receive these checks before
world or asset mutation. Errors use `invalid_collider` for local shape data and
`invalid_collider_transform` for composed shapes, with source JSON paths.

Instantiation attaches immutable `engine_collision::Collider` components.
Installing `CollisionPlugin`, configuring query budgets, refreshing the spatial
index and consuming sensor events remain gameplay/runtime responsibilities.
Scene disposal removes owned collider entities through the same entity ownership
rules; it does not take ownership of an external collision world or other scenes.

`prefabs` is an optional array of `{ "id": "lantern", "entities": [...] }`
definitions. Each definition uses the scene's existing named assets; it never
owns duplicate mesh, material or texture allocations. Template parents reference
only local entity IDs. Nested prefab references are rejected, and unused
definitions receive the same validation as placed ones. An instance is one
top-level entity such as `{ "id": "west", "prefab": "lantern" }`; it cannot
also have mesh/material fields. Every template root attaches to that instance
root. Template entity IDs and parent references expand to `west.local-id`.
Expanded IDs must remain unique and fit the same 64-byte ID limit; a top-level
entity can explicitly reference an expanded ID as its parent. Instance placement
does not override template fields: edit the definition to change all placements.

`SceneTexture` stores width/height and canonical padded base64 RGBA8 sRGB pixels
in row-major top-to-bottom order. Alpha must be 255. Dimensions are 1..2048;
encoded texture payload shares the complete 1 MiB scene source limit. A
material's optional texture field is an authored ID. Procedural meshes provide
UVs; inline vertices default to zero UV for legacy sources. Sampling is
trilinear and clamp-to-edge over mip levels the renderer generates from the
authored pixels. PNG/GLB decoding belongs to native `tools/assetc`;
base64 decoding belongs only to validation/loading here.

Errors expose stable `code`, JSON `path` and a human-readable `message`.
Parsing, validation and canonical serialization do not mutate a world or save a
file. `to_json` validates, preserves array order, writes stable field order and a
final newline. The caller owns source persistence and revision increments.
`SceneDocument::schema_json` generates structural metadata through the additive
`schema` feature for host tools. Runtime validation additionally checks unique
IDs, references and numeric/budget invariants; schemas do not replace it.

Instantiate between frames with exclusive world access. The world must already
contain an `AssetLibrary`. All CPU data and the complete expanded hierarchy are
validated before insertion; asset or relationship admission failure removes
newly inserted assets/entities and leaves existing entities and the camera
intact. Relationships install in one transactional batch and global transforms
refresh before return. Successful instantiation sets the global camera. CPU allocations
and identifier lookups occur during loading; normal frame work never serializes
or walks authored strings. Loading clones source geometry into immutable CPU
assets, with limits enforced by the owning library.

Retain each returned `SceneInstance` and explicitly call `despawn` in the world
that created it. Disposal removes only that instance's entities and assets,
tolerates entities already removed by gameplay, and leaves the global camera
unchanged. Despawning against another world panics before mutation. Dropping the
instance alone does not dispose resources. Copied handles are weak; outside
entities cannot keep scene assets alive after disposal. Existing render
snapshots can retain immutable mesh/texture data until those snapshots are released.

`SceneInstance::textures` exposes at most 64 authored IDs and weak handles for
development readiness inspection; normal simulation never walks these strings.
`SceneInstance::entities` includes all expanded IDs in lexical order. Use its
checked entity references and `SceneEntity` tags for one-time game setup.
`SceneEntity` records expanded `authored_parent`, `prefab_id`, `prefab_instance`
and `prefab_local_id` alongside identity/name/tags. Instance roots have prefab
and instance IDs but no local ID. These fields describe source; runtime
reparenting is exposed through core `Parent` and does not rewrite authored data.
Despawning a parent never cascades to children, including entities outside this
instance. Core propagation marks descendants with missing parents invalid until
gameplay reparents or disposes them.

For reload, instantiate the new document, then dispose the old instance only
after success. Budgets must temporarily admit both instances. A failed reload
leaves the old scene live. Multiple instances may share the same authored scene
ID; their lookups and runtime resources remain independent. There is no implicit
plugin cleanup, asset hot-reload watcher, editor transport or background loading.
