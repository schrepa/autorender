# engine_assets

Owner: engine maintainers. This crate owns immutable CPU meshes, opaque sRGB
textures, linear color materials, validation, and bounded storage. Its dependencies
are `glam` and Bevy ECS's `Resource` contract. It imports no GPU or browser implementation.

`Mesh::new(vertices, indices)` validates indexed triangles, finite positions,
finite nonzero normals and finite UV coordinates. It normalizes normals and takes
ownership of exact-sized vertex/index allocations. `Mesh::cube`, `plane`, and `octahedron` provide centered
unit geometry with UV coordinates; the plane faces +Y. UV (0, 0) is the upper-left
image corner; finite UVs outside [0, 1] remain valid. `Material::new` accepts finite
linear color channels in [0, 1] with alpha exactly 1. Geometry uses counterclockwise triangles;
the caller remains responsible for winding and intentional degenerate geometry.

`Mesh::bounds()` returns immutable `Aabb` metadata with local `min()` and `max()`
endpoints. Bounds are computed once during validation from every vertex, including
unused vertices, so reading them never scans geometry. Finite endpoint storage
avoids overflowing center/extent calculations for meshes spanning extreme finite
coordinates. The bounds add no separate allocation and are excluded from mesh
payload bytes; they support conservative extraction culling in `engine_core`.

`Texture::new(width, height, rgba8)` owns tightly packed row-major RGBA8
pixels, with rows from top to bottom. Width and height must each be in
1..=8192, the byte length must equal width × height × 4, and every alpha byte
must be 255 (`Texture::with_emission_mask` takes any alpha). Pixels stay
encoded on the CPU; the renderer decodes color maps as sRGB and data maps
(metallic-roughness, normal, occlusion) as linear when sampled. The texture
holds one level; the renderer generates its mip chain on the GPU.
`Texture::compressed` instead holds block-compressed levels (`TextureFormat`
`Bc7`, `Etc2Rgba8` or `Astc4x4`, [decision 037](../../docs/adr-037-compressed-textures.md))
that renderers upload as they are. This crate does not decode or transcode
image containers.

`Material` is a glTF 2.0 metallic-roughness material
([decision 031](../../docs/adr-031-gltf-materials.md)): a linear color factor,
a `Surface` (roughness and metallic factors plus emission and ripples) and
optional base color, metallic-roughness, normal, occlusion and emissive maps.
Each map is a `TextureRef` with a `Sampler` (wrap modes and filters; glTF's
repeating trilinear default) and a `UvTransform`. `AlphaMode` chooses opaque,
masked or blended alpha, and `with_double_sided` draws back faces.
`Material::textured(color, texture_handle)` keeps its clamped base color
texture; `Material::new` remains the color-only API. `has_maps` tells
renderers which materials need their own maps bound.

Meshes take optional per-vertex streams: `with_colors` and `with_uv1`
([decision 034](../../docs/adr-034-vertex-colors-and-second-coordinates.md))
and `with_skin`, four joints and weights for skins
([decision 036](../../docs/adr-036-skinning.md)). An `AnimationClip`
([decision 035](../../docs/adr-035-animation-clips-and-players.md)) holds
keyframed translation, rotation and scale channels of target indices, sampled
with glTF's step, linear or cubic spline interpolation; `insert_clip` returns
a `ClipHandle`.

`AssetLibrary` is the sole library owner. Insert assets to obtain typed weak
`MeshHandle`, `TextureHandle`, and `MaterialHandle` values; lookups borrow CPU data.
Runtime handles contain private library identities, slot indices and generations. Stale,
removed, and foreign-library handles fail lookup. Handles are never authored
identifiers or persistent save data. Removing an asset invalidates the handle;
inserting new data requires a new handle. Only materials change in place
(`replace_material`, [decision 033](../../docs/adr-033-tints-and-material-replacement.md)).
Material insertion rejects texture references that are stale or belong to another
library. Removing a texture leaves any material's weak reference unresolved;
extraction must skip that material's instances. No fallback silently replaces a
missing texture.

The default `AssetBudget` admits everything, so only memory bounds a library
([decision 029](../../docs/adr-029-budgets-and-device-limits.md)).
`AssetLibrary::new` and `AssetLibrary::set_budget` accept custom limits, including zero. Insertion
fails without admission on capacity or byte overflow. Counters cover live assets
and vertex/index/material-record/pixel payload, excluding bounds/allocator/slot metadata
and asset data retained outside the library. `AssetError` provides stable `code()` values
and contextual display messages.
`texture_bytes()` reports the pixel portion already included in `resident_bytes()`.
Both counts and bytes change only after successful admission or removal.

Meshes and textures use `Arc` so render snapshots retain their source after explicit
removal or library disposal. These external references extend CPU lifetime and are not
charged to the library after removal. Keep the number of in-flight snapshots
bounded; the current runtime reuses one snapshot. No data copying, reference
counting per vertex or per entity, browser calls, or background jobs occur here.
Execution is synchronous in the world's owning context. Assets are immediately
CPU-ready; there are no placeholder asynchronous states.

Native tests cover mesh, UV, texture and material validation, primitive winding
and UV area, stale and foreign handles, exhausted generations, byte/capacity
admission, unresolved material dependencies, and source lifetime after disposal.
Bounds tests include unused vertices, degenerate axes, primitive dimensions,
extreme finite coordinates, and unchanged payload accounting.
