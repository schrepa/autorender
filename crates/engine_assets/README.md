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

`Texture::new(width, height, rgba8)` owns tightly packed row-major RGBA8 sRGB
pixels, with rows from top to bottom. Width and height must each be in 1..=2048,
the byte length must equal width × height × 4, and every alpha byte must be 255.
Pixels remain encoded as sRGB on the CPU; the renderer decodes them when sampled.
The texture holds one level; the renderer generates its mip chain on the GPU.
This crate does not decode image containers.
`Material::textured(color, texture_handle)` multiplies sampled linear texture
color by an opaque linear color factor. `Material::new` remains the color-only API.

`AssetLibrary` is the sole library owner. Insert assets to obtain typed weak
`MeshHandle`, `TextureHandle`, and `MaterialHandle` values; lookups borrow CPU data.
Runtime handles contain private library identities, slot indices and generations. Stale,
removed, and foreign-library handles fail lookup. Handles are never authored
identifiers or persistent save data. Removing an asset invalidates the handle;
inserting new data requires a new handle. There is no in-place mutation API.
Material insertion rejects texture references that are stale or belong to another
library. Removing a texture leaves any material's weak reference unresolved;
extraction must skip that material's instances. No fallback silently replaces a
missing texture.

The default `AssetBudget` admits 128 meshes, 256 materials, 64 textures and 32 MiB
of resident CPU payload. `AssetLibrary::new` accepts custom limits, including zero. Insertion
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
