# Decision 034: Vertex colors and second texture coordinates

Status: adopted for scene version 8. Meshes may carry vertex colors
(glTF `COLOR_0`) and second texture coordinates (`TEXCOORD_1`), which any
texture map may read; meshes without them keep the compact vertex path of
decision 032.

## Context

The importer rejected vertex colors and every map reading `TEXCOORD_1`,
common in baked occlusion and lightmapped assets. Adding them to every vertex
would undo decision 032: 24-byte vertices and four vertex output slots.

## Decision

- **Optional mesh attributes.** `Mesh::with_colors` (linear RGBA in
  `[0, 1]`) and `Mesh::with_uv1` (finite) take one value per vertex,
  validated as `asset.attribute_count` and `asset.invalid_attribute`.
- **A second stream, only where needed.** Meshes with either attribute
  upload a second 16-byte vertex stream: the color as four 16-bit fractions
  (white without colors) and the second coordinates (the first without
  them). Their draws use extended pipeline variants: `_extended` vertex and
  fragment entry points that read the stream (locations 9 and 10) and pass
  it as two more output slots. Other meshes keep the four-slot path. The
  variant is per draw group, since groups are per mesh.
- **Shading.** Vertex colors multiply the material color and opacity, as
  glTF's `COLOR_0` multiplies the base color; opacity then follows the
  material's alpha mode. `TextureRef::tex_coord` (0 or 1, glTF `texCoord`)
  selects each map's set through a bit per map in the material uniform;
  meshes without second coordinates read their first, and the binding
  includes the selection.
- **Authoring.** Scene version 8 inline vertices take optional `color` and
  `uv1` (every vertex or none); mesh buffers with either are `EMB2`
  (`encode_mesh_buffer_with`: after the counts a flags word, then the
  vertices, colors and second coordinates). Texture map objects take
  `tex_coord`. Earlier versions refuse all three.
- **Importing.** The importer reads `COLOR_0` in any glTF format (the
  reader converts normalized integers and RGB) and `TEXCOORD_1`, writes
  `EMB2` buffers for primitives with them, and maps `texCoord` 1; flat
  normals keep each corner's attributes.

## Observable behavior

- Khronos' VertexColorTest shows its vertex-colored checkmarks matching its
  texture reference row, and MultiUVTest its emissive logo placed by
  `TEXCOORD_1` on every face.
- Meshes without colors or second coordinates draw exactly as before.

## Consequences

Masked shadow casters (decision 031) test their base color through the
first coordinates. Further coordinate sets or color sets are not read.
