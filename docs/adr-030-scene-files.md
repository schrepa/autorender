# Decision 030: Scene files

Status: adopted for scene version 7. Scenes reference image and mesh-buffer
files beside them instead of carrying every pixel and vertex inline, and
scene sources have no count budget.

## Context

A scene carried its textures as base64 RGBA8 and its meshes as JSON arrays,
and the whole source had to fit 1 MiB, with further caps of 64 textures,
128 meshes, 256 materials and 10,000 entities. Real assets do not fit: one
2048² texture alone is 16 MiB of pixels.

## Decision

- **Files.** A version 7 texture sets `uri`, a path relative to the scene,
  instead of `width`, `height` and `rgba8`; a mesh source of kind `buffer`
  sets `uri` to an engine mesh buffer (`EMB1`: vertex and index counts, then
  each vertex's position, normal and UV as little-endian f32, then u32
  indices). Paths stay inside the scene's folder.
- **The host reads and decodes.** `SceneDocument::files` lists what a
  document references; the host fetches each file relative to the scene
  and decodes images with the platform's own decoder (in browsers,
  `createImageBitmap`) into RGBA8, supplied as `SceneFiles`
  (`SceneFileSet` in generated bindings, `loadSceneFiles` in the web host).
  `SceneDocument::validate` checks structure alone;
  `validate_with` and `instantiate_with` check and use the files, and a
  missing one fails as `missing_file` before the world changes. Decoding
  happens at load boundaries only; the runtime carries no image decoder.
- **No count budget.** Scene sources may be 64 MiB (a parsing sanity
  bound); textures up to 8,192 texels a side; entity, mesh, material,
  texture, prefab and tag counts are unbounded (decision 029).
- The development server serves `.png`, `.jpg`, `.webp`, `.ktx2`, `.bin`
  and `.evg` (virtual meshes, decision 044) files under `/scenes/` as it
  serves scene sources. KTX2 images
  transcode to compressed textures at load ([decision 037](adr-037-compressed-textures.md)).

## Observable behavior

- A version 7 copy of the playground scene with its texture as a PNG and
  its meshes as buffers renders identically to the original, from 18 KB of
  JSON instead of 96 KB.
- Versions 1 to 6 keep their meaning; inline pixels and meshes still load.

## Consequences

The asset importer still writes inline data; it moves to files with the
glTF material work. Native tools validate file-referencing scenes
structurally; checking the files themselves needs a native decoder.
