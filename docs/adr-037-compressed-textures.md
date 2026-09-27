# Decision 037: Compressed textures

Status: adopted for scene version 8. Scenes reference KTX2 images with Basis
Universal supercompression (glTF `KHR_texture_basisu`); hosts transcode them
at load to a block-compressed format the GPU samples, and the renderer
uploads their levels as they are.

## Context

Textures stayed RGBA8 on the GPU: four bytes per texel, a quarter to an
eighth of which block compression needs, and every level generated at
upload. Large textured models (Khronos' FlightHelmet has fifteen maps up to
2048²) cost hundreds of megabytes of GPU memory. GPUs sample different block
formats: BC on desktops, ASTC or ETC2 on phones. KTX2 with Basis Universal
is the portable glTF answer: one file that transcodes cheaply to any of
them. Decision 030 keeps decoders out of the frame loop and in the host.

## Decision

- **Assets.** `TextureFormat` names how a texture's texels are stored:
  `Rgba8` (one level; renderers generate the rest) or the 4 × 4 block
  formats `Bc7`, `Etc2Rgba8` and `Astc4x4` (16 bytes per block).
  `Texture::compressed` takes a compressed texture's levels from the base
  down: dimensions multiples of 4, one level up to a full chain, each
  exactly its whole blocks (`asset.invalid_compressed_texture`,
  `asset.invalid_texture_length`). Compressed textures keep alpha, as
  version 8 textures do.
- **Hosts transcode at load.** A scene texture's `uri` may name a KTX2 file
  (`.ktx2`, recognized by its identifier). The web host fetches it with the
  scene's other files and transcodes it with the vendored upstream Basis
  Universal transcoder (`web/public/basis`, loaded only for scenes with KTX2
  images), on the main thread at load, never per frame, to the first format
  the device's adapter samples: BC7, then ASTC 4 × 4, then ETC2; to RGBA8
  (base level) without one or for dimensions that are not multiples of 4.
  `SceneFiles::add_compressed_image` (`SceneFileSet.add_compressed_image`)
  supplies the result. Compressed images require scene version 8.
- **The renderer uploads levels as they are.** It requests every texture
  compression feature the adapter has, creates compressed textures in the
  matching sRGB or linear format per use, and writes each level; it
  generates no levels for them. A texture in a format the device lacks
  samples as white rather than failing the frame.
- **Importing.** The importer reads `KHR_texture_basisu`: a texture's KTX2
  image replaces its source, and KTX2 images are written beside the scene
  unchanged (`ID/imageN.HASH.ktx2`), checked only for the KTX2 identifier.

## Observable behavior

- FlightHelmet's KTX2 variant (ETC1S base colors, UASTC normal and
  occlusion-roughness-metal maps) renders as its PNG variant does, with its
  fifteen maps resident as BC7 on desktop GPUs, about a quarter of their
  RGBA8 size.
- `resident_texture_bytes` counts compressed levels as uploaded.

## Consequences

Only Basis Universal KTX2 files transcode; KTX2 files already in a GPU
format, cube maps and arrays are refused. Transcoding runs on the main
thread at load (tens of milliseconds for a 2048² map); a worker would be the
next step for scenes with many large maps. The importer does not encode
KTX2 from PNG or JPEG; tools such as `gltf-transform` with KTX-Software do.
