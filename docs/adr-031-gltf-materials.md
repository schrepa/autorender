# Decision 031: glTF metallic-roughness materials

Status: adopted for scene version 8. Materials follow glTF 2.0's
metallic-roughness model: texture maps with their own samplers and texture
transforms, alpha modes and double-sided surfaces. Importers map glTF
materials one to one.

## Context

A material was a linear color, one clamped base color texture and a
[`Surface`](adr-019-surface-materials-and-reflections.md) (roughness,
metalness, emission, ripples). Textures could not repeat, nothing could
vary roughness, metalness or normals across a surface, and the importer
rejected every glTF material except a matte diffuse one. glTF's model is the
common language of authoring tools and asset stores, and its factors already
matched the engine's.

## Decision

- **The glTF model.** `Material` keeps its color (glTF's base color factor)
  and surface (roughness and metallic factors, plus the engine's emission and
  ripples) and gains glTF's maps: base color, metallic-roughness (green
  roughness, blue metalness), normal (with a scale), occlusion (red, with a
  strength) and emissive textures, and an emissive color (glTF's emissive
  factor times `KHR_materials_emissive_strength`). Each map is a
  `TextureRef`: a texture, a `Sampler` (wrap modes, magnification,
  minification and mip filters) and a `UvTransform`
  (`KHR_texture_transform`: coordinates scaled, rotated, then offset, as
  Khronos' TextureTransformTest checks). Only `TEXCOORD_0` exists.
- **Alpha modes.** `AlphaMode::Opaque` ignores alpha, `Mask { cutoff }`
  discards surfaces whose alpha (color alpha times base color texture alpha)
  is below the cutoff, and `Blend` draws in the transparent pass (decision
  026). `with_opacity` below 1 still makes an opaque material blend. Opaque
  materials keep the engine extension: base color texture alpha masks
  `Surface::emission`. **Double-sided** materials draw back faces, lit from
  their own side.
- **Colour spaces.** Base color and emissive textures decode sRGB; the others
  read linear data. The renderer uploads a texture in the colour space its
  maps read it in, once per space, with a GPU mip chain averaged in that
  space.
- **Normal maps without tangents.** The fragment shader builds each pixel's
  tangent frame from screen derivatives of its position and of the normal
  map's coordinates, with green pointing up the image as glTF specifies.
  Meshes carry no tangents, imported `TANGENT` attributes are ignored, and
  mirrored texture coordinates work.
- **Occlusion** darkens indirect light (sky, rim and sky reflections) only.
- **Draws key on materials with maps.** Materials without maps
  (`Material::has_maps`) draw from their color and surface as before, batched
  across materials. A material with maps is its own draw group, carried in
  `RenderInstance::material` and listed in `RenderSnapshot::materials` with
  its textures; its maps, samplers and a 192-byte uniform are one bind group.
- **Shader variants, not an uber-shader.** Pipeline constants select what a
  material's shader does: `MATERIAL_MAPS` (the four extra maps, transforms
  and texture alpha coverage), `NORMAL_MAP`, `ALPHA_MASK` (the only variant
  with `discard`) and `DOUBLE_SIDED` (no culling). Materials without maps and
  untransformed opaque base color textures use the unchanged basic variant,
  so plain workloads compile to the same shading. Variant pipelines of each
  pass are created with the first frame that draws them.
- **Scene version 8** authors all of it: `texture` is a texture ID or a map
  object (`id`, `wrap_u`, `wrap_v`, `mag_filter`, `min_filter`,
  `mipmap_filter`, `offset`, `rotation`, `scale`), beside
  `metallic_roughness_texture`, `normal_texture` and `normal_scale`,
  `occlusion_texture` and `occlusion_strength`, `emissive`,
  `emissive_texture`, `alpha_mode`, `alpha_cutoff` and `double_sided`. Plain
  IDs take glTF's defaults (repeat, trilinear) from version 8; earlier
  versions clamp, and `SceneDocument::upgrade` keeps that meaning.
- **Importing.** `cargo xtask asset import` maps every glTF material field,
  sampler and texture transform, writes embedded PNG and JPEG images
  unchanged and meshes as `EMB1` buffers (decision 030) beside the scene,
  named by content hash so reimports never overwrite files an existing
  scene reads, computes flat normals for primitives without normals (as
  glTF requires), ignores and reports optional extensions it does not read,
  and rejects required ones. There is no count, vertex or decoded-size
  budget.

## Observable behavior

- Khronos' DamagedHelmet imports and renders with all five maps; its JPEG
  textures stay files the browser decodes.
- Khronos' TextureTransformTest shows every offset, rotation and scale arrow
  on its "correct" marker.
- A ground with a repeating tile set (color, roughness, normal and occlusion
  maps, `scale` 10) shows lit bevels and rough grooves.
- Scenes of versions 1 to 7 keep their meaning, including clamped textures.

## Consequences

Alpha-masked materials cast solid shadows: the shadow pass does not read
materials yet. Only `TEXCOORD_0` is supported; a second texture coordinate
set, vertex colours and texture compression remain future work. Normal maps
baked against MikkTSpace tangents can differ slightly from the derivative
frame on strongly distorted UV layouts.
