# Decision 019: Surface materials and reflections

Status: adopted for every renderer path. Materials describe how their surface
responds to light; captures, previews and CPU presentation shade exactly like
ordinary frames.

## Context

Every surface shaded alike (decision 018): a matte base color and one
Blinn-Phong highlight whose strength came from the atmosphere. Glass towers
looked like painted walls, water like blue carpet, metal like plastic, and
nothing could glow: lit windows were only bright texels, invisible at night.
Base color textures were filtered trilinearly, so walls seen at grazing
angles blurred to the level of their most compressed axis.

## Decision

`Surface`, carried by `Material` (`Material::with_surface`, validated), sets
four values: roughness and metalness in `[0, 1]`, emission as a multiple of
the surface color in `[0, 16]`, and ripples in `[0, 1]`. The default,
`Surface::MATTE`, is roughness 0.8 and nothing else. Cubes are matte.
Extraction copies the surface into `RenderInstance::surface`; a change
advances the slot's color stamp.

The renderer packs a surface into four 8-bit fractions in the fourth word of
each slot's 16-byte color record (`[r, g, b, surface]`), so the record size,
its uploads and its change tracking are unchanged; public instance records
keep their 80-byte stride.

Each fragment:

- takes the sun's highlight from a GGX distribution, Smith visibility and
  Schlick Fresnel with a reflectance of 0.04 for dielectrics and the base
  color for metals, whose diffuse light fades with metalness;
- reflects the analytic sky (decision 018) in the mirror direction, blurred
  toward the hemisphere's average as roughness grows and weighted by Karis's
  fit of the split-sum environment BRDF, so glass and water mirror the sky
  and every surface gains a Fresnel sheen at grazing angles, without
  environment maps;
- emits its color times its emission, times the texture's alpha, times the
  atmosphere's new `emission` scale (lights dim by day and full at night);
- with ripples, tilts its normal by three directional waves that scroll with
  presentation time, which the camera uniform now carries (`position.w`).

Texture alpha is the emission mask. `Texture::new` still requires opaque
alpha everywhere (so a material with emission glows over its whole
surface); `Texture::with_emission_mask` accepts any alpha. Imported and
authored scene textures stay opaque. `Atmosphere::specular` becomes a scale
of highlights and reflections where 1 is physically based, and every preset
uses 1.

Base color sampling is 16x anisotropic: each lookup filters along the
texture's steep axis instead of reading the level of its most compressed one.

Scene version 5 authors a material's `surface` (all fields optional, matte
by default); versions 1 to 4 keep their meaning.

## Observable behavior

- Materials without a surface look nearly as before, with broader, dimmer
  sun highlights and a faint sky sheen at grazing angles.
- The Stress Lab city gives glass towers, metal, vehicles, water, neon and
  lamps their own surfaces, and facade textures mask emission to lit
  windows; the tower field varies roughness and metalness and lights every
  fourth tower's stripes.
- The renderer clamps surfaces of public snapshots into range and reads NaN
  as matte.

## Measurements

None yet: frame costs of this decision, 020 and 021 are to be measured
together with fixed configurations in a later pass.

## Consequences

Materials remain opaque with one texture: there is no per-texel roughness or
metalness and no normal map, so a façade is glossy or matte as a whole.
Reflections show the sky, not the scene: street canyons reflect sky they
cannot see, which ambient occlusion (decision 021) only partly darkens.
Emission lights only its own surface; bloom (decision 021) spreads its glow
on screen, and no surface receives light from another.
