# Decision 038: Glass and refraction

Status: adopted for scene version 8. Materials may let light through
(glTF `KHR_materials_transmission`, `KHR_materials_ior` and the thickness of
`KHR_materials_volume`). With the opt-in `Refraction` resource, transmissive
surfaces show the opaque scene behind them, bent and blurred.

## Context

Transparency blended surfaces over what is behind them (decision 026): good
for smoke and tinted panes, but glass looked like faded plastic. It neither
bent what showed through it nor frosted it, colored glass could not tint the
light it passed, and imported glass (Khronos' FlightHelmet goggles) was
opaque because the importer ignored `KHR_materials_transmission`.

## Decision

- **Transmission is material data.** `Material::with_transmission` takes a
  `Transmission`: the share of light passing through in `[0, 1]` (tinted by
  the base color), an index of refraction in `[1, 3]` and the thickness in
  world units of the volume beneath the surface (0 is a thin wall).
  Transmissive materials draw in the transparent pass and bind their own
  parameters; their pipeline variant (`TRANSMISSION`) shades glass. Glass
  diffuses only the light it does not transmit, and keeps its reflections
  and highlights.
- **Refraction is an opt-in effect.** `Refraction { strength, blur }`
  (missing by default, like sun shafts and reflections): glass samples the
  frame's opaque color where its ray leaves the volume, the view ray bent
  by its index of refraction through its thickness times `strength`, in
  up to nine taps spread by its roughness times `blur` pixels, tinted by
  its color, over which its own light adds. The transparent pass reads the
  scene target its accumulation does not write; frames drawing glass with
  refraction render into that target even without other screen effects.
  Without the resource, or with strength 0, glass blends as plain
  transparency by the share it transmits.
- **Authoring and agents.** Scene version 8 materials take
  `transmission`, `ior` and `thickness` (the last two need the first); the
  environment's `refraction` section (version 8) sets the resource, which
  omitting it removes. The `Refraction` inspector projects it, the
  `environment` control changes it, and `rendering.refraction` reports
  whether a frame refracted.
- **Importing.** `KHR_materials_transmission`'s factor, `KHR_materials_ior`
  and `KHR_materials_volume`'s thickness factor map to the material; the
  thickness counts as world units and transmission textures are not read.

## Observable behavior

- FlightHelmet's goggle lenses show the face behind them; clear blocks bend
  what is behind them at their edges, frosted ones blur it and amber glass
  tints it.
- Scenes without transmissive materials, or without `Refraction`, render
  exactly as before and pay nothing.

## Consequences

Glass refracts the opaque scene only: other transparent surfaces and
particles behind it do not show bent, and stacked glass layers blend by
weight (decision 026). Refraction samples the screen, so rays leaving it
clamp at its edges. Thickness does not follow an object's scale.
