# Decision 033: Per-object tints and material replacement

Status: adopted. Objects take per-object colors through a `Tint` component,
and materials change at runtime in place, keeping their handles.

## Context

Materials were immutable: changing one meant inserting a new material and
swapping every object's handle, and giving objects their own colors meant a
material per color. The Stress Lab city has 132 materials that differ only in
color. Per-object color is common (team colors, damage flashes, selection,
fading), and gameplay changes it every frame.

## Decision

- **`Tint`.** A component of four factors in `[0, 1]` multiplying an
  object's color channel by channel: its material's base color and opacity,
  or its cube's color. Out-of-range factors read as the nearest bound and
  non-finite ones as 1. Opacity follows the usual rules: without maps an
  opacity below 1 draws transparent (decision 026), with maps the alpha mode
  decides (decision 031).
- **Tints are per-object data.** Renderers already carry each object's
  color, so a tint never splits draw groups or bindings, and changing one
  re-resolves and uploads only that object. Retained extraction keys each
  slot on the change ticks of its rendered source (its `Mesh3d`, else its
  `Cube`) and its `Tint`, keeping the per-slot state at 32 bytes; GPU frames
  also compare the number of tints, since removing one changes no tick.
- **`AssetLibrary::replace_material`.** It replaces a live material in
  place, keeping the handle, after checking its textures, and advances the
  library revision. Like any library change, the next extraction
  re-resolves every object: suited to occasional edits, not per-frame color.
  Because draws key on bindings (decision 031), a replacement that changes
  maps moves its objects to the matching binding without other work.
- **Authoring and agents.** Scene version 8 entities take `tint`
  (`[r, g, b, a]`, validated `invalid_tint`); the `Tint` inspector projects
  it in the same shape.

## Observable behavior

- Tints reach owned snapshots, CPU-prepared and GPU-presented retained
  frames when inserted, changed or removed.
- A replaced material draws from the next extraction on both paths, including
  one that gains a texture.

## Consequences

Per-object overrides beyond color (surface, emission) would follow the same
pattern: per-object data, never new bindings.
