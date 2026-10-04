# Decision 039: Decals

Status: adopted for scene version 8. A `Decal` projects a material onto the
opaque surfaces inside its entity's box: paint, posters and road markings
that cover a surface, or dirt, cracks and stains that darken it.

## Context

Surface detail that does not belong to a mesh (a marking across several
floor tiles, grime at the foot of a wall, a sign on any building) needed
extra geometry placed by hand, or a texture per combination. Engines solve
this with decals: a volume whose contents take a material, whatever meshes
fill it.

## Decision

- **A decal is a component.** `Decal { material, blend, opacity,
  fade_angle }` on any entity with a transform. Its box is the entity's
  local `[-0.5, 0.5]` cube placed by its world pose, so scaling the entity
  sizes the decal. The material's color and opacity, base color map (with
  its sampler and transform), roughness, metalness and emission span the
  box's local x (u) and z (v), projected down its local y onto the surfaces
  facing its +y. Surfaces turning away fade out from `fade_angle` to a right
  angle, so a decal does not smear down the sides of what it covers.
- **Blends.** `DecalBlend::Alpha` covers the surface by its alpha and is lit
  by the sun, sky, shadows, fog and local lights like any surface.
  `DecalBlend::Multiply` scales the surface's shaded color by its own by
  its alpha, keeping the surface's light.
- **Rendering from depth.** After the opaque pass each decal draws its box's
  far faces (36 vertices, instanced, no vertex buffers). Fragments rebuild
  the opaque surface's position from the depth buffer (without the
  temporal jitter, decision 017), keep those inside the box, and take the
  surface's slope from the neighbor pixels whose depths are nearest their
  own, so the edges of other objects neither tilt the normal nor blur the
  map. Every opaque surface in the box receives the decal, still or moving;
  transparent surfaces, particles and the sky do not. Decals draw in
  extraction order, in runs of one material and blend.
- **Poses.** Retained frames list each decal's pose after the skins' joints
  (decision 036): CPU-prepared frames upload its prepared matrix, and
  GPU-presented frames compose its displayed pose as they do slots', so a
  decal on a moving object stays on it. Owned snapshots drew each decal's
  prepared matrix; since 2026-10-03 they copy a frame's decals and joints.
- **Optional, with a budget.** `Decals { enabled, max_decals }` is inserted
  by default and draws every decal; `max_decals` (decision 029) keeps the
  nearest and counts the rest in `dropped_decals`, as do decals beyond the
  GPU's storage binding. Removing the resource or disabling it draws none.
  Frames without decals create nothing and keep the depth buffer
  discardable as before.
- **Authoring and agents.** Scene version 8 entities take `decal`
  (`material` by ID, `blend` `alpha` or `multiply`, `opacity`,
  `fade_degrees`; the last three optional), and the environment a `decals`
  section (`enabled`, `max_decals`) that an environment omitting it resets
  to the default. The `Decal` and `Decals` inspectors project them, the
  `environment` control changes the settings, and `rendering.decals` and
  `rendering.dropped_decals` report a frame's decals.

## Observable behavior

- A marking spans floor tiles and the ground between them; a stain box
  turned 45 degrees covers a wall's foot and the floor in front of it; a
  sign on a wall is lit by a lamp like the wall; a mark on a moving box
  follows it.
- Frames without decals render exactly as before.

## Consequences

Decals read the depth buffer, so they cover what the opaque pass drew:
alpha-masked surfaces receive them where their texels stayed, and glass
shows decals on what is behind it. Decals do not write the glow and
reflection targets, nor the scene target's ambient share; normal,
occlusion and emissive maps of the material are not read. Very distant
decals inherit the depth buffer's precision in their normals. Overlapping
decals blend in extraction order, which follows the ECS query order.
