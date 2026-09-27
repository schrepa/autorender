# Decision 020: Sun shadows

Status: adopted for every renderer path, on by default (`Shadows`).
GPU-presented frames keep the shadows of still casters between frames;
exact captures, previews and CPU presentation draw every caster each frame
and shade alike. Decision 022 makes the resource optional (missing means
off), creates the shadow pipelines only once shadows run, and exposes the
settings to scenes and agents. Decision 028 splits the near map into
cascades and replaces the lookup biases with receiver planes.

## Context

Nothing cast shadows: buildings looked pasted onto the ground and a low sun
lit every street. Drawing every caster from the sun each frame would cost
about another opaque pass, the pass that already bounds large scenes
(decision 007 measurements), and most renderables in large scenes never
move.

## Decision

`Shadows`, an engine resource inserted by default, enables shadows of the
atmosphere's sun and sets their region: `bounds` (a world box the scene fits
in) or, without them, a cube of `distance` around an anchor that follows the
camera a quarter of the distance at a time; the distance where shadows fade
out; the view distance of the near map; and the texel count of the far
maps (a power of two in 1024 to 4096, 2048 by default). Snapshots carry it
sanitized.

The renderer keeps three 16-bit depth maps seen along the sun direction,
flattening casters in front of a map onto its near plane:

- the near map, half the far resolution, covers the smallest sphere around
  the view within the near distance, moved in whole texels, and is redrawn
  every frame;
- the kept far map covers the region and holds casters whose pose has not
  changed for 8 frames (180 once a kept caster moved again: casters that
  pause stay in the moving map instead of redrawing kept strips at every
  pause). It is redrawn whole when the sun, the region or the
  resolution changes, and otherwise only where it is dirty: one of 16
  horizontal strips per frame, cleared under a scissor;
- the moving far map covers the region too and holds every other caster
  (and kept ones for 64 frames, until the kept map surely holds them). It is
  redrawn every frame.

A compute pass after GPU presentation's culling visits every candidate,
visible or culled, reads the pose culling wrote, transforms its mesh bounds
into light space, and appends its slot to the instance IDs of each map it
can darken, counting indirect draws per mesh (up to 128 meshes). A caster
can only darken receivers under it along the sun: each near cascade and the
moving map select casters over the part of their region where visible
receivers can be, the light-space bounds of the view frustum up to the
cascade's reach (or the shadow distance, or the sun shafts' when longer),
widened by the lookups' normal offset and texel gathers. Maps keep their
regions, so selection changes no texel; the kept map selects from its whole
region, which every view reuses. On
GPU-presented frames it also keeps, per slot, a 32-bit key of the pose and
mesh, a stillness count and the kept strips the slot was last drawn into; a
kept slot that moves, or one that becomes still, marks strips dirty in a
buffer read back a few frames later without waiting. Removing candidates
marks every strip dirty. Fragments look up the near map within its distance
and otherwise the far maps (the kept one only on GPU-presented frames), with
filtered comparisons, a slope-scaled depth bias and a normal offset of 1.5
texels. Shadows darken the sun's diffuse light and highlight only; sky light
stays.

CPU-prepared retained frames list every caster, culled or not
(`RenderFrame::casters`), and draw them all into the near and moving maps.
Public snapshots draw their own instances, which are visible ones.

## Observable behavior

- Every renderable casts and receives sun shadows; a still scene costs, per
  frame, the selection pass, the near map and the moving casters.
- GPU and CPU captures of the same state match: the GPU-presentation smoke
  check compares both with every renderable moving and with none.
- A kept caster that moves or disappears leaves a stale shadow until its
  strips are redrawn, a few frames later.
- The Stress Lab bounds each scene and uses 4096 texels; its Effects control
  (and `?effects=`) turns shadows off.

## Measurements

None yet: to be measured with decisions 019 and 021. Memory at the default
2048 texels is 18 MiB of maps; at 4096, 72 MiB.

## Consequences

Changing the sun redraws the kept map whole each time: animating it every
frame costs a full static caster pass per frame. A fixed region is sharpest
when `bounds` fit the scene; without them the map follows the camera and
spends resolution behind it. Public snapshots lack shadows of casters
outside the view. Only one directional light casts shadows; lamps and
emissive surfaces do not.
