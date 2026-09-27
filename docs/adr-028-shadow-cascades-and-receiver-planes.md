# Decision 028: Shadow cascades and receiver planes

Status: adopted for every renderer path. `Shadows::cascades` (3 by default)
splits the near map into cascades, and surface lookups compare against the
receiving surface's own plane instead of a margin that grows with texel
size.

## Context

Shadows of small objects near the camera looked dented and triangular, and
their shapes changed as they moved away:

- One near map (half the far resolution) covered the whole near distance:
  in the Stress Lab about 0.11 world units per texel, so a car or a boat
  cabin covered a dozen texels up close, which filtering turned into stair
  steps with diagonal ramps.
- To keep tilted surfaces from shadowing themselves, every lookup moved 1.5
  texels along the surface normal and the maps added a slope-scaled depth
  bias of 1.5. Both grow with texel size, so the lowest part of every caster
  stopped shadowing what it stood on: about 0.2 units within the near map
  and 0.75 units in the far maps, all of a boat's hull.

## Decision

- **Cascades.** The near distance is split into `cascades` maps (1 to
  `MAX_SHADOW_CASCADES`, 4): each reaches a third as far as the next, so it
  has three times its texel density. Each is the smallest sphere around the
  view slice within its reach, layers of one depth texture, redrawn every
  frame from the casters it covers. Lookups take the first cascade that
  reaches a fragment and blend into the next one across the last tenth of
  its reach.
- **Receiver planes.** A surface lookup gathers the four texels around it
  and compares each with the surface's own plane at that texel's center,
  derived from how the surface's position changes per pixel. Tilted surfaces
  match their own depths, so the normal offset drops to half a texel and the
  maps keep one unit of constant depth bias; sun shafts, which march through
  the air, keep plain comparisons.
- **Authoring and agents.** Scene version 6 authors `shadows.cascades`
  (earlier versions refuse it); `Shadows` inspects and changes through the
  `environment` control.

## Observable behavior

- In the Stress Lab city at golden hour, cars and boat cabins within 30
  units cast crisp shadows, masts cast lines, and shadows touch their
  casters at every distance.
- `cascades: 1` restores one near map, for comparison.

## Measurements

None yet. Each extra cascade is a depth pass over the casters it covers: in
the Stress Lab city, the cascades reaching 10 and 30 units hold hundreds of
casters against a few thousand in the one reaching 90. Maps cost 8 MiB per
cascade at the Stress Lab's resolution.

## Consequences

Resolution still falls off beyond the near distance: the far maps cover the
whole shadow region, about 0.45 units per texel over the Stress Lab city, so
casters under a unit tall (cars, boat cabins) cast shadows too small for a
texel there. A camera-following cached level between the near cascades and
the far maps would close that gap. Where a lookup's texels meet another
surface (a wall's foot, a roof edge), a texel's width may still darken.
