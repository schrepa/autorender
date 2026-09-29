# Decision 048: Octahedral impostors

Status: adopted for GPU-presented frames, on with levels of detail by
default. `LevelOfDetail::impostors` (scene version 8 environment
`level_of_detail.impostors`, Stress Lab `?impostors=off`) turns them off,
and the optional `max_impostors` budgets them.

## Context

Levels of detail (decision 047) shrink meshes whose triangles can collapse.
At a few pixels even a coarse level still costs its vertices and a cloud of
tiny triangles, each of whose 2x2 pixel quads shades pixels it does not
cover. Games replace such instances with impostors: cards showing the mesh
as it was rendered beforehand from around the view direction.

The established form is the octahedral impostor, used by Unreal's impostor
baker, Amplify Impostors and, for vegetation, in Fortnite and Horizon:
- frames are rendered from the grid points of an octahedral map of view
  directions;
- a card blends the three frames of the grid triangle holding the view, by
  barycentric weights, each at the point the view projects onto the frame's
  plane;
- frames store surface attributes (albedo, normal, depth, material), so the
  card is lit at runtime like the meshes around it.

## Decision

### Which draw groups

An impostor is baked for every opaque, non-skinned draw group:
- whose mesh has at least 64 triangles (`IMPOSTOR_MIN_TRIANGLES`);
- when GPU-presented frames draw levels of detail with impostors on.

Virtual geometry (decision 044) qualifies and is baked from its fallback
cut. Its camera clusters are selected only for instances not drawn as
impostors, because those never reach the level-0 visible IDs that
selection reads.

A draw group's mesh and material key its impostor. A material replaced in
place (decision 033) is rebaked.

### Baking

The first frame a group draws, its mesh is rendered with its material into
one layer of four RGBA8 texture arrays: 128x128 texels (16 per frame) with
four levels, recorded after the frame's new texture levels.
- **Views:** 8x8 frames sit on the grid points of an octahedral map of the
  whole sphere about the mesh's +y, edges and corners included. Each frame
  is an orthographic view that fills its cell with the mesh's bounding
  sphere; one draw of 64 instances renders them all.
- **Channels:**
  - albedo and coverage;
  - emissive color and emission mask, both sRGB;
  - normal and depth;
  - roughness and metalness factors, and occlusion.
- **Coverage:** frames rasterize four samples per texel. The resolve
  averages them with the cleared zeros, so every channel is premultiplied
  by coverage.
- **Levels:** each level averages 2x2 blocks of the one above. Cells halve
  with the levels, so frames never mix.
- **Material:** textures, normal maps and vertex colors are sampled as a
  fragment would. The slot's color, surface and tint multiply at draw time.

### Selection

An impostor's error is one texel of its first level, 2r/16 in mesh units.
The culling pass tests it like a mesh level (decision 047): the impostor
draws where that error projects within the pixel error. It is level 3, and
where it fits it wins over the mesh's own levels, since it is the cheapest.
Frames draw impostors only when their settings allow; the frame uniform's
`eye.w` carries that.

### Drawing

Every group's impostor instances compact into one list of visible IDs and
draw groups. Each drawing phase draws that list with one non-indexed
indirect draw, six vertices per instance.

The vertex shader places each card:
- The mesh-space direction toward the viewer picks the grid triangle around
  it: three frames and their weights.
- The card is the bounds' outline seen along that direction, placed in
  front of the bounds so the ground never clips it.
- Each frame's coordinates come from projecting the card point along the
  view onto that frame's plane through the center.
- On temporal frames, a card corner's previous position comes from the
  slot's previous pose.

Fragments work in this order:
1. Sample the three frames' coverage at the level matching their texel
   footprint, clamped half a texel of the coarser level inside each frame's
   cell.
2. Stop where the blended coverage is under a half: most of a card is empty.
3. Read and unpremultiply the other channels.
4. Shade with the forward shading functions. The normal goes through the
   pose's cofactors, the surface is rebuilt at the blended depth, and
   emission, occlusion and factors come from their texels.

Normals are stored as plain vectors: they filter and average across levels,
which octahedral normals cannot where they fold.

Impostor instances cast shadows as their meshes, since shadow maps pick
their own levels. Receiving is approximate:
- Blended frames measure depth along their own views, so the rebuilt
  surface can sit a few tenths of the radius off the mesh that casts its
  shadow.
- The sun is therefore looked up a bounding radius toward it, with a flat
  receiver plane.
- An impostor keeps the shadows of other casters and loses its own on
  concave parts, a few pixels at 16 or fewer.

### Budget

`max_impostors` (none by default) and the GPU's texture array layers bound
the baked impostors. Draw groups beyond either draw their meshes and are
reported as dropped. Lowering the budget below the baked count drops them
all, and the draw groups first in the plan bake again up to it.

## Observable behavior

- `rendering.impostors` counts the impostors baked, `dropped_impostors` the
  groups left drawing their meshes, and `impostor_instances` (read back a
  few frames late) the instances drawn as impostors.
- Profiles count `render.impostors`, `render.dropped_impostors` and
  `render.impostor_instances`, and `memory.gpu_impostor_bytes` the arrays.
- `web/scripts/lod-smoke.mjs` checks that:
  - impostors bake and draw from the Metro overview;
  - the frame stays within 5% of the exact frame's pixels (3.2% measured,
    and under 1% beyond a difference of 64 levels), and within 1% without
    impostors;
  - a budget of 2 keeps 2 and reports the rest dropped.

Measurements (Firefox on the laptop's integrated GPU, one build, three
alternating runs each):

| Metro overview | fps (mean) | Opaque pass |
|---|---|---|
| Levels of detail off | 50.7 | 9.60 ms |
| Levels, no impostors | 56.6 | 7.45 ms |
| Levels and impostors | 58.0 | 6.93 ms |

- **Metro overview:** 104 impostors (44.8 MB) and 14,570 instances drawn as
  impostors, which take over from nearly every mesh level (6 instances stay
  at level 1). In an earlier build (opaque pass 7.46 ms without impostors),
  skipping those instances entirely, the bound for any impostor, measured
  6.21 ms and 57.6 fps.
- **Dense scene:** 527 of 1,190 visible instances drawn as impostors;
  virtual triangles 483,414 to 434,738. Throughput unchanged: 67.7 fps off,
  67.6 fps on. Virtual geometry already draws distant boulders with a few
  clusters.
- **Loading:** in Chrome the frame gaps of Metro's first 15 seconds show no
  added hitch; the first frames' scene construction and pipeline compiles
  dominate them either way.

## Consequences

- An impostor costs about 340 KB of GPU memory with its levels, and arrays
  grow by doubling.
- Bakes happen when groups first draw, so the first frame of a scene
  records them all (78 to 104 on Metro).
- Kept shadow maps are redrawn whole when their casters' levels change,
  which happens when levels of detail turn on or off (decision 047).
- Impostors pay off where an instance's triangles far outnumber its pixels:
  imported models of thousands of triangles repeated many times. Metro's
  props mostly have tens to a few hundred, so there even free impostors
  would save 1.25 ms.
- Objects under the size where the impostor switches in show small shifts
  of up to a texel, and slightly different lighting on concave parts.

## Alternatives and evidence

- **Octahedral normals** (tried first). Filtering and levels averaged
  normals across the folds of the octahedral lower hemisphere, turning the
  tumbled boulders of the dense scene the color of shade. Vectors replaced
  them.
- **Surface-depth shadow lookups** (tried first). The rebuilt surfaces of
  the dense scene's boulders fell into their own shadows.
- **One dominant frame** instead of three. It was 0.2 ms cheaper on Metro
  but pops between frames as views turn. Not adopted.
- **Discarding after shading** (the first version). Sampling coverage
  first and stopping on empty fragments saved 0.47 ms instead of 0.38 ms.
- **Hemi-octahedral frames.** They would double the angular resolution of
  views from above but show nothing from below, and tumbled or flying
  objects need every direction.
- **Writing fragment depth** from the baked depth. That would correct how
  impostors hide one another, but disables early depth tests. Not done.
- **Comparing captures taken apart.** Reflection probes (decision 043)
  capture their cubes over many frames, so captures taken seconds apart can
  differ. `lod-smoke.mjs` turns probes off and waits for read-back counts
  to hold steady before comparing.
