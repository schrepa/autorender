# Decision 036: Skinning

Status: adopted for scene version 8. Meshes with joints and weights deform
by the displayed poses of the joints a `Skin` names, as glTF 2.0 skins do,
on both presentation paths.

## Context

Characters are skinned meshes: each vertex follows up to four joints. The
joints are ordinary entities with transforms and parents, animated by
players (decision 035). Displayed joint poses exist on the CPU only after
CPU preparation; GPU-presented frames (decision 014) compose poses on the
GPU and keep no complete per-node pose buffer.

## Decision

- **Mesh attributes.** `Mesh::with_skin` takes four joint indices and four
  weights per vertex (glTF `JOINTS_0` and `WEIGHTS_0`), validated
  `asset.attribute_count` and `asset.invalid_attribute`: weights finite, not
  negative, not all zero, normalized to sum to one; zero-weight joints read
  as joint 0. `Mesh::joint_count` is one more than the largest weighted
  joint index.
- **`Skin`.** A component with the joint entities and one inverse bind
  matrix per joint. A vertex moves by the weighted sum of its joints'
  displayed world matrices times their inverse bind matrices; the entity's
  own pose does not move skinned vertices. A skin with fewer joints than
  the mesh needs draws the mesh rigidly, and a joint without a valid pose
  counts as the identity.
- **Palettes per frame.** Retained extraction gives every skinned slot a
  palette of its skin's joints in `RenderSnapshot::joints`, from the offset
  in its instance's `skin`, rebuilt each frame. CPU-prepared frames fill
  each joint's matrix from `PresentedTransforms`. GPU-presented frames give
  each joint's node instead, and a compute pass (`skin_palette`) composes
  the joint's displayed pose exactly as culling composes slots', so joints
  and rigid objects attached to them present alike. Renderers upload the
  offset with the slot's color (one more than it in the color record's
  flags), so an offset change is a color change.
- **Drawing.** Skinned meshes upload a skin stream (four 16-bit joint
  indices and weights) and always the second stream (decision 034). Their
  draws through retained slots use skinned pipeline variants: entry points
  that blend the palette matrices (linear blend skinning) for skinned slots
  and use the pose for rigid slots of the same mesh. Owned snapshots draw
  every mesh rigidly.
- **Bounds.** Joints place skinned vertices anywhere, so meshes with joints
  are never frustum culled, cast into every shadow map drawn each frame and
  never into the kept map (decision 020), and their temporally antialiased
  frames (decision 017) always read the previous frame's palette for motion.
- **Authoring.** Scene version 8 inline vertices take `joints` and
  `weights` (every vertex or none); mesh buffers carry them as `EMB2` flag
  4 (joints as four u16, then weights as four f32). Documents take `skins`
  (`id`, `joints` as entity IDs resolved like animation targets, optional
  column-major `inverse_binds`) and entities with a mesh take `skin`. The
  importer maps glTF skins to `ID.skinN` and keeps skinned vertices in bind
  space. The `Skin` inspector lists the joints' authored IDs.

## Observable behavior

- Khronos' CesiumMan and RiggedFigure walk and Fox runs its clips, with
  shadows following their limbs, the same on CPU-prepared, GPU-presented and
  temporally antialiased frames.
- Rigid meshes draw exactly as before.

## Consequences

Palettes cost one compute invocation per joint on GPU-presented frames and
48 bytes per joint of upload on CPU-prepared ones. Tighter bounds (from
joint positions) would allow culling skinned meshes; dual quaternion
skinning and more than four joints per vertex are not supported.
