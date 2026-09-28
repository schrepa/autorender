# Decision 035: Animation clips and players

Status: adopted for scene version 8. Keyframed clips animate the local
transforms of the entities a player binds, as glTF 2.0 animations do.
Imported animated models keep their node hierarchy.

## Context

The importer rejected every animated or skinned model and baked node
transforms into vertices. Animation is the other half of most glTF content:
moving parts, doors and characters. The engine already interpolates fixed
simulation poses for display (decision 008) and presents them on the GPU
(decision 014), so animation only has to write local transforms.

## Decision

- **Clips are assets.** An `AnimationClip` holds channels, each animating
  the translation, rotation or scale of one target index with keys at
  strictly increasing times and glTF's step, linear or cubic spline
  interpolation (Hermite curves with tangents scaled by the key span;
  rotations slerp, or normalize the spline). Channels validate as
  `asset.invalid_animation`. `AssetLibrary::insert_clip` counts their keys
  against the byte budget; one clip serves every instance of a model.
- **Players are components.** `AnimationPlayer { clip, targets, time,
  speed, looping }` plays a clip on the entities `targets` names, target
  `i` being the clip's target index `i`. `animate` runs in
  `FixedSimulation` under `AnimationSystems::Animate`: it advances `time`
  by the tick times `speed`, wrapping when looping and holding the last pose
  otherwise, and writes each channel's sample into its target's local
  `Transform` only when it changed, so held poses cost propagation nothing.
  Missing or despawned targets are skipped. Game systems that adjust
  animated poses order after the set. Players sample through
  `AnimationClip::sample_channels`: channels keyed at the same times as the
  channel before them share its key search, and linear rotation channels
  derive each key pair's arc once (near-side end, angle and its sine's
  reciprocal), so a sample costs two sines. Values keep the bits of
  `AnimationChannel::sample` and of glam's scalar and wasm32 `Quat::slerp`,
  and are the same natively (glam's SSE2 and NEON `slerp` use a SIMD sine
  of their own, within a few ulp).
- **Smooth presentation.** Players run at the fixed rate; targets with
  `InterpolateTransform` display blended between ticks on either
  presentation path, so animation needs no presentation code of its own.
  Scene instantiation marks every player target.
- **Authoring.** Scene version 8 documents take `animations`: an `id`, an
  optional `name`, the `targets` (entity IDs) and either inline `channels`
  (`target`, `property`, `interpolation`, `times`, `values`) or a `uri` to an
  `EAN1` animation buffer (`encode_animation_buffer`: channel count, then
  per channel its target, property, interpolation, key and value counts,
  times and values, all little-endian). Entities take `animation` (`clip`,
  `speed`, `loop`, `time`). A player inside a prefab placement resolves its
  targets against the placement's local IDs first, so a prefab animates each
  placement's own nodes. `SceneInstance::clip` and `clips` give the handles
  gameplay switches between.
- **Importing.** Models with animations or skins keep their nodes: one
  entity per node (`ID.nodeN`) with its local transform and parent, below a
  root (`ID.root`) that plays the first animation; primitives are children
  (`ID.nodeN.pM`) in their node's space. Each animation becomes a clip
  (`ID.animN`) in an `EAN1` file; repeated key times are dropped. Mirrored,
  zero-scale and sheared rest poses cannot be authored and are rejected.
  Static models still bake node transforms.
- **Agents.** The `AnimationPlayer` inspector projects a player in its
  scene shape with the clip's duration and its targets' authored IDs.

## Observable behavior

- Khronos' BoxAnimated lifts and turns its box; InterpolationTest's cubes
  step, move linearly and follow cubic splines, those scaled to zero
  hiding as invalid transforms.
- A looping player wraps at the clip's last key; a held pose leaves its
  transforms' change ticks unchanged.

## Consequences

Morph target weights remain unsupported. Blending two clips, events and
root motion extraction are game code over `AnimationPlayer` and
`Transform` for now.
