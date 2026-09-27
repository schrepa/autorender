# Decision 026: Transparency and particles

Status: adopted for every renderer path. `Particles` is inserted by default
and costs nothing without emitters; transparency costs nothing without
transparent materials.

## Context

Every surface was opaque, and nothing moved but meshes: glass, smoke,
spray and sparks were impossible.

## Decision

- **Transparent materials.** `Material::with_opacity` gives a material an
  opacity below 1 (its color's alpha; `Material::new` still requires 1, and
  cubes stay opaque). Per-slot colors become four 16-bit fractions of color
  and opacity in the same 16-byte record. Batches split by opacity class;
  the opaque pass skips transparent batches, which cast no shadow.
- **Order-independent blending.** After the opaque pass, transparent
  surfaces (lit by local lights when they are on) and alpha particles add
  their weighted color and coverage into an accumulation target and
  multiply what they let through into a revealage target, in any order,
  with read-only depth (McGuire and Bavoil's weighted blended transparency).
  A composite blends their weighted average over the frame's color, keeping
  only the opaque share of the ambient light that ambient occlusion
  darkens; additive particles then add their light (and glow). The targets
  exist only while frames draw something transparent.
- **Particles.** `ParticleEmitter` is a component: rate, lifetime, speed,
  a local direction and spread cone, world acceleration, size and color
  from birth to death, emission, alpha or additive blending, streaking,
  a local offset, a birth radius and a seed. Particle `k` is born at
  `k / rate` seconds of presentation time; each emitter's
  `rate * lifetime` instances draw the latest births, so particles are a
  function of emitter, index and time: pausing freezes them, captures are
  exact, and the CPU extracts one record per emitter per frame. A moving
  emitter leaves its particles along its path. Billboards face the camera
  or streak along their motion; they fade into opaque surfaces (soft
  particles). Alpha particles are lit by the sun and sky; additive ones
  are unlit and glow.
- **Settings.** `Particles { enabled, max_particles }` (decision 029 makes
  the budget optional, none by default, and removes the emitter cap): beyond
  the budget the emitters nearest the
  camera win, whole.
- **Authoring and agents.** Scene version 6 materials author `opacity`,
  entities author `particles`, and the environment's `particles` section
  sets the settings. `ParticleEmitter`, `Particles` and `Mesh3d.opacity`
  inspect in those shapes; `rendering` reports `transparent_draws` and
  `particles`.

## Observable behavior

- The Stress Lab city's chimneys smoke, its fountains spray and its glass
  domes are transparent; `effects=no-particles` turns particles off.

## Measurements

None yet. The city draws about 39,000 particles from its nearest 1,024
emitters.

## Consequences

Glass that bends and blurs what is behind it is decision 038; decals on
opaque surfaces are decision 039. Weighted
blending approximates order: overlapping transparent layers of very
different opacity or distance blend less exactly than sorted ones, and
transparent surfaces write no depth, so ambient occlusion, reflections and
sun shafts see the opaque surface behind them. Temporally antialiased frames
reproject them with the motion of what is behind them. Particles collide
with nothing, and reflections show them only where they cover the surface a
reflected ray hits.
