# Decision 022: Scene environments and agent controls

Status: adopted. Scene version 5 authors an optional `environment`; agents
discover, inspect and temporarily replace it; missing effect resources turn
their effects off and free their GPU work.

## Context

Decisions 018–021 made the look engine resources (`Atmosphere`, `Shadows`,
`AmbientOcclusion`, `Bloom`), but only Rust code could see or change them. An
agent could not discover their fields, read the current values, try another
look without a rebuild, or save one with a scene, and replies did not say
which effects a frame ran or what its shadows cost. The renderer also created
every shadow and post-processing pipeline at startup whether a game used them
or not, and an effect could be turned off but not left out.

## Decision

One JSON shape, `SceneEnvironment`, describes the look everywhere: an
`atmosphere` (a preset, `golden_hour` by default, plus any fields that differ),
`shadows`, `ambient_occlusion` and `bloom`. `engine_scene` owns the types and
their validation; errors carry the code `invalid_environment` and the path of
the field, such as `$.environment.shadows.resolution`. Shadow resolutions are
1024, 2048 or 4096, the near distance is below the shadow distance, the fog
maximum is at most 1, and every other value is finite and nonnegative.

- **Scenes.** A version 5 document may author an `environment`. Instantiating
  it sets all four resources, omitted sections taking the engine defaults, so
  a saved look reproduces the same frame after every reload. A document
  without one leaves the world's look untouched, so looks set in code keep
  working. Disposal does not restore the previous look: the environment
  belongs to the world, not to the instance.
- **Inspection.** `Atmosphere`, `Shadows`, `AmbientOcclusion` and `Bloom` are
  resource projections in the shape of their section; the atmosphere lists
  every field. `Environment` reports the whole look, which can be pasted into
  a scene as is. Missing effect resources read as off.
- **Control.** `{"op":"environment", ...}` takes the same sections, validates
  them alike, replaces only the resources it names, and renders while
  paused. It never changes source: it lasts until another control, an
  instantiated scene with an environment, or a replaced runtime.
- **Observation.** `rendering` in every reply reports `shadows_enabled`,
  `shadow_draws` (beside `draw_calls`, which stays the camera pass's count
  that baselines and checks compare), `shadow_kept_redraw` (0 none, 1 one
  strip, 2 the whole kept map), the casters each shadow map drew
  (`shadow_near_casters`, `shadow_moving_casters`, `shadow_kept_casters`,
  read back a few frames late and null until then), `ambient_occlusion` and
  `bloom`. `Mesh3d` reports the material `surface` and whether its texture
  masks emission (`texture_emission_mask`).
- **Modularity.** `Engine` still inserts the four resources, so the defaults
  look the same, but none is required. Removing `Shadows`,
  `AmbientOcclusion` or `Bloom` turns that effect off; removing `Atmosphere`
  shades with its default. The renderer creates the shadow pipelines the
  first time shadows are on and the post-processing pipelines the first time
  an effect or temporal antialiasing runs, releases the shadow maps when
  shadows turn off, and releases the scene, glow and occlusion targets when
  no effect or temporal antialiasing needs them.

## Observable behavior

- `describe` lists the five projections with generated schemas and defaults,
  and `describe BrowserControl` includes the `environment` request.
- Agent smoke checks inspect the look, reject an invalid shadow resolution
  without changing it, compare captures before and after a night atmosphere,
  and reload a saved version 5 environment that replaces the temporary one.
- Version 1–4 documents with an `environment` fail with `version_required`.

## Measurements

None: the controls and projections run only on request, and removing an
effect removes its passes. Games that never enable shadows or effects no
longer compile their pipelines at startup.

## Consequences

A scene with an environment overrides a look set in code on every reload;
games that set their look in code should not author one, or should author
all of it. Pasting the `Environment` projection into a scene inserts
disabled effects instead of leaving them out, which renders alike. Two
instantiated scenes share one look: the last instantiated with an
environment wins. Pipelines stay once created, so turning an effect off and
on does not recompile shaders.
