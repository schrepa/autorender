# Development interface

Owner: development tooling. `engine_dev` depends on core, collision and authored scenes;
the runtime does not depend on it unless the façade's development feature is
enabled. There are no GPU or browser imports. Native file operations are absent
from Wasm. Schema generation runs during the host build; browser dependencies
contain serialization but no reflection or schema generator.

`DevTools` owns an explicit registry of read-only component/resource projections.
Call `execute(&mut Engine, Option<&SceneInstance>, &str)` synchronously between
frames. Inspection refreshes the engine-owned transform propagation boundary;
it never advances simulation/presentation, changes input, or writes gameplay state.
The host supplies session identity, transport,
pause/step/input, rendering diagnostics, reload, and screenshots.

## Requests and observations

Every request requires `version: 1`. Unknown fields and operations are errors.

```json
{"version":1,"op":"status"}
{"version":1,"op":"describe"}
{"version":1,"op":"describe","name":"Transform"}
{"version":1,"op":"describe","name":"BrowserControl"}
{"version":1,"op":"inspect","offset":0,"limit":20}
{"version":1,"op":"inspect","entity":"player","component":"Transform"}
{"version":1,"op":"inspect","entity":"player-accessory","component":"GlobalTransform"}
{"version":1,"op":"inspect","entity":"player","component":"InterpolateTransform"}
{"version":1,"op":"inspect","entity":"player-accessory","component":"PresentedTransform"}
{"version":1,"op":"inspect","component":"PresentationState"}
{"version":1,"op":"inspect","parent":"west-lantern","component":"Hierarchy","limit":20}
{"version":1,"op":"inspect","component":"Transform","offset":20,"limit":20}
{"version":1,"op":"inspect","component":"Camera"}
{"version":1,"op":"assets"}
```

Responses contain `version`, `ok`, `observation`, and either `result` or
`error: {code, path, message}`. Observation includes frame, fixed tick, and the
active scene's ID and revision. Frame, tick, and revision are **decimal strings**
so JavaScript preserves all u64 values. The host may place this observation
inside its transport result, along with session and rendering information.

Observation also includes `presentation`, sourced from the engine's actual last
prepared extraction: `mode` (`current` or `interpolated`), `alpha`, decimal-string
`frame`/`tick`, and history storage/node counters. Before extraction, its frame
and tick are null. `current` uses alpha one. These extraction counters can differ
from the current authoritative frame/tick after gameplay advances; they do not
promise GPU submission or completion. Browser capture verifies the rendered result.

Entity inspection is scoped to the supplied scene instance, checked against the
engine world. Runtime-only entities are counted by status but not enumerated.
Authored IDs, readable names, tags, and registered component names are returned.
Queries filter by one component, sort by authored ID, omit removed entities, and
return `total` and `next_offset`. Inspecting a removed entity directly reports
`entity_removed`; a removed component reports `missing_component`. Components
are serialized only when explicitly selected. A resource uses `component`
without an entity selector. Camera is a resource; Transform and Mesh3d are
components. Mesh3d reports asset readiness, geometry size, the material
`surface` and whether its texture masks emission, never persisted runtime
handles. `parent` filters direct current children, is mutually exclusive
with `entity`, and cannot select resources. `Hierarchy` distinguishes the current
parent from its source parent and identifies prefab definition, instance and local
ID. `GlobalTransform` reports a column-major matrix and translation; invalid
orphaned branches return `valid:false` and null values. The full matrix preserves
shear, so no misleading rotation/scale decomposition is returned.
`InterpolateTransform` reports `{enabled:true}` only where the opt-in marker is
present. `PresentedTransform` exposes the same matrix/translation shape as
`GlobalTransform`, but retains the last prepared pose, including interpolated
ancestors. It is invalid with null matrix/translation before preparation or for
an invalid presented branch. Inspecting it, authoritative transforms, or
`PresentationState` never recomputes the presented matrix or runs gameplay,
with one exception: after a GPU-presented frame
([decision 014](../../docs/adr-014-gpu-presentation-and-culling.md)), an
`inspect` request first prepares that frame's exact CPU reference while its
inputs are unchanged, before refreshing transforms. Observations and
`PresentationState` report `gpu_pending` while such a frame is unprepared on the
CPU; the request's observation precedes that preparation.
The two pose names are entity-scoped protocol projections backed by the engine's
canonical resources, not Bevy component types. Their JSON schemas and null-before-
preparation behavior are unchanged; typed clients use `GlobalTransforms` and
`PresentedTransforms`. See the [migration](../../docs/migrations/0.1-canonical-poses.md).
`parent_in_active_scene` verifies that the parent ID resolves to that exact entity
in the active instance, including when another instance has identical source IDs.

`describe` lists projections. A named description includes schema, description,
and default where supported. Transform, GlobalTransform, InterpolateTransform,
PresentedTransform, PresentationState, Hierarchy, Camera, Mesh3d, Collider,
CollisionWorld, and protocol schemas are
generated from their actual serialized DTO definitions during compilation.
`Atmosphere`, `Shadows`, `AmbientOcclusion`, `Bloom` and the combined
`Environment` use the scene version 5 `environment` section types from
`engine_scene` ([decision 022](../../docs/adr-022-scene-environments-and-agent-controls.md)),
so an inspected value can be sent to the `environment` control or saved in a
scene unchanged. `Environment` reads missing effect resources as off; the
individual projections report them as missing.
Defaults come from gameplay types. `SceneDocument` uses the checked-in generated
scene schema; `cargo xtask schema` remains its source of truth. Referential and
semantic constraints also run in Rust validation.

`BrowserControl` describes the host's actual typed input, step, pause, resume,
render, present, capture, reload, camera, culling, environment and benchmark requests. Movement axes default to zero in [-1, 1],
look deltas default to zero in [-1000, 1000], and held/pressed/released actions
are u32 bitmasks. Step accepts 1–120 ticks. Reload requires decimal u64 strings
`expected_revision` (running scene) and `revision` (saved source). Input, step,
render, capture, and reload require a paused browser. These controls are
temporary; persisted authoring goes through the separate source helper.

`{"version":1,"op":"present","alpha":0.5}` requires pause and previews
existing fixed-tick history without advancing either schedule, clocks or frame
counters, or consuming input. Alpha must be finite and in [0,1]; validation uses
the request's double precision before conversion to the engine's float. Invalid
values return `invalid_presentation_alpha` at `$.alpha` before mutation. Ordinary
`step`, `render`, and `capture` show authoritative current poses. Capture accepts
an optional `alpha` to prepare and capture a fractional preview explicitly;
omitting it restores current poses. With an alpha, `present` and `capture` also
accept `"gpu": true` to render that fraction through GPU presentation instead of
the exact CPU path; the reply's visible count is read back from an earlier GPU
frame, so repeat an identical preview to observe its own count. These controls
never change authored source.

Camera/culling/environment controls also require pause and render without
advancing clocks. Camera validation precedes mutation; its full pose/lens must
produce a finite projection. The environment control validates every named
section like a scene (`invalid_environment` at the field's path) before
replacing only those resources, and reports them in `environment_changed`. The host owns RAF-paced `benchmark` sampling with generated request
metadata, bounded warm-up/sample counts and a ten-second deadline. It excludes
concurrent controls and cancels on lifecycle/size changes. Step mode clears input
and advances gameplay; render mode does not. Present mode samples existing history
at alpha 0.5 without advancing gameplay; seed a fixed tick before measuring moving
poses. Results include interpolated timing
percentiles, counter ranges and CPU/GPU asset bytes, with explicit limitations:
these are not GPU timestamps or full-process memory. See `docs/benchmarking.md`.

`Mesh3d` includes immutable local bounds. `status` reports the current culling
mode; the browser adds the latest rendered candidate/visible/culled/dropped
counts and upload bytes. Native inspection does not trigger render extraction.
History byte counters cover retained history vector payload.
`presentation_scratch_bytes` reports reserved presentation scratch payload;
`presented_store_bytes` reports canonical presented-pose/entity storage.
`presented_component_bytes` remains as a zero-valued compatibility field because
presented poses are no longer ECS components.
These exclude allocator and map overhead, other ECS data, and GPU allocations;
they are not process-memory totals.

## Collision observations

Install `CollisionPlugin` to enable spatial queries. `describe` includes
`Collider` and `CollisionWorld`; the generated `protocol` schema describes these
read-only requests. They do not require pause:

```json
{"version":1,"op":"inspect","entity":"player","component":"Collider"}
{"version":1,"op":"inspect","component":"CollisionWorld"}
{"version":1,"op":"raycast","origin":[0,2,10],"direction":[0,0,-1],"max_distance":20,"filter":{"sensors":"exclude","exclude":"player"}}
{"version":1,"op":"overlap","shape":{"kind":"sphere","radius":2},"position":[0,1,0],"limit":20}
{"version":1,"op":"shape_cast","shape":{"kind":"cuboid","half_extents":[0.5,1,0.5]},"position":[0,2,0],"direction":[1,0,0],"max_distance":3}
{"version":1,"op":"collision_events","entity":"player","offset":0,"limit":20}
```

Raycasts and shape casts return the single closest `hit`, or null for a complete
miss. Overlap returns `hits` and `total`; it fails if the complete result cannot
fit the requested limit (1–100, default 20). Native candidate/result exhaustion,
incomplete worlds, and numerical failures produce structured errors without
partial hit results. A partial scan is never reported as a nearest hit or miss.
Successful queries include `complete:true` and a `snapshot` with the collision
cache revision, source frame/tick, admission counts, storage payload and limits.
All u64 identities/counters use decimal strings.

Query shapes use world dimensions in [0.0001,10000]. Positions and ray endpoints
must remain within +/-1000000; travel distance is in [0,2000000]. Directions must
be finite and nonzero and are normalized. Overlap/cast `rotation` defaults to
identity; supplied XYZW quaternions must have squared length within 0.001 of one
and are normalized before use. A cast translates with fixed orientation, zero
skin, and stops at an initial overlap. Validation happens before derived cache
refresh, including resolution of an optional excluded entity.

Filters default to `memberships:1`, `filter:4294967295`, `sensors:"include"`, and
no excluded entity. Layer filtering is symmetric: both participants' membership
bits must intersect the other's filter bits. Sensor selection also accepts
`exclude` and `only`. An exclusion must resolve to a live authored entity in the
active scene. Hits cover the entire collision world, including outside-scene and
runtime-only blockers. Each identity includes `id`, `scene`, `name`,
`in_active_scene` and `alive`; null IDs and `in_active_scene:false` are explicitly
non-addressable through this scene's inspection API. Runtime indices are never
exposed as authored IDs.
Optional `filter.exclude` and event `entity` accept null to mean no selection;
required geometry fields, filter objects, rotations and page sizes reject null.

Queries and collider/cache inspection refresh authoritative transforms and the
collision cache only. They do not advance simulation, frame/tick counters, input,
sensor-event state, or last presented matrices. Collider inspection separates
world-transform admission from cache status. Plain status reports the latest
cache without refreshing it. Spatial operations without the plugin return its
structured missing-plugin error.

`collision_events` reads the latest attempted fixed tick's sensor Enter/Exit
transitions without refreshing or consuming them. It reports `sequence`, `tick`,
`committed_tick`, `complete`, candidate/active-pair counts and a bounded error.
Incomplete scans retain the previous complete pair state and publish no partial
transitions; `committed_tick` identifies that retained state. Multi-tick stepping
exposes the final attempted tick, not an accumulated stream. Optional entity
selection may resolve a removed authored entity, preserving identity for exits.
Pagination never clears events and returns `next_offset`. There are no rigid-body
contact manifolds or solver observations in this milestone.
Event offsets are bounded to 131072 (twice the native maximum retained sensor
pairs, allowing an exit and enter batch). Collider `not_cached` includes entities
outside the configured capacity; consult `CollisionWorld.overflow_colliders`.

## Extension and costs

Register a game resource once when creating the development interface:

```rust,ignore
tools.register_resource::<GameProgress>(
    InspectorDescriptor::new(
        "GameProgress",
        "Collected crystals and completion state",
        serde_json::json!({"type":"object"}),
    ),
    |progress| serde_json::json!({"collected": progress.collected}),
)?;
```

Use `register_component::<T>` for components. Duplicate/reserved names are
rejected. There is no implicit reflection or discovery of arbitrary private
gameplay data. The game owns extension schemas and must keep them consistent
with their projections. Callbacks are trusted game code with read-only typed
access; keep their own allocations and work bounded. They must not scan the
world. Registration does not add gameplay systems.

| Bound | Limit |
| --- | --- |
| Request UTF-8 bytes | 16 KiB |
| Encoded response | 64 KiB; overflow gives a structured error |
| One projection or descriptor | 16 KiB encoded |
| Registered inspectors | 64 |
| Inspector names | 64 ASCII letters/digits/`_.-` |
| Descriptor description | 512 UTF-8 bytes |
| Entity page | 1–100, default 20 |
| Authored records scanned per request | 10,000 |

Pagination performs an on-demand scan of the scene's bounded, ordered entity
index; it does not scan the ECS world. Filtering checks component presence
without serialization, and projects only the requested page. Response assembly
may hold at most one page of bounded projections before bounded encoding. No
inspection work or serialization runs during ordinary gameplay frames.

## Saved source edits

Native clients share `authoring::validate_scene`, `read_scene_source`, and
`save_scene`. Reads and candidates are capped at 1 MiB. Save requires an existing
valid source, the same scene ID, the expected saved revision, and a candidate
revision exactly one greater. Candidate bytes are preserved, including
formatting. A successful report identifies the path and resulting revision and
sets `persisted: true`; this does not imply the browser has loaded that revision.
Clients reload explicitly and verify the running observation afterward.

Save creates `.NAME.engine-dev.lock` with exclusive creation and holds it through
revision checks and replacement. Cooperating writers therefore cannot both
commit the same revision. A synced same-directory temporary file replaces the
source with an atomic rename. Validation failures and stale revisions leave the
source unchanged. The helper checks for direct-editor changes immediately
before replacement, but this is not an OS-level compare-and-swap for
noncooperating writers. Stop external writers during a save.

A crashed process can leave a lock or temporary file. Confirm no writer is
running before manually removing those files. The helper does not silently
break a stale lock. Atomic replacement does not promise full directory-entry
durability across sudden power loss. Static GLB import in `tools/assetc` uses
this same save path through xtask. Project creation and additional import formats
remain separate future operations.

Texture observations include bounded authored IDs, dimensions, CPU readiness
and byte counts without pixel payloads or runtime handles. The browser façade
adds per-ID GPU residency from the renderer at the same command boundary.
`Mesh3d` projects texture dependency readiness/dimensions. Scene format schemas
are generated directly from `engine_scene` in the host build, alongside view
schemas, and stay outside production and ordinary frame execution.
