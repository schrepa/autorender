# Architecture decision 002: reusable assets and authored scenes

Status: adopted. This extends the initial frame contract in `architecture.md`; the initial cube-only scope is historical.

## Problem and resulting API

The first renderer could draw only built-in cubes, and creating a different world required editing Rust scene-generation code. The engine now accepts validated indexed triangle meshes and opaque color materials through typed runtime handles. A versioned JSON document describes meshes, materials, cameras, transforms, and named entities. Crystal Garden exercises these APIs through the façade without renderer or JavaScript types in gameplay.

`engine_assets` is a separate crate because it owns immutable CPU assets, budgets, and handle validity independently of the ECS world and GPU. `engine_scene` owns authoring validation and scene lifetime. `engine_core` references asset contracts; `engine_render` consumes extracted core snapshots. `engine` composes them and supplies an optional fly-camera plugin. Core still cannot import the renderer or browser. This boundary replaces the proposed early asset-tool scaffolding with working runtime ownership first.

## Lifetime and cost contract

The asset library owns meshes and materials. Handles are typed, nonowning, generational, and library-specific. They cannot be serialized as authored identifiers. Removal invalidates future lookups; an already extracted snapshot retains immutable geometry through one `Arc` per distinct mesh. Retained snapshots can keep removed source bytes alive, so library counters measure library ownership rather than all process memory. There is no hidden strong-handle reference count or asynchronous loading state machine.

Defaults bound a library to 128 meshes, 256 materials, and 32 MiB of CPU payload. Mesh creation validates normals, positions, triangle counts, indices, and payload size. Material creation accepts finite opaque linear color. Source IDs, entity counts, text size, and transform ranges are bounded during scene validation. GPU residency has independent limits of 128 meshes / 32 MiB; instance data is bounded at 100,000 records. Low-level Rust clients own finite, nonsingular transform validity; authored documents enforce it before mutation.

Extraction resolves materials to instance colors and collects only referenced mesh sources. Missing handles are skipped and counted; they do not consume the valid-instance budget. The renderer groups opaque instances by mesh using reusable index storage, then performs one indexed instanced draw per mesh. Different colors share a draw. First use uploads geometry; ordinary frames upload only camera/instance records. Unreferenced cached meshes retire at the next valid render boundary, including zero-sized frames. No GPU readback is required.

Batch planning takes O(N log N + M log M) for N instances and M sources. Gameplay/extraction remain O(N). Sorting, staging, and snapshots retain capacity, while a workload increase or newly resident asset can allocate. These are design costs, not universal performance claims. Texture/material specialization and transparent draw ordering need a later contract.

## Scenes and authoring

`SceneDocument::from_json`, `validate`, `to_json`, and `instantiate` are the shared authoring/runtime path. Format version 1 uses stable scene/entity/asset strings plus a caller-managed revision. Unknown fields, duplicate IDs/tags, unresolved references, and invalid numerical data report `SceneError { code, path, message }`. JSON Schema is generated from the same Rust types through the optional `schema` feature; semantic validation also checks relationships and budgets that structural schema cannot express.

Instantiation validates and prepares CPU data before world mutation, rolls back inserted assets on admission failure, and spawns only after all dependencies are available. The returned `SceneInstance` owns its entities and asset handles. Disposal is explicit and tied to the original world; it is unrelated to Bevy plugin startup cleanup. Multiple instances can coexist, including instances with the same authored ID. Runtime lookups remain scoped to the instance.

Reload creates the new instance before disposing the old one. This preserves the old scene when validation/allocation fails, and requires temporary budget for both instances. Success changes the camera and resets sample gameplay state. Browser fetches source text only at load/reload boundaries; serialization stays outside simulation and extraction. Runtime controls do not persist changes. Edit the source file, update its revision, validate with `cargo xtask assets`, and reload while using the development server. A production bundle requires rebuilding after source edits.

## Input and extension

`ActionId` selects one of 32 action bits. `ActionInput` preserves press/release edges until the next fixed tick and exposes them only for that tick; held actions survive catch-up. A quick tap between frames retains both edges. Repeated edges before a tick coalesce. Suspension clears held/pending state. The host owns physical key/pointer bindings and sends one bounded batch per frame.

The fly-camera plugin has the public `CameraSystems` ordering point. The sample game defines ordered movement/collection/animation sets. No gameplay behavior requires a renderer branch. The legacy `Cube` remains a convenience component backed by the same mesh path; new reusable content should use `Mesh3d`.

## Verification and follow-up

Native tests cover handle invalidation, foreign libraries, budget/metadata exhaustion, schema/data validation, deferred input, rollback, disposal, mesh batches, shader layout, and gameplay. Browser tests exercise both examples, scene replacement errors, revision changes, and stable CPU/GPU residency across repeated reloads. Local tools and CI use the same xtask vocabulary.

The next concrete content requirement should determine textures, file importers, and cooking. Visibility culling, interpolation, hierarchy, automatic device reconstruction, an inspection bridge, and worker execution remain explicit future work. These capabilities are not represented as completed stubs.
