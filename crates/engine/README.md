# engine

Owner: engine composition. Public entry points: `prelude`, core reexports, and `browser::BrowserApp` on Wasm.

Gameplay uses the explicitly Bevy-based core API. `BrowserApp` owns one core runtime, reusable render snapshot, renderer, timing baseline, and last diagnostic record on the browser main thread. The browser supplies input once per frame. GPU and DOM types are confined to the browser module; no JavaScript memory views escape Rust.

Dependencies: core, scene, and collision everywhere; renderer, wasm-bindgen, and web-sys on Wasm. Suspension resets the clock/input without disposing the scene. Errors propagate to the host, which stops animation and displays a reload message. Stats report CPU timings and cumulative dropped simulation time; they do not measure GPU execution.

Ordinary interpolated frames use GPU presentation ([decision 014](../../docs/adr-014-gpu-presentation-and-culling.md)): blending, composition, validation and culling run in compute passes, and core gathers inputs only when ticks or edits change them. `BrowserApp::set_gpu_presentation(false)` selects the CPU presentation path instead; `gpu_presentation()` reports the setting. Current-mode frames and exact development step/render/capture stay on the CPU path, as do `present` previews unless a request asks for `gpu`. Visible, culled and submitted counts of GPU-presented frames come from an asynchronous read-back a few frames old; candidate, invalid and missing-asset counts are current.

The faÃ§ade also reexports `engine_scene` as `scene`, with authoring/validation/instance types in the prelude. Scene serialization runs at load/reload boundaries. `camera::FlyCameraPlugin` provides reusable presentation controls with a named `CameraSystems` insertion point. `BrowserStats` is a generated owned diagnostics object that hosts poll and free outside per-entity work. It includes mesh upload/residency counters. Scene schema generation stays in optional host-tool builds.

Optional feature `dev` reexports `engine_dev` as `dev` and enables exact core stepping. On Wasm, `dev_browser::DevBrowser` composes shared inspection/control requests with `BrowserApp::dev_step` and `dev_render`. Input batches and observation serialization run only on explicit commands. The TypeScript host requires pause before mutation and owns the session's lifecycle; Rust controls input validation and exact simulation. See [agent workflow](../../docs/agent-workflow.md).

`collision` reexports `engine_collision`; common checked geometry, query and
plugin types are in the prelude. Games install `CollisionPlugin` explicitly.
The façade does not infer colliders from render meshes or add a physics solver.
See [collision contracts](../../docs/adr-009-collision-and-spatial-queries.md).
