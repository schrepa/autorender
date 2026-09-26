# ADR 003: explicit agent development sessions

Accepted 24 September 2026. Extends ADR 002 and retains its gameplay and CPU/GPU ownership boundaries.

The primary development client is an AI coding pipeline. Sample-specific DOM probes cannot serve as the engine interface. Development requires schemas, revision-aware authoring, identified browser sessions, exact controls, and observations tied to rendered state.

`engine_dev` owns versioned inspection, typed projections, generated metadata, and native source validation/persistence. It depends on core/scene contracts. Core never depends on development tooling. The façade's optional `dev` feature composes this with browser stepping; production excludes it.

The Node/Playwright bridge owns a browser and loopback endpoint. It supplies session/request identity, bounded serial requests, deadlines, diagnostics, and capture. TypeScript coordinates pause/lifecycle operations; Rust owns simulation, inspection, validation, and reload. CLI/HTTP share implementation. A future MCP adapter can delegate to the same operations.

Queries select registered projections and authored IDs. Build-time schemas come from serialized DTOs. Commands execute between frames; source fetch/presentation waits exclude competing mutations. Persistent saves and temporary controls have separate responses/lifetimes.

Costs: on-demand JSON copies, at most 100 returned entities, 10,000 authored records scanned, 64 projections, 16 KiB requests/projection values, and 64 KiB runtime responses. The bridge has a separate 1 MiB transport cap and bounded queue. Callbacks must bound their own allocations. Reload temporarily holds old and replacement assets.

The complete first workflow uses Crystal Garden; Orbit supports resource inspection without authored entities. Production/development bindings have separate outputs. A replaced runtime requires a fresh session. Import tooling, arbitrary mutation, and runtime entity discovery remain separate milestones.
