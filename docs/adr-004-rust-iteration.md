# ADR 004: build generations and session replacement

Accepted 24 September 2026. Extends the agent development interface from ADR 003 without changing gameplay scheduling or asset ownership.

Rust edits require a newly compiled Wasm artifact. Agents should receive compiler diagnostics and a usable replacement session through the same local interface, while a broken edit leaves the previous game available.

The Node supervisor owns source polling, one compiler job, Vite, the bridge, and browser lifetimes. A captured native xtask worker owns Cargo invocation and binding generation. Its immutable outputs live under `web/pkg-iterations/BUILD_ID`; production and ordinary dev outputs remain separate. A dedicated `target/iteration-wasm` avoids races with production builds. One supervisor holds a working-copy lock, while other checkouts can use independent ports/output paths.

Source content hashes drive a serial coordinator with at most one pending fingerprint. It verifies freshness after compilation, after browser preparation, and before activation. Each complete candidate must expose the expected build identity, start paused, and render successfully. Compilation, binding, source, or activation failures preserve the prior runtime. Candidate invalidation and shutdown prevent promotion.

The bridge drains already accepted commands before replacing its session identity. Each response retains the build/session that accepted its request. Arrivals during replacement are rejected explicitly; stale session IDs cannot mutate the new runtime. A successful replacement resets gameplay and temporary controls. Persisted scenes reload from source. Runtime state migration is outside this decision.

`GET /development` is available before the first successful build and reports active and latest-attempt state separately, with bounded compiler diagnostics. The direct Node CLI remains usable while shared Rust is invalid; Cargo-based CLI launch may itself require compiling that Rust. Native tool changes require a supervisor restart. Progress events are compact; full diagnostic results are queried on demand.

Three successful binding generations are retained. Failed and obsolete outputs are removed only after verifying their owned paths. Shutdown stops the owned process tree and all servers/browsers, including resources acquired during startup. No watcher, compiler, reflection, or new serialization work is introduced into the game frame loop.
