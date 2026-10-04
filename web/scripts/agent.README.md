# Local agent browser bridge

For automatic Rust builds and session replacement, use `cargo xtask agent watch` and read [Rust iteration](../../docs/rust-iteration.md). Its managed server defaults to port 5174 and bridge to 7331. `node web/scripts/agent.mjs development` reads `/development` without requiring an active runtime; responses separate the latest build attempt from the active session. This direct client remains usable during Rust compiler errors. Managed command responses and session discovery include the immutable build ID.

Start the development server with `cargo xtask dev`, then start a browser session:

```sh
node web/scripts/agent.mjs serve --url http://127.0.0.1:5173/playground.html --port 7331
```

The bridge launches headless Chrome with WebGPU enabled and prints one JSON readiness record containing `session`, `url`, and `endpoint`. Keep that process running. Use `--browser /absolute/path/to/browser` to select another Chromium executable, or `BROWSER_CHANNEL` to select a Playwright channel. It uses the installed Chrome channel by default; it does not download browsers. Ctrl+C or `close --session SESSION_ID` closes its browser and HTTP server.

The browser API exists only in development builds with `?engine_dev=1`, which the bridge adds automatically. The bridge listens on `127.0.0.1`, rejects browser Origin headers and unexpected Host headers, and accepts only a loopback HTTP development URL. It exposes no browser debugging port. It owns one browser session; launching another bridge creates another session ID.

All client commands print JSON. Failure responses set a nonzero exit status. Discover sessions and use the returned session ID explicitly:

```sh
node web/scripts/agent.mjs sessions --port 7331
node web/scripts/agent.mjs describe --port 7331 --session SESSION_ID
node web/scripts/agent.mjs pause --port 7331 --session SESSION_ID
node web/scripts/agent.mjs inspect --port 7331 --session SESSION_ID --json '{"limit":16}'
node web/scripts/agent.mjs input --port 7331 --session SESSION_ID --json '{"forward":1}'
node web/scripts/agent.mjs step --port 7331 --session SESSION_ID --json '{"ticks":30}'
node web/scripts/agent.mjs capture --port 7331 --session SESSION_ID
node web/scripts/agent.mjs diagnostics --port 7331 --session SESSION_ID
```

For shells with different quoting rules, put the command object in a UTF-8 JSON file and pass `request --file command.json --session SESSION_ID`. A file may instead contain a complete request envelope, including its own session and request ID:

```json
{"version":1,"id":"inspect-player-1","session":"SESSION_ID","command":{"op":"inspect","limit":16}}
```

The equivalent HTTP call is `POST /request` with `Content-Type: application/json` and the envelope above. `GET /sessions` returns `{version:1,sessions:[{session,url,started_at,state}]}`. Command responses are `{version:1,id,session,ok,observation,result}` or `{version:1,id,session,ok:false,observation,error:{code,path,message}}`. Transport failures before dispatch have no observation. Frame, tick, and scene revision counters are decimal **strings**, preserving all `u64` values through JSON and JavaScript. Runtime observations report the frame, tick, and scene revision to which a result belongs, including rejected runtime commands.

`capture` requires a paused runtime. It redraws current authoritative state, waits for presentation, and captures the canvas before the next queued operation runs. Optional `--json '{"alpha":0.5}'` instead captures an explicit interpolation fraction; add `"gpu":true` to capture it through GPU presentation. `present --json '{"alpha":0.5}'` previews that fraction without advancing clocks; a later default capture still redraws current state. Observe `presentation` metadata and inspect `PresentedTransform` separately from authoritative transforms. Capture returns an artifact path under `artifacts/agent/SESSION_ID/` alongside the corresponding observation. Screenshots never accept arbitrary filesystem destinations. `diagnostics` returns at most 100 recent browser warnings and errors, each capped at 2,000 characters.

Temporary `input` applies to paused stepping; it does not edit source. Persistent changes belong in scene source files. Validate and save source through the scene CLI, then request reload with both the currently observed revision and the saved revision.

Paused `camera`, `culling` and `environment` commands update temporary render
state and render without advancing gameplay. `environment` takes the sections of
a scene `environment` (for example `--json '{"atmosphere":{"preset":"night"},"sun_shafts":{}}'`),
replaces only those, and never edits source; inspect `Environment` first. `benchmark` holds the same busy boundary across RAF
sampling (maximum ten seconds), clearing input and optionally stepping gameplay.
It returns bounded CPU/frame distributions and residency/counter ranges. Use the
[benchmark runner](../../docs/benchmarking.md) for repeated fresh-session A/B
measurements and compatible baseline comparisons. The bridge retains its normal
15-second execution deadline; interrupted commands may have completed some steps.

A reload request names both revisions:

```json
{"op":"reload","expected_revision":"1","revision":"2"}
```

Reload failures preserve the active game. Read `describe` for supported runtime fields and limits rather than assuming arbitrary ECS components are inspectable.

## Transport bounds and recovery

Requests and responses are limited to 1 MiB. At most 16 operations may be executing or queued together. Commands execute serially, and each has a 15-second deadline including queue time. An ordinary rejected command leaves the session usable. If an executing command times out, its mutation outcome is uncertain: the bridge closes its browser, marks the session unavailable, and rejects later work. Start a new session rather than repeating the mutation automatically.

Accepted request IDs cannot be reused within a session. Clients do not retry automatically. Each session permits at most 65,536 accepted IDs; start a new session when this bound is reached. Responses are not retained for replay. If a response is lost, inspect state with a new request ID before deciding whether further mutation is appropriate.

Main-frame navigation, page closure/crash, and a replaced runtime identity invalidate the session. Host-code hot reload therefore requires a new bridge session. Authored source reload preserves the runtime identity and reports the new scene revision. An unavailable bridge still accepts `close` so an agent can release its owned process and port.

Run transport checks with `node --test web/scripts/agent.test.mjs`. These tests do not require a browser or GPU.

Run `node web/scripts/agent-lifecycle-smoke.mjs` for browser-backed checks that runtime replacement and navigation invalidate a session, and that unavailable sessions still close cleanly. This starts isolated local HTML fixtures and requires installed Chrome, but no engine server or GPU rendering.

Run the complete agent loop against a running development server with `node web/scripts/agent-smoke.mjs` (activate the workspace Rust toolchain first). It creates an isolated authored fixture, discovers a session, inspects and steps gameplay, captures a screenshot, inspects and temporarily changes the environment, validates and saves a source revision with a version 5 environment through `cargo xtask scene`, reloads that revision, and checks diagnostics. It removes its own scene fixture when finished. The default authored game source is untouched. `DEMO_URL` overrides the default development server address. Screenshots and candidate files remain under ignored `artifacts/agent/` for review.

`raycast`, `overlap`, `shape_cast` and `collision_events` are read-only runtime
verbs and need no pause. Pass their typed fields using `--json` or `--file`.
Discover the `protocol` schema (`describe` with name `protocol`), `Collider` and `CollisionWorld` first; see
[collision observations](../../docs/agent-workflow.md#collision-observations).
Geometric queries refresh the authoritative cache; they never advance gameplay
or change interpolation previews. Sensor snapshots remain available until the
next fixed tick and include explicit completeness/commit identity.
