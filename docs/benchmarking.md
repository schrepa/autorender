# Repeatable browser measurements

Owner: native development tooling. The Node runner owns its Chrome process,
fresh browser pages, and report artifact; each development command owns a bounded
sampling window between ordinary host frames. It adds no production frame work.

Start a development server with fresh Wasm bindings, then run:

```sh
cargo xtask benchmark --output artifacts/my-baseline.json
```

The equivalent direct client remains available while Rust fails to compile:

```sh
node web/scripts/benchmark.mjs --url http://127.0.0.1:5173 --output artifacts/my-baseline.json
```

The runner does not start a server or rebuild Wasm. `cargo xtask benchmark`
ensures the locked web dependencies are installed and forwards options to Node.
Use Node 24 or later and installed Chrome; `BROWSER_CHANNEL`, `BROWSER_PATH`, and
`SOFTWARE_GPU=1` select the same browser controls as the integration smoke tests.
The default server is loopback port 5173. Port 5174 can target a managed iteration
server. Avoid editing or rebuilding during a suite: loaded Wasm/host changes
invalidate the run. Production intentionally has no development benchmark API.

Close other browser workloads and stop concurrent builds/tests before collecting
a baseline. Record power mode and other relevant machine conditions alongside
the artifact. Matching metadata establishes comparable settings, not identical
thermal, power, background-process or operating-system scheduling conditions.

## Fixed workload suite

The viewport is 1440 × 1000 with device scale factor 1. Actual canvas dimensions
and browser pixel ratio are recorded. Every run gets a fresh development page,
paused at tick zero. Orbit's density is selected through its normal control
before sampling. Camera and culling controls render without advancing simulation.

| Workload | Candidates | Camera position → target | Culling expectation |
| --- | ---: | --- | --- |
| Orbit overview | 1,000 | `[0,75,110]` → `[0,0,0]` | All visible |
| Orbit partial | 10,000 | `[0,31,64]` → `[0,0,0]` | Some visible, some culled |
| Orbit partial | 50,000 | `[0,31,64]` → `[0,0,0]` | Some visible, some culled |
| Orbit away | 10,000 | `[0,31,64]` → `[0,70,140]` | None visible |
| Crystal Garden overview | Authored scene | `[17,21,26]` → `[0,0,0]` | All visible |

Orbit uses a 60° vertical field of view, near 0.1 and far 1000. Crystal Garden
uses 48°, near 0.1 and far 150. The report includes all camera values and observed
before/after camera data. Camera direction is checked independently of target
distance because the fly controller represents its target as a unit direction.

Each workload runs with culling disabled and enabled, three repetitions by
default. The first repetition measures disabled then enabled; the next reverses
that order. Each repetition uses fresh pages and the same simulation ticks.
The authored scene response is pinned to one exact source snapshot for the suite.
Changing visibility mode does not replace the CPU gameplay workload.

| Option | Default | Bound / meaning |
| --- | --- | --- |
| `--url URL` | `http://127.0.0.1:5173/` | HTTP loopback base URL; no credentials/query/fragment |
| `--warmup N` | 30 | 0–120 discarded frames per run |
| `--samples N` | 120 | 1–240 measured frames per run |
| `--repeats N` | 3 | 1–5 fresh-page repetitions per workload/mode |
| `--mode step\|render\|present` | `step` | One exact tick and current render, current render only, or half-tick render only |
| `--setup-ticks N` | 1 for `present`, otherwise 0 | 0–120 exact ticks before warm-up/sampling |
| `--output PATH` | Unique file in `artifacts/benchmarks/` | Creates a new report; existing files are never overwritten |
| `--baseline PATH` | None | Compare with a compatible complete report |
| `--max-regression-percent N` | None | Optional 0–1000 threshold; requires a baseline |

For a short integration check, use `--warmup 5 --samples 15 --repeats 1`.
That verifies the workflow; it is not equivalent to the default baseline.
The suite is capped at ten minutes, each host sampling command at ten seconds,
and the report at 2 MiB. Slow systems can use fewer samples/warm-up frames.
Interrupted, hidden, resized, timed-out or incomplete runs return a structured
failure. Completed run records remain in an incomplete artifact when possible;
incomplete artifacts cannot serve as baselines. The runner closes its own pages
and browser on completion or failure. It never retries an uncertain benchmark
mutation.

## What is measured

The host primes one animation-frame boundary, executes one `step` or `render`,
and measures the interval to the following animation frame, including the final
sample. Warm-up observations are discarded. In step mode, the tick difference is
exactly `warmup + samples`; render and present modes advance no ticks. Input is
cleared. Configured setup ticks happen before that window and are excluded from
timing. `present` samples always use an explicit interpolation fraction of 0.5.
All three modes measure the exact CPU presentation path. Ordinary running frames
present interpolated poses on the GPU instead
([decision 014](adr-014-gpu-presentation-and-culling.md)); measure that path with
ordinary-frame recordings or profiles, not with these modes.

Per run, the artifact records min/median/p95/p99/max for CPU simulation,
extraction, render preparation/submission, their sum, and browser animation-frame
intervals. Percentiles use linear interpolation at `(count - 1) × fraction`.
CPU stage timers exclude JSON/stat collection; frame intervals include that host
work and browser pacing. This is an instrumented development workload, not a
production FPS claim. **CPU render/submission time is not GPU execution time.**
There are no GPU timestamps or JS/Wasm/process-heap measurements.

Reports also include candidate/visible/culled/submitted counts, draws, uploads,
simulation steps, dropped time, missing assets, invalid transforms, CPU asset
payload bytes and resident GPU mesh/texture bytes. Those byte counters exclude
allocator overhead, ECS/game state, retained snapshots and driver allocations.
All measured-frame ranges are checked, so a temporary dropped entity or missing
asset cannot silently make a workload appear faster. Candidate counts must stay
constant, and the visibility expectations above must hold throughout sampling.

The summary's `median_of_runs` takes the median of each run statistic. For
example, `total_cpu_p95_ms` there is the median of three per-run p95 values; it is
**not** the p95 of pooled frames. `repeat_spread` retains the distribution of run
medians, including their minimum and maximum. Every original run remains in the
artifact. Small differences near timer resolution or repeat variation should
not be presented as a general engine improvement.

## Provenance and comparisons

The runner records actual browser version, channel/launch flags, headless and
software-GPU settings, operating system/architecture, CPU model/core count,
physical memory, user agent and a WebGPU adapter-info probe. Adapter information
is reported as supplied by the browser, may be incomplete, and does not claim
physical hardware execution. The probe requests the same high-performance
preference as the renderer. See the [WebGPU adapter information contract](https://gpuweb.github.io/gpuweb/#gpuadapterinfo)
and [Playwright browser version API](https://playwright.dev/docs/api/class-browser#browser-version).

Actual loaded Wasm bytes and host script responses receive SHA-256 fingerprints;
all pages for one game must retain the same fingerprints. Host collection is
bounded to 64 modules, 1 MiB per module and 8 MiB in total, outside measured
frames. The exact scene bytes/hash, scene version and lossless decimal u64
revision are recorded. Available local Cargo/npm lockfiles are also fingerprinted.
These hashes identify artifacts; they are not authenticity signatures or a Git
commit claim.

```sh
cargo xtask benchmark --baseline artifacts/my-baseline.json --output artifacts/my-current.json
cargo xtask benchmark --baseline artifacts/my-baseline.json --output artifacts/my-gated-run.json --max-regression-percent 15
```

Comparison rejects different suites, workload/camera settings, viewport,
sample/warm-up/repeat/mode settings, browser versions, environment/adapter
metadata, observed canvas sizes/cameras, or authored scene snapshots. It allows
code/Wasm/host fingerprints to change so engine revisions can be compared.
Summaries are recomputed from the validated run records rather than trusting
saved summary fields.

`culling_comparison` reports enabled/disabled ratios within this suite.
`baseline_comparison` reports current/baseline ratios for matching workloads and
culling modes. Optional regression gating applies to total CPU median/p95, frame
interval median/p95, CPU asset bytes, GPU mesh bytes and GPU texture bytes.
Without a threshold, the runner reports changes without failing for slower
numbers. A zero-to-positive metric has `ratio:null`, `percent:null` and
`zero_baseline:true`; with gating, that counts as a regression. Zero-to-zero is
ratio 1 and change 0. Threshold failures still save the complete measured report.

Stdout contains one versioned JSON success/failure envelope with the artifact
path and summaries. Progress uses bounded per-run JSON lines on stderr. Report
paths are new files; comparing never rewrites a baseline. A failed compatibility
check returns `incompatible_baseline` rather than a misleading numeric comparison.

Unit contracts live in `web/scripts/benchmark-report.test.mjs` and the host's
development adapter tests; `cargo xtask test` runs both. Browser measurements must
be collected separately after build/lint/testing finishes to avoid interference.

## Interpolation correctness and cost

The isolated browser check drives the ordinary browser frame entry at controlled
timestamps to produce one fixed tick followed by a half-tick frame, then checks
five explicit fractions between exact player ticks. A child must follow the
parent's presented pose in both paths. Local and
authoritative world transforms, gameplay state and the simulation tick remain
unchanged by those previews. It also checks explicit fractional captures,
ordinary exact captures, invalid fractions, focus/pause/reset/reload history,
repeated replacement storage, and equal pixels with culling enabled/disabled at
a far-plane crossing:

```sh
node web/scripts/interpolation-smoke.mjs
```

Screenshots and a compact JSON result are saved/reported under
`artifacts/interpolation-smoke-<id>/`. The fixture is intercepted inside its own
page; the sample scene source is never edited.

For the cost of fractional extraction at the same authoritative pose, collect
two suites after other builds/tests finish:

```sh
cargo xtask benchmark --mode render --setup-ticks 1 --output artifacts/current-render.json
cargo xtask benchmark --mode present --setup-ticks 1 --output artifacts/half-tick-render.json
```

Both start at tick one and remain there during warm-up and measurement. Compare
the culling-disabled rows when assessing extraction overhead: all candidates are
submitted in both modes. Fractional poses can legitimately change visible counts
in culling-enabled rows. These are different modes, so automatic baseline
comparison rejects mixing them; compare each mode against later reports of the
same mode for regression gating. Fixed `step` suites measure history maintenance
as part of the real simulation path, while the render/present pair excludes new
history capture. A fixed half-tick benchmark measures cost, not visual smoothness;
the browser smoke verifies the actual intermediate positions and pixels.

Each observation retains the actual presentation mode, fraction and extraction
frame/tick, plus history node/capacity/byte counts, presented node count,
presented component payload bytes and retained presentation scratch bytes.
Summaries include their per-run medians when
available. The storage counters describe their own retained vectors/component
payloads; they exclude scratch maps, ECS table metadata, allocators and complete
JS/Wasm heaps.
Default `step`/`render` runs keep zero setup ticks, preserving the original suite
configuration for compatible comparisons with earlier artifacts. Nonzero setup
ticks and the half-tick alpha are explicit configuration fields and must match
any baseline.

### Initial interpolation measurements

The local 2026-09-25 UTC run used headless Chrome 153.0.8010.50, the default
30 warm-up frames / 120 samples / three repetitions, and the viewport above.
The browser reported an Intel adapter; GPU execution time was not measured.
Both render suites used identical source/code fingerprints and environment
metadata, with one setup tick and culling disabled for the comparison below.
Values are medians of the three run statistics, in milliseconds.

| Workload / mode | Total CPU median | Per-repeat CPU medians | Extraction median | Total CPU p95 | Frame interval p95 |
| --- | ---: | --- | ---: | ---: | ---: |
| 50k current render | 9.60 | 9.70, 9.60, 9.30 | 5.90 | 10.50 | 16.90 |
| 50k half-tick render | 18.60 | 18.60, 18.70, 17.80 | 14.70 | 20.03 | 33.40 |
| Playground current | 0.30 | 0.30, 0.30, 0.30 | 0.10 | 0.41 | 16.80 |
| Playground half-tick | 0.30 | 0.30, 0.30, 0.30 | 0.10 | 0.50 | 16.90 |

A separate default `step` suite passed compatibility checks against the earlier
culling baseline. It includes history maintenance and current-pose rendering,
but does not measure a fractional render in the same frame. At 50k, the new path
has a substantial CPU cost even before fractional extraction:

| 50k step configuration | Total CPU median, prior → new | Simulation median, prior → new | Extraction median, prior → new | Frame interval p95, prior → new |
| --- | ---: | ---: | ---: | ---: |
| Culling disabled | 14.60 → 24.40 | 9.00 → 17.00 | 2.10 → 4.70 | 16.90 → 33.50 |
| Culling enabled | 13.60 → 25.90 | 8.30 → 17.80 | 3.70 → 6.40 | 16.91 → 33.50 |

The culling-disabled step run medians were 25.00, 24.40 and 23.00 ms; the earlier
baseline had 15.30, 13.20 and 14.60 ms. Playground step medians stayed around
0.30 ms, near the timer's resolution for this workload. These measurements
establish a high-density performance limitation; opt-in interpolation is not a
free operation or a demonstrated improvement to throughput.

At 50k, retained history was 12,800,000 bytes and presented component payloads
were 4,000,000 bytes. Fractional extraction additionally retained 8,000,000 bytes
of presentation scratch: at least 24.8 MB across those counters, excluding map,
allocator and ECS overhead. Playground's corresponding counters were 1,792,
2,640 and 5,280 bytes. Capacity reflects previous workload sizes selected through
normal controls, so it need not scale exactly with the current entity count.

The complete local artifacts are
`artifacts/interpolation-current-render.json`,
`artifacts/interpolation-half-tick.json`, and
`artifacts/interpolation-step-benchmark.json`; each retains all 30 runs and its
provenance. `artifacts/interpolation-measurement-summary.json` contains the
extracted phase/repeat/storage figures, including the previous baseline.
