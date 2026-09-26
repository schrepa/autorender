# Decision 015: Pipelined fixed ticks

Status: adopted for GPU-presented browser frames. Extends decisions 008, 013 and
014. Simulated time, tick order and authoritative results are unchanged; only
the frames that run each tick's work move.

## Context

With [decision 014](adr-014-gpu-presentation-and-culling.md), frames between
fixed ticks cost well under a millisecond, and the frame that runs a tick bounds
smoothness. At 60 Hz ticks on a 120 Hz display every other frame ran gameplay,
propagation, post-simulation and the GPU input gather: in Firefox 156 about
14–16 ms of main-thread work for the 90,000-object all-moving Stress Lab workload,
so frames missed every 8.33 ms deadline even though two frames' budget
(16.7 ms) nearly covered one tick.

## Decision

`Engine::run_ahead(predicted_delta_seconds)`, called after a frame is submitted,
runs at most one stage of the next fixed tick ahead of its due frame:

- **First stage** (tick begin, `FixedPreSimulation`, its refresh and
  `FixedSimulation`) starts when the tick would be due within two frames.
- **Second stage** (propagation, `FixedPostSimulation`, its refresh and interval
  capture) completes a started tick when it would be due by the next frame.
  The clock counts it when it becomes due; a frame never runs it twice.

"Due" treats a quarter frame early as due, and at most one tick completes ahead.
At display rates of at least twice the tick rate, each frame therefore runs at
most one stage after submitting, and the due frame only gathers and presents
the completed tick. Stages still pending when a tick is due run then, as before:
at 60 Hz the second stage runs at the start of the due frame and the first after
it, exactly as unpipelined apart from the split.

`BrowserApp` runs ahead only with GPU presentation (the default), predicting the
next frame from the last frame interval; `set_tick_pipelining(false)` disables
it and completes a started tick. The Stress Lab accepts `?pipeline=off`.

When the second stage completes a tick ahead, the same frame also gathers its
GPU inputs (`gather_ahead`), unless the tick changed the node space. The due
frame's gather then finds its inputs current and only presents, so the gather
moves from the frame that runs gameplay to the frame that propagates. Frames
that arrive after that gather but before the tick is due present with
`GpuPresentation::hold`: the consumer keeps presenting the inputs it applied
last, at the frame's fraction, and applies the held gather at the first view
without `hold` (`render.gpu_held`). When a late frame runs another tick before
presenting a gather made ahead, the new gather supersedes it: consumers never
applied the skipped gather, so the new one sends explicit previous poses and its
`shift_sequence` excludes the skipped copy. Blending continues across the late
frame, as it did when every gather waited for its due frame.

## Observable behavior

- Authoritative state, gameplay order and fixed time are identical to running
  every tick when due; tests compare published globals and gathered GPU inputs
  bit for bit at 120 Hz.
- While a tick is in flight, and while a completed tick is not yet due, GPU
  frames present the last applied interval at the requested fraction without
  refreshing or gathering again (a gather made ahead is held): the display never
  shows a tick before the clock counts it, even when a frame arrives earlier than
  predicted.
- Frame-rate systems (`Presentation`) can run between a tick's stages. They then
  observe its first stage: advanced `FixedTime`, gameplay components written by
  `FixedSimulation`, but authoritative `GlobalTransforms` of the last completed
  tick. Gameplay that needs a consistent frame-rate view disables pipelining.
- Input for a tick is the input present when its first stage runs: up to two
  frames earlier than without running ahead. Its simulated time is unchanged.
- Exact extraction, stepping, development commands and `reset_clock` complete a
  started tick first (`Engine::complete_pending_tick`), so observations always
  describe completed ticks; an agent observing a running game can therefore see
  one tick more than an unpipelined engine would. Ordinary GPU frames still
  present the previous interval until the clock counts that tick. Development
  `present`/`capture` previews with `gpu: true` reveal it
  (`Engine::reveal_ticks_ahead`) so they match the exact CPU preview of the same
  state. `prepare_presented_transforms` refuses while a tick is in flight or
  hidden. After a frame the renderer did not submit, the next GPU frame completes
  and presents the latest tick (see decision 014). Checks that interleave development commands with ordinary frames
  disable pipelining (`set_tick_pipelining(false)`, exposed by the Stress Lab
  and playground bindings).

## Measurements

Firefox 156 on a 120 Hz display, Core Ultra 9 275HX with the harness pinned to
performance cores, headful profiling builds, the user's 90,000-object all-moving
configuration and camera. Each comparison alternated two builds in one session
(ABBA, three page loads each, throughput over 4 s after a 6 s warm-up). The same
build varied by up to ±20% between sessions, so compare rows only within a
session. Main-thread milliseconds per tick (engine frame spans divided by ticks)
vary far less. See `artifacts/firefox-120hz-2026-09-25.md`.

| Session | Builds compared | Frame rate (fps) | Main thread per tick (ms) |
| --- | --- | --- | --- |
| 1 | decision 014 as adopted / first stage one frame early | 48.7–48.9 / 57.1–59.4 | 15.7–16.0 / 15.7–15.9 |
| 2 | first stage early / one stage per frame | 53.5–59.8 / 59.0–81.8 | 15.0–16.1 / 15.4–15.8 |
| 3 | one stage per frame / + upload counts, matrix and refit work | 66.1–75.3 / 87.1–92.5 | 15.3–15.9 / 14.4–15.3 |

The last rows of decision 014's measurements list the later input work. With it,
frames that gather and run gameplay measured 6–9 ms (median) and frames that
propagate 5–7 ms; 48–64% of frames still exceeded 8.33 ms in these runs
(25–45% in in-page captures). Splitting the input
gather across both frames did not help: it moved work into the propagation frame,
which then missed deadlines more often (three fresh sessions: 98.0–106.7 fps
split against 101.3–110.7 fps unsplit, in-page captures).

## Consequences

With [deferred leaf globals](adr-016-deferred-leaf-globals.md) the propagation
stage shrank below the gameplay stage, and gathering ahead rebalanced them: in
one page load at 120,000 objects, frames running gameplay measured 5.67 ms and
frames propagating and gathering 5.42 ms (median), against 6.63 and 4.18 ms
gathering when due, and no frame exceeded 8.33 ms (p95 6.2 ms).

The pipeline trades up to two frames of input latency for fitting one tick into
two frames. Frames can only shorten to the larger of a tick's two stages plus
presentation, so the heavier stage — gameplay, or propagation plus the gather —
bounds the frame rate. Further gains must reduce those stages; workers remain a
separate decision.
