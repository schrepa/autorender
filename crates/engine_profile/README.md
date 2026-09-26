# Optional profiling

This leaf crate has no default features or dependencies. Normal builds inline
scope/counter/aggregate hooks to no operations. `enabled` adds a main-thread,
thread-local collector and Serde export; the platform supplies the clock.
An executable may opt into `TrackingAllocator::new(System)`. Libraries never
select a global allocator. Core remains independent of browser and GPU APIs.

Call `set_clock`, then `start(1..=65536)`, and `finish` once. Starting twice is
rejected. The event vector is allocated before capture; static names, scope
entry/exit and counters do not allocate. Overflow increments `dropped_events`
and marks the result incomplete. Finishing with open guards likewise marks it
incomplete. Guards from an old capture cannot modify a later one. Guards cannot
move between threads. Calling `finish` without an active capture returns an
incomplete artifact. This is single-thread capture, not a worker tracing system.

`span` records nested wall-clock CPU intervals. `Aggregate::measure` accumulates
hot-loop timings and counts into one event; those summed intervals are **not**
continuous timeline slices. Collision uses aggregates to distinguish broad
traversal and precise tests without retaining an event per collider pair.
Clock calls still add overhead to each measured operation. Captures are for
diagnosis; compare production workload recordings separately before claiming
an optimization. Ordinary builds omit all of this instrumentation.

Allocation counters measure successful allocations/deallocations/reallocations,
requested and freed bytes, current live requested bytes, lifetime high-water
bytes, and allocation failures. A successful reallocation counts its full old
size as freed and its full new size as requested. These process-wide Rust
counters include gameplay, engine, dependencies, and profiling itself. They do
not identify allocation sites or include allocator metadata, fragmentation,
committed Wasm pages, JavaScript heaps, GPU resources, or driver memory. Initial
capture buffer allocation precedes the recorded delta; serialization follows it.
Lifetime live/peak values include retained profiler storage. Snapshots are not
transactional across threads; the current engine is single-threaded.

Unsafe code is confined to `allocator.rs`, which forwards the standard allocator
contract unchanged and updates nonallocating atomic counters. Every other file
retains `deny(unsafe_code)`. Other engine crates retain workspace `forbid`.
