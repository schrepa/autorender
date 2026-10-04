use crate::{AllocationSnapshot, allocation_snapshot};
use serde::Serialize;
use std::cell::RefCell;
use std::{marker::PhantomData, rc::Rc};

pub const MAX_EVENTS: usize = 65_536;

/// `aggregate` durations are summed samples, not continuous timeline spans.
#[derive(Debug, Serialize)]
pub struct Event {
    pub kind: &'static str,
    pub category: &'static str,
    pub name: &'static str,
    pub start_ms: f64,
    pub duration_ms: f64,
    pub value: Option<f64>,
    pub depth: u32,
    pub frame: u64,
    pub complete: bool,
}

#[derive(Debug, Default, Serialize)]
pub struct AllocationDelta {
    pub allocations: u64,
    pub deallocations: u64,
    pub reallocations: u64,
    pub requested_bytes: u64,
    pub freed_bytes: u64,
    pub live_bytes_change: i128,
    pub failures: u64,
}

#[derive(Debug, Serialize)]
pub struct Capture {
    pub version: u32,
    pub id: u64,
    pub start_ms: f64,
    pub end_ms: f64,
    pub max_events: usize,
    pub dropped_events: u64,
    pub incomplete_spans: usize,
    pub complete: bool,
    pub events: Vec<Event>,
    pub allocations_start: AllocationSnapshot,
    pub allocations_end: AllocationSnapshot,
    pub allocations_delta: AllocationDelta,
}

struct Collector {
    clock: Option<fn() -> f64>,
    generation: u64,
    active: bool,
    start_ms: f64,
    frame: u64,
    depth: u32,
    max_events: usize,
    dropped: u64,
    events: Vec<Event>,
    allocations: AllocationSnapshot,
}
impl Collector {
    const fn new() -> Self {
        Self {
            clock: None,
            generation: 0,
            active: false,
            start_ms: 0.0,
            frame: 0,
            depth: 0,
            max_events: 0,
            dropped: 0,
            events: Vec::new(),
            allocations: AllocationSnapshot {
                available: false,
                allocations: 0,
                deallocations: 0,
                reallocations: 0,
                requested_bytes: 0,
                freed_bytes: 0,
                live_bytes: 0,
                peak_live_bytes: 0,
                failures: 0,
            },
        }
    }
    fn now(&self) -> f64 {
        self.clock.map_or(0.0, |clock| clock())
    }
    fn push(&mut self, event: Event) -> Option<usize> {
        if self.events.len() == self.max_events {
            self.dropped = self.dropped.saturating_add(1);
            return None;
        }
        let index = self.events.len();
        self.events.push(event);
        Some(index)
    }
}
thread_local! { static COLLECTOR: RefCell<Collector> = const { RefCell::new(Collector::new()) }; }

/// Install before capture. Changing clocks during a capture is ignored.
pub fn set_clock(clock: fn() -> f64) {
    COLLECTOR.with_borrow_mut(|collector| {
        if !collector.active {
            collector.clock = Some(clock);
        }
    });
}
pub fn active() -> bool {
    COLLECTOR.with_borrow(|collector| collector.active)
}
pub fn capture_id() -> Option<u64> {
    COLLECTOR.with_borrow(|collector| collector.active.then_some(collector.generation))
}
pub fn current_frame() -> u64 {
    COLLECTOR.with_borrow(|collector| collector.frame)
}
pub fn set_frame(frame: u64) {
    COLLECTOR.with_borrow_mut(|collector| collector.frame = frame);
}

/// Allocate the entire event budget before opening the measured window.
pub fn start(max_events: usize) -> Result<u64, &'static str> {
    if !(1..=MAX_EVENTS).contains(&max_events) {
        return Err("event budget must be 1..=65536");
    }
    COLLECTOR.with_borrow_mut(|collector| {
        if collector.active {
            return Err("capture already active");
        }
        if collector.clock.is_none() {
            return Err("profiling clock not installed");
        }
        collector.events = Vec::with_capacity(max_events);
        collector.max_events = max_events;
        collector.generation = collector
            .generation
            .checked_add(1)
            .ok_or("capture IDs exhausted")?;
        collector.depth = 0;
        collector.dropped = 0;
        collector.start_ms = collector.now();
        collector.allocations = allocation_snapshot();
        collector.active = true;
        Ok(collector.generation)
    })
}

/// Stops immediately; moving the event buffer out requires no new allocation.
/// Open guards mark this result incomplete and cannot affect the next capture.
pub fn finish() -> Capture {
    COLLECTOR.with_borrow_mut(|collector| {
        let was_active = collector.active;
        collector.active = false;
        let end_ms = collector.now();
        let end = allocation_snapshot();
        let begin = collector.allocations;
        let events = std::mem::take(&mut collector.events);
        let incomplete_spans = events.iter().filter(|event| !event.complete).count();
        Capture {
            version: 1,
            id: collector.generation,
            start_ms: collector.start_ms,
            end_ms,
            max_events: collector.max_events,
            dropped_events: collector.dropped,
            incomplete_spans,
            complete: was_active && collector.dropped == 0 && incomplete_spans == 0,
            events,
            allocations_start: begin,
            allocations_end: end,
            allocations_delta: AllocationDelta {
                allocations: end.allocations.saturating_sub(begin.allocations),
                deallocations: end.deallocations.saturating_sub(begin.deallocations),
                reallocations: end.reallocations.saturating_sub(begin.reallocations),
                requested_bytes: end.requested_bytes.saturating_sub(begin.requested_bytes),
                freed_bytes: end.freed_bytes.saturating_sub(begin.freed_bytes),
                live_bytes_change: i128::from(end.live_bytes) - i128::from(begin.live_bytes),
                failures: end.failures.saturating_sub(begin.failures),
            },
        }
    })
}

pub struct Span {
    generation: u64,
    index: Option<usize>,
    thread_bound: PhantomData<Rc<()>>,
}
pub fn span(category: &'static str, name: &'static str) -> Span {
    COLLECTOR.with_borrow_mut(|collector| {
        if !collector.active {
            return Span {
                generation: 0,
                index: None,
                thread_bound: PhantomData,
            };
        }
        let event = Event {
            kind: "span",
            category,
            name,
            start_ms: collector.now(),
            duration_ms: 0.0,
            value: None,
            depth: collector.depth,
            frame: collector.frame,
            complete: false,
        };
        let index = collector.push(event);
        collector.depth = collector.depth.saturating_add(1);
        Span {
            generation: collector.generation,
            index,
            thread_bound: PhantomData,
        }
    })
}
impl Drop for Span {
    fn drop(&mut self) {
        if self.generation == 0 {
            return;
        }
        COLLECTOR.with_borrow_mut(|collector| {
            if !collector.active || collector.generation != self.generation {
                return;
            }
            collector.depth = collector.depth.saturating_sub(1);
            if let Some(index) = self.index {
                let end = collector.now();
                let event = &mut collector.events[index];
                event.duration_ms = (end - event.start_ms).max(0.0);
                event.complete = true;
            }
        });
    }
}
pub fn counter(name: &'static str, value: f64) {
    if !value.is_finite() {
        return;
    }
    COLLECTOR.with_borrow_mut(|collector| {
        if !collector.active {
            return;
        }
        let event = Event {
            kind: "counter",
            category: "counters",
            name,
            start_ms: collector.now(),
            duration_ms: 0.0,
            value: Some(value),
            depth: collector.depth,
            frame: collector.frame,
            complete: true,
        };
        collector.push(event);
    });
}

/// Fine-grained hot-loop timing sums into one event. Timing each operation has
/// overhead: these diagnostic aggregates must not be mistaken for baseline cost.
pub struct Aggregate {
    generation: u64,
    frame: u64,
    depth: u32,
    category: &'static str,
    name: &'static str,
    clock: Option<fn() -> f64>,
    start: f64,
    elapsed: f64,
    count: u64,
    thread_bound: PhantomData<Rc<()>>,
}
impl Aggregate {
    pub fn new(category: &'static str, name: &'static str) -> Self {
        COLLECTOR.with_borrow(|collector| {
            let clock = collector.active.then_some(collector.clock).flatten();
            Self {
                generation: collector.generation,
                frame: collector.frame,
                depth: collector.depth,
                category,
                name,
                clock,
                start: clock.map_or(0.0, |clock| clock()),
                elapsed: 0.0,
                count: 0,
                thread_bound: PhantomData,
            }
        })
    }
    pub fn measure<T>(&mut self, run: impl FnOnce() -> T) -> T {
        let Some(clock) = self.clock else {
            return run();
        };
        let start = clock();
        let result = run();
        self.elapsed += (clock() - start).max(0.0);
        self.count += 1;
        result
    }
}
impl Drop for Aggregate {
    fn drop(&mut self) {
        if self.clock.is_none() || self.count == 0 {
            return;
        }
        COLLECTOR.with_borrow_mut(|collector| {
            if !collector.active || collector.generation != self.generation {
                return;
            }
            collector.push(Event {
                kind: "aggregate",
                category: self.category,
                name: self.name,
                start_ms: self.start,
                duration_ms: self.elapsed,
                value: Some(self.count as f64),
                depth: self.depth,
                frame: self.frame,
                complete: true,
            });
        });
    }
}
