use std::borrow::Cow;
use std::cell::Cell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use serde::ser::{Serialize, SerializeMap, Serializer};

use crate::Fact;

#[cfg(not(feature = "profiling"))]
const UNCOUNTED_HEAP: &str = "heap_counted_in_profiling_builds_only";

#[derive(Clone, Debug)]
pub enum Field {
    Count(u64),
    Number(f64),
    Text(String),
    Unavailable(&'static str),
}

impl From<Fact<'_>> for Field {
    fn from(fact: Fact<'_>) -> Self {
        match fact {
            Fact::Count(count) => Self::Count(count),
            Fact::Number(number) => Self::Number(number),
            Fact::Text(text) => Self::Text(text.to_owned()),
            Fact::Unavailable(reason) => Self::Unavailable(reason),
        }
    }
}

impl Serialize for Field {
    fn serialize<S: Serializer>(&self, out: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Count(count) => out.serialize_u64(*count),
            Self::Number(number) => out.serialize_f64(*number),
            Self::Text(text) => out.serialize_str(text),
            Self::Unavailable(reason) => {
                let mut map = out.serialize_map(Some(1))?;
                map.serialize_entry("unavailable", reason)?;
                map.end()
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum Facts {
    Span {
        category: &'static str,
        ms: f64,
        depth: u32,
    },
    Counter(f64),
    Listed {
        kind: &'static str,
        facts: Vec<(&'static str, Field)>,
    },
}

#[derive(Clone, Debug)]
pub struct Event {
    pub name: Cow<'static, str>,
    pub frame: u64,
    pub at_ms: f64,
    pub cause: u32,
    pub facts: Facts,
}

impl Event {
    pub fn kind(&self) -> &'static str {
        match &self.facts {
            Facts::Span { .. } => "span",
            Facts::Counter(_) => "counter",
            Facts::Listed { kind, .. } => kind,
        }
    }
}

impl Serialize for Event {
    fn serialize<S: Serializer>(&self, out: S) -> Result<S::Ok, S::Error> {
        let mut map = out.serialize_map(None)?;
        map.serialize_entry("kind", self.kind())?;
        map.serialize_entry("name", &self.name)?;
        map.serialize_entry("frame", &self.frame)?;
        map.serialize_entry("at_ms", &self.at_ms)?;
        if self.cause != 0 {
            map.serialize_entry("cause", &self.cause)?;
        }
        match &self.facts {
            Facts::Span {
                category,
                ms,
                depth,
            } => {
                map.serialize_entry("category", category)?;
                map.serialize_entry("ms", ms)?;
                map.serialize_entry("depth", depth)?;
            }
            Facts::Counter(value) => map.serialize_entry("value", value)?,
            Facts::Listed { facts, .. } => {
                for (key, field) in facts {
                    map.serialize_entry(key, field)?;
                }
            }
        }
        map.end()
    }
}

struct Sink {
    budget: usize,
    dropped: u64,
    events: Vec<Event>,
    opened: Vec<Event>,
    #[cfg(feature = "profiling")]
    kept: Option<Kept>,
}

#[cfg(feature = "profiling")]
struct Kept {
    limit: usize,
    dropped: u64,
    events: Vec<Event>,
}

static SINK: Mutex<Sink> = Mutex::new(Sink {
    budget: 0,
    dropped: 0,
    events: Vec::new(),
    opened: Vec::new(),
    #[cfg(feature = "profiling")]
    kept: None,
});
static RECORDING: AtomicBool = AtomicBool::new(false);
static FRAME: AtomicU64 = AtomicU64::new(0);
static CLOCK: OnceLock<fn() -> f64> = OnceLock::new();

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

fn sink() -> MutexGuard<'static, Sink> {
    SINK.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn now() -> Option<f64> {
    CLOCK.get().map(|clock| clock())
}

fn push(event: Event) {
    let mut sink = sink();
    if RECORDING.load(Relaxed) {
        held(&mut sink, event);
    }
}

fn held(sink: &mut Sink, event: Event) {
    if sink.events.len() >= sink.budget {
        sink.dropped = sink.dropped.saturating_add(1);
        return;
    }
    #[cfg(feature = "profiling")]
    if let Some(kept) = &mut sink.kept {
        if kept.events.len() < kept.limit {
            kept.events.push(event.clone());
        } else {
            kept.dropped = kept.dropped.saturating_add(1);
        }
    }
    sink.events.push(event);
}

#[cfg(feature = "profiling")]
pub fn keep(limit: usize) {
    sink().kept = Some(Kept {
        limit,
        dropped: 0,
        events: Vec::with_capacity(limit),
    });
}

#[cfg(feature = "profiling")]
pub fn take_kept() -> (Vec<Event>, u64) {
    sink()
        .kept
        .take()
        .map_or((Vec::new(), 0), |kept| (kept.events, kept.dropped))
}

fn enter() -> u32 {
    DEPTH.with(|depth| depth.replace(depth.get().saturating_add(1)))
}

fn leave() {
    DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
}

type Named = (&'static str, Cow<'static, str>);
type Stamped = (u64, u32);

fn spanned((category, name): Named, (frame, cause): Stamped, (start, end): (f64, f64)) {
    push(Event {
        name,
        frame,
        at_ms: start,
        cause,
        facts: Facts::Span {
            category,
            ms: (end - start).max(0.0),
            depth: DEPTH.with(Cell::get),
        },
    });
}

pub fn install_clock(clock: fn() -> f64) {
    CLOCK.get_or_init(|| clock);
}

pub fn recording() -> bool {
    RECORDING.load(Relaxed)
}

pub fn set_frame(frame: u64) {
    FRAME.store(frame, Relaxed);
}

fn current_frame() -> u64 {
    FRAME.load(Relaxed)
}

pub fn begin(budget: usize) -> Result<(), &'static str> {
    if budget == 0 {
        return Err("event_budget_below_one");
    }
    if CLOCK.get().is_none() {
        return Err("clock_not_installed");
    }
    let mut sink = sink();
    if RECORDING.load(Relaxed) {
        return Err("already_recording");
    }
    sink.budget = budget;
    sink.dropped = 0;
    sink.events = Vec::with_capacity(budget);
    for event in std::mem::take(&mut sink.opened) {
        held(&mut sink, event);
    }
    RECORDING.store(true, Relaxed);
    Ok(())
}

pub fn drain(into: &mut Vec<Event>) -> u64 {
    into.clear();
    let mut sink = sink();
    std::mem::swap(&mut sink.events, into);
    let budget = sink.budget;
    sink.events.reserve(budget);
    std::mem::take(&mut sink.dropped)
}

pub fn end() {
    let mut sink = sink();
    RECORDING.store(false, Relaxed);
    sink.events = Vec::new();
    sink.budget = 0;
    sink.dropped = 0;
}

pub struct Span {
    started: Option<(Named, Stamped, f64)>,
    thread_bound: PhantomData<Rc<()>>,
}

pub fn span(category: &'static str, name: &'static str) -> Span {
    named_span(category, || Cow::Borrowed(name))
}

pub fn span_of(category: &'static str, name: impl std::fmt::Display) -> Span {
    named_span(category, || Cow::Owned(name.to_string()))
}

fn named_span(category: &'static str, name: impl FnOnce() -> Cow<'static, str>) -> Span {
    let started = recording()
        .then(now)
        .flatten()
        .map(|start| ((category, name()), (current_frame(), crate::cause().0), start));
    if started.is_some() {
        enter();
    }
    Span {
        started,
        thread_bound: PhantomData,
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some((named, stamped, start)) = self.started.take() {
            leave();
            if let Some(end) = now() {
                spanned(named, stamped, (start, end));
            }
        }
    }
}

pub fn counter(name: &'static str, value: f64) {
    if !recording() || !value.is_finite() {
        return;
    }
    if let Some(at_ms) = now() {
        push(Event {
            name: Cow::Borrowed(name),
            frame: current_frame(),
            at_ms,
            cause: crate::cause().0,
            facts: Facts::Counter(value),
        });
    }
}

fn listed(kind: &'static str, name: &str, facts: &[(&'static str, Fact<'_>)]) -> Option<Event> {
    Some(Event {
        name: Cow::Owned(name.to_owned()),
        frame: current_frame(),
        at_ms: now()?,
        cause: crate::cause().0,
        facts: Facts::Listed {
            kind,
            facts: facts
                .iter()
                .map(|&(key, fact)| (key, Field::from(fact)))
                .collect(),
        },
    })
}

pub fn event(kind: &'static str, name: &str, facts: &[(&'static str, Fact<'_>)]) {
    if !recording() {
        return;
    }
    if let Some(event) = listed(kind, name, facts) {
        push(event);
    }
}

pub(crate) fn opened(origin: &str, facts: &[(&'static str, Fact<'_>)]) {
    let Some(event) = listed("cause", origin, facts) else {
        return;
    };
    let mut sink = sink();
    if RECORDING.load(Relaxed) {
        held(&mut sink, event);
    } else {
        sink.opened.push(event);
    }
}

#[derive(Default)]
pub struct Runs {
    runs: AtomicU64,
    nanos: AtomicU64,
    #[cfg(feature = "profiling")]
    allocations: AtomicU64,
    #[cfg(feature = "profiling")]
    bytes: AtomicU64,
}

impl Runs {
    pub fn runs(&self) -> Result<u64, &'static str> {
        Ok(self.runs.load(Relaxed))
    }

    pub fn ns(&self) -> Result<u64, &'static str> {
        if CLOCK.get().is_none() {
            return Err("clock_not_installed");
        }
        Ok(self.nanos.load(Relaxed))
    }

    #[cfg(feature = "profiling")]
    pub fn allocations(&self) -> Result<u64, &'static str> {
        Ok(self.allocations.load(Relaxed))
    }

    #[cfg(feature = "profiling")]
    pub fn allocated_bytes(&self) -> Result<u64, &'static str> {
        Ok(self.bytes.load(Relaxed))
    }

    #[cfg(not(feature = "profiling"))]
    pub fn allocations(&self) -> Result<u64, &'static str> {
        Err(UNCOUNTED_HEAP)
    }

    #[cfg(not(feature = "profiling"))]
    pub fn allocated_bytes(&self) -> Result<u64, &'static str> {
        Err(UNCOUNTED_HEAP)
    }
}

pub fn run<T>(
    runs: &Runs,
    category: &'static str,
    name: &'static str,
    work: impl FnOnce() -> T,
) -> T {
    runs.runs.fetch_add(1, Relaxed);
    let Some(&clock) = CLOCK.get() else {
        return work();
    };
    let spanned_here = recording();
    if spanned_here {
        enter();
    }
    let stamped = (current_frame(), crate::cause().0);
    #[cfg(feature = "profiling")]
    let heap = crate::allocator::thread_heap();
    let start = clock();
    let done = work();
    let end = clock();
    #[cfg(feature = "profiling")]
    {
        let (allocations, bytes) = crate::allocator::thread_heap();
        runs.allocations
            .fetch_add(allocations.wrapping_sub(heap.0), Relaxed);
        runs.bytes.fetch_add(bytes.wrapping_sub(heap.1), Relaxed);
    }
    runs.nanos
        .fetch_add(((end - start).max(0.0) * 1e6) as u64, Relaxed);
    if spanned_here {
        leave();
        spanned((category, Cow::Borrowed(name)), stamped, (start, end));
    }
    done
}
