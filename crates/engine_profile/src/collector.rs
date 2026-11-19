use std::borrow::Cow;
use std::cell::Cell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::runs::now;

#[derive(Clone, Debug)]
pub enum Facts {
    Span {
        category: &'static str,
        ms: f64,
        depth: u32,
    },
    Counter(f64),
}

#[derive(Clone, Debug)]
pub struct Event {
    pub name: Cow<'static, str>,
    pub at_ms: f64,
    pub facts: Facts,
}

struct Kept {
    limit: usize,
    dropped: u64,
    events: Vec<Event>,
}

static KEPT: Mutex<Option<Kept>> = Mutex::new(None);
static KEEPING: AtomicBool = AtomicBool::new(false);

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

fn kept() -> MutexGuard<'static, Option<Kept>> {
    KEPT.lock().unwrap_or_else(PoisonError::into_inner)
}

fn keeping() -> bool {
    KEEPING.load(Relaxed)
}

fn push(event: Event) {
    if let Some(kept) = kept().as_mut() {
        if kept.events.len() < kept.limit {
            kept.events.push(event);
        } else {
            kept.dropped = kept.dropped.saturating_add(1);
        }
    }
}

pub fn keep(limit: usize) {
    *kept() = Some(Kept {
        limit,
        dropped: 0,
        events: Vec::with_capacity(limit),
    });
    KEEPING.store(true, Relaxed);
}

pub fn take_kept() -> (Vec<Event>, u64) {
    KEEPING.store(false, Relaxed);
    kept()
        .take()
        .map_or((Vec::new(), 0), |kept| (kept.events, kept.dropped))
}

pub struct Span {
    started: Option<(&'static str, Cow<'static, str>, f64)>,
    thread_bound: PhantomData<Rc<()>>,
}

pub fn span(category: &'static str, name: &'static str) -> Span {
    named_span(category, || Cow::Borrowed(name))
}

pub fn span_of(category: &'static str, name: impl std::fmt::Display) -> Span {
    named_span(category, || Cow::Owned(name.to_string()))
}

fn named_span(category: &'static str, name: impl FnOnce() -> Cow<'static, str>) -> Span {
    let started = keeping()
        .then(now)
        .flatten()
        .map(|start| (category, name(), start));
    if started.is_some() {
        DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
    }
    Span {
        started,
        thread_bound: PhantomData,
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some((category, name, start)) = self.started.take() {
            let depth = DEPTH.with(|depth| {
                depth.set(depth.get().saturating_sub(1));
                depth.get()
            });
            if let Some(end) = now() {
                push(Event {
                    name,
                    at_ms: start,
                    facts: Facts::Span {
                        category,
                        ms: (end - start).max(0.0),
                        depth,
                    },
                });
            }
        }
    }
}

pub fn counter(name: &'static str, value: f64) {
    if !keeping() || !value.is_finite() {
        return;
    }
    if let Some(at_ms) = now() {
        push(Event {
            name: Cow::Borrowed(name),
            at_ms,
            facts: Facts::Counter(value),
        });
    }
}
