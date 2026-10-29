use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock, PoisonError};

pub const MAX_BRICKS: usize = 64;
pub const CORE: u16 = 0;
pub const CORE_NAME: &str = "core";
const DEPTH: usize = 32;

static NAMES: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
static CLOCK: OnceLock<fn() -> f64> = OnceLock::new();

pub(crate) static HEAP_LIVE: [AtomicU64; MAX_BRICKS] = [const { AtomicU64::new(0) }; MAX_BRICKS];
pub(crate) static HEAP_PEAK: [AtomicU64; MAX_BRICKS] = [const { AtomicU64::new(0) }; MAX_BRICKS];
pub(crate) static HEAP_ALLOCATIONS: [AtomicU64; MAX_BRICKS] =
    [const { AtomicU64::new(0) }; MAX_BRICKS];

#[derive(Clone, Copy)]
struct Open {
    brick: u16,
    start: f64,
    nested: f64,
}

struct Clocked {
    open: [Open; DEPTH],
    depth: usize,
    spent: [f64; MAX_BRICKS],
}

impl Clocked {
    const fn new() -> Self {
        Self {
            open: [Open {
                brick: CORE,
                start: 0.0,
                nested: 0.0,
            }; DEPTH],
            depth: 0,
            spent: [0.0; MAX_BRICKS],
        }
    }

    fn open(&mut self, brick: u16, now: f64) -> bool {
        if self.depth == DEPTH {
            return false;
        }
        self.open[self.depth] = Open {
            brick,
            start: now,
            nested: 0.0,
        };
        self.depth += 1;
        true
    }

    fn close(&mut self, now: f64) {
        let Some(depth) = self.depth.checked_sub(1) else {
            return;
        };
        self.depth = depth;
        let open = self.open[depth];
        let elapsed = (now - open.start).max(0.0);
        self.spent[slot(open.brick)] += (elapsed - open.nested).max(0.0);
        if let Some(parent) = depth.checked_sub(1) {
            self.open[parent].nested += elapsed;
        }
    }
}

thread_local! {
    static CURRENT: Cell<u16> = const { Cell::new(CORE) };
    static CLOCKED: RefCell<Clocked> = const { RefCell::new(Clocked::new()) };
}

pub(crate) fn slot(brick: u16) -> usize {
    usize::from(brick).min(MAX_BRICKS - 1)
}

pub fn install_clock(clock: fn() -> f64) {
    CLOCK.get_or_init(|| clock);
}

pub fn brick_index(name: &'static str) -> u16 {
    if name == CORE_NAME {
        return CORE;
    }
    let mut names = NAMES.lock().unwrap_or_else(PoisonError::into_inner);
    if names.is_empty() {
        names.push(CORE_NAME);
    }
    if let Some(index) = names.iter().position(|&known| known == name) {
        return index as u16;
    }
    if names.len() == MAX_BRICKS {
        return CORE;
    }
    names.push(name);
    (names.len() - 1) as u16
}

pub fn brick_name(index: u16) -> &'static str {
    NAMES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(usize::from(index))
        .copied()
        .unwrap_or(CORE_NAME)
}

pub fn current_brick() -> u16 {
    CURRENT.try_with(Cell::get).unwrap_or(CORE)
}

pub struct BrickScope {
    previous: u16,
    timed: bool,
    thread_bound: PhantomData<Rc<()>>,
}

pub fn enter(brick: u16) -> BrickScope {
    let previous = CURRENT.replace(brick);
    let timed = CLOCK
        .get()
        .is_some_and(|clock| CLOCKED.with_borrow_mut(|clocked| clocked.open(brick, clock())));
    BrickScope {
        previous,
        timed,
        thread_bound: PhantomData,
    }
}

impl Drop for BrickScope {
    fn drop(&mut self) {
        if self.timed
            && let Some(clock) = CLOCK.get()
        {
            let now = clock();
            CLOCKED.with_borrow_mut(|clocked| clocked.close(now));
        }
        CURRENT.set(self.previous);
    }
}

pub fn take_spent() -> [f64; MAX_BRICKS] {
    CLOCKED.with_borrow_mut(|clocked| std::mem::replace(&mut clocked.spent, [0.0; MAX_BRICKS]))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BrickHeap {
    pub live_bytes: u64,
    pub peak_bytes: u64,
    pub allocations: u64,
}

pub fn brick_heap(index: u16) -> BrickHeap {
    let slot = slot(index);
    BrickHeap {
        live_bytes: HEAP_LIVE[slot].load(Relaxed),
        peak_bytes: HEAP_PEAK[slot].load(Relaxed),
        allocations: HEAP_ALLOCATIONS[slot].load(Relaxed),
    }
}
