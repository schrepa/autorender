use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub const MAX_BRICKS: usize = 64;
pub const CORE: u16 = 0;
const DEPTH: usize = 32;
pub(crate) const CALIBRATION_PAIRS: u32 = 4_096;

static CLOCK: OnceLock<fn() -> f64> = OnceLock::new();
static SCOPE_MS: OnceLock<f64> = OnceLock::new();
static ALLOCATION_MS: OnceLock<f64> = OnceLock::new();

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

#[derive(Clone, Copy)]
struct Clocked {
    open: [Open; DEPTH],
    depth: usize,
    spent: [f64; MAX_BRICKS],
    scopes: u64,
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
            scopes: 0,
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
        self.scopes += 1;
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
    let clock = *CLOCK.get_or_init(|| clock);
    SCOPE_MS.get_or_init(|| scope_calibration(clock));
    #[cfg(feature = "enabled")]
    ALLOCATION_MS.get_or_init(|| crate::allocator::tracking_calibration(clock));
}

fn scope_calibration(clock: fn() -> f64) -> f64 {
    let saved = CLOCKED.with_borrow(|clocked| *clocked);
    let start = clock();
    for _ in 0..CALIBRATION_PAIRS {
        drop(std::hint::black_box(enter(CORE)));
    }
    let elapsed = clock() - start;
    CLOCKED.with_borrow_mut(|clocked| *clocked = saved);
    (elapsed / f64::from(CALIBRATION_PAIRS)).max(0.0)
}

pub fn scope_cost_ms() -> f64 {
    SCOPE_MS.get().copied().unwrap_or(0.0)
}

pub fn allocation_cost_ms() -> f64 {
    ALLOCATION_MS.get().copied().unwrap_or(0.0)
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

pub fn take_scopes() -> u64 {
    CLOCKED.with_borrow_mut(|clocked| std::mem::take(&mut clocked.scopes))
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
