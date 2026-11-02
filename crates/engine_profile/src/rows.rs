use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::Row;

const SPEED_ROUNDS: u32 = 4096;
const SLOW_MS: f64 = 4.0;
const SLOW_KEPT: usize = 1 << 18;

static ROWS: Mutex<Vec<Row>> = Mutex::new(Vec::new());
static WAITS: Mutex<Vec<Row>> = Mutex::new(Vec::new());
static SPEED: Mutex<Vec<f64>> = Mutex::new(Vec::new());
static SLOW: Mutex<Vec<(u64, &'static str, f64)>> = Mutex::new(Vec::new());
static FRAME: AtomicU64 = AtomicU64::new(0);

fn held<T>(table: &'static Mutex<T>) -> MutexGuard<'static, T> {
    table.lock().unwrap_or_else(PoisonError::into_inner)
}

fn found(rows: &mut Vec<Row>, name: &'static str) -> usize {
    if let Some(at) = rows
        .iter()
        .position(|row| std::ptr::eq(row.name, name) || row.name == name)
    {
        return at;
    }
    rows.push(Row {
        name,
        ms: 0.0,
        worst_ms: 0.0,
        runs: 0,
        allocations: 0,
    });
    rows.len() - 1
}

#[cfg(target_arch = "wasm32")]
pub fn now_ms() -> f64 {
    0.0
}

#[cfg(not(target_arch = "wasm32"))]
pub fn now_ms() -> f64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f64()
        * 1000.0
}

pub fn row(name: &'static str) -> usize {
    found(&mut held(&ROWS), name)
}

pub struct RowScope {
    row: usize,
    start: f64,
    allocations: u64,
}

pub fn timed(row: usize) -> RowScope {
    RowScope {
        row,
        start: now_ms(),
        allocations: crate::allocation_snapshot().allocations,
    }
}

impl Drop for RowScope {
    fn drop(&mut self) {
        let ms = now_ms() - self.start;
        let allocations = crate::allocation_snapshot().allocations - self.allocations;
        let Some(name) = held(&ROWS).get_mut(self.row).map(|row| {
            row.ms += ms;
            row.worst_ms = row.worst_ms.max(ms);
            row.runs += 1;
            row.allocations += allocations;
            row.name
        }) else {
            return;
        };
        if ms >= SLOW_MS {
            let mut slow = held(&SLOW);
            if slow.len() < SLOW_KEPT {
                slow.push((FRAME.load(Relaxed), name, ms));
            }
        }
    }
}

pub fn tally(name: &'static str, amount: u64) {
    if amount > 0 {
        let mut rows = held(&ROWS);
        let at = found(&mut rows, name);
        rows[at].runs += amount;
    }
}

pub fn waited(kind: &str, asked: f64) {
    let ms = now_ms() - asked;
    let mut waits = held(&WAITS);
    let at = match waits.iter().position(|wait| wait.name == kind) {
        Some(at) => at,
        None => found(&mut waits, Box::leak(kind.to_owned().into_boxed_str())),
    };
    let wait = &mut waits[at];
    wait.ms += ms;
    wait.worst_ms = wait.worst_ms.max(ms);
    wait.runs += 1;
}

pub fn mark_speed() {
    FRAME.fetch_add(1, Relaxed);
    let start = now_ms();
    let mut value = 0x9e37_79b9_7f4a_7c15_u64;
    for _ in 0..SPEED_ROUNDS {
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        value = std::hint::black_box(value);
    }
    let ms = now_ms() - start;
    held(&SPEED).push(ms);
}

pub fn take_rows() -> Vec<Row> {
    let rows = held(&ROWS);
    let taken = rows.clone();
    drop(rows);
    for row in held(&ROWS).iter_mut() {
        row.ms = 0.0;
        row.worst_ms = 0.0;
        row.runs = 0;
        row.allocations = 0;
    }
    taken
}

pub fn frame_mark() -> u64 {
    FRAME.load(Relaxed)
}

pub fn take_slow() -> Vec<(u64, &'static str, f64)> {
    std::mem::take(&mut *held(&SLOW))
}

pub fn take_waits() -> Vec<Row> {
    std::mem::take(&mut *held(&WAITS))
}

pub fn take_speed() -> Option<(f64, f64)> {
    let mut marks = std::mem::take(&mut *held(&SPEED));
    marks.sort_by(f64::total_cmp);
    let share = |part: usize| marks.get((marks.len() - 1) * part / 10).copied();
    (!marks.is_empty())
        .then(|| share(1).zip(share(9)))
        .flatten()
}
