//! Loading a terrain's tiles: after its bakes, one planner turns the tiles
//! a frame finds missing into tasks, most urgent first, and one of two
//! executors carries them out: the frame itself within its work time
//! ([`InPlace`]), or the host's workers within the job budget
//! ([`OnWorkers`]). Tiles of terrain files decode as they arrive instead
//! (`files`).
//!
//! The planner hands either executor the same tasks: the missing tiles,
//! and the objects standing far away that resident tiles still lack, each
//! where its view ranks it. Each executor keeps its own policy: which tasks
//! wait (under way, or failed a moment ago), how a tile generates (staged,
//! in bands of rows, its objects standing far away after it) and what it
//! does ahead of need. A terrain whose workers start or stop switches
//! executors, the one taking over dropping what the other left under way.

mod in_place;
mod workers;

use in_place::InPlace;
use workers::OnWorkers;
pub(super) use workers::{SharedRecipe, share_recipe};

use super::*;

/// A frame's terrain work: slices of samples' worth handed
/// out while its time lasts, or a single slice without a clock.
pub(super) struct Work {
    pub(super) slice: u64,
    pub(super) clock: Option<fn() -> f64>,
    pub(super) end: f64,
    /// Bytes tiles may still bring the GPU this frame.
    pub(super) upload: std::cell::Cell<u64>,
    /// The most bands a tile under the view splits into.
    pub(super) bands: usize,
}

impl Work {
    /// Refills `budget` with a slice once spent, while the frame's time
    /// lasts; whether it did.
    pub(super) fn refill(&self, budget: &mut u64) -> bool {
        if *budget > 0 {
            return false;
        }
        match self.clock {
            Some(clock) if clock() < self.end => {
                *budget = self.slice;
                true
            }
            _ => false,
        }
    }

    /// Whether the work's time is up; never without a clock.
    pub(super) fn expired(&self) -> bool {
        self.clock.is_some_and(|clock| clock() >= self.end)
    }

    /// Whether tiles may still become resident within the frame's upload
    /// budget.
    pub(super) fn uploads(&self) -> bool {
        self.upload.get() > 0
    }

    /// Spends `bytes` of the frame's upload budget.
    pub(super) fn upload(&self, bytes: u64) {
        self.upload.set(self.upload.get().saturating_sub(bytes));
    }
}

/// Loads `missing` tiles of `state`, most important first, within `work`'s
/// time from slices of `budget`: generated after the terrain's bakes,
/// decoded, or requested and decoded once delivered; `target` is every tile
/// it loads and `keep` every tile it keeps. Frames load what they find
/// missing, and hosts what is left between frames. Where
/// `slots` has workers, they bake and generate instead, each
/// tile at its job priority. Returns whether a bake is still
/// in progress, which nothing loads behind.
#[expect(clippy::too_many_arguments, reason = "the frame's disjoint borrows")]
pub(super) fn load_tiles(
    state: &mut TerrainState,
    missing: &[(TileKey, u32)],
    (target, keep): (&HashSet<TileKey>, &HashSet<TileKey>),
    work: &Work,
    budget: &mut u64,
    frame: u64,
    requests: &mut Vec<TileRequest>,
    mut slots: Option<&mut JobBudget<'_>>,
) -> bool {
    match state.terrain.source.clone() {
        TerrainSource::Generated(authored) => {
            let layers = state.baked.clone().unwrap_or(authored);
            if bake(state, &layers, work, budget, slots.as_deref_mut()) {
                return true;
            }
            // Edits stand over every bake.
            let layers = state.generation(layers);
            // Workers generate tiles where they run, unless
            // they failed it again and again.
            if let Some(slots) = slots.as_deref_mut()
                && slots.jobs.workers() > 0
                && state.failures.count < WORKER_FAILURES
                && let Some(mut workers) = OnWorkers::new(state, (target, keep), work, slots, frame)
            {
                generate(state, missing, &mut workers);
                return false;
            }
            generate(
                state,
                missing,
                &mut InPlace::new(layers, work, budget, slots),
            );
        }
        TerrainSource::File {
            header,
            url: _,
            data: Some(data),
        } => decode_file(state, (&header, &data), missing, (work, budget), frame),
        TerrainSource::File {
            header,
            url,
            data: None,
        } => fetch_file(
            state,
            (&header, &url),
            (missing, target),
            (work, budget),
            frame,
            requests,
        ),
    }
    false
}

/// What a frame wants done about one tile: its key and job priority, and
/// whether only the objects standing far away on it are wanted, the tile
/// resident without them.
#[derive(Clone, Copy, Debug)]
pub(super) struct TileTask {
    pub(super) key: TileKey,
    pub(super) priority: u32,
    pub(super) far: bool,
}

/// Where a terrain's generated tiles are made, and how: each step of a
/// frame's generating, which [`generate`] runs in order.
pub(super) trait Executor {
    /// Drops what the other executor left under way: tiles generate here
    /// from now on.
    fn take_over(&mut self, state: &mut TerrainState);

    /// Takes in what finished since the frame before, and stops what is no
    /// longer wanted.
    fn collect(&mut self, state: &mut TerrainState, missing: &[(TileKey, u32)]);

    /// Whether `task` waits this frame: under way here already, failed a
    /// moment ago, or nothing this executor does.
    fn waits(&self, state: &TerrainState, task: &TileTask) -> bool;

    /// Before the frame's tasks start: what deciding each one needs to know
    /// of them all.
    fn begin(&mut self, state: &TerrainState, tasks: &[TileTask]);

    /// Starts or continues `task`; false once the frame can start no more.
    fn start(&mut self, state: &mut TerrainState, task: TileTask) -> bool;

    /// With the frame's tasks started, work ahead of need, where nothing
    /// else is wanted.
    fn ahead(&mut self, state: &mut TerrainState, missing: &[(TileKey, u32)]);
}

/// Generates `state`'s `missing` tiles with `executor`: it takes over,
/// takes in what finished, starts the planner's tasks most urgent first
/// until the frame can start no more, then works ahead of need.
pub(super) fn generate(
    state: &mut TerrainState,
    missing: &[(TileKey, u32)],
    executor: &mut impl Executor,
) {
    executor.take_over(state);
    executor.collect(state, missing);
    let tasks = tasks(state, missing, executor);
    executor.begin(state, &tasks);
    for task in tasks {
        if !executor.start(state, task) {
            break;
        }
    }
    executor.ahead(state, missing);
}

/// The planner: `state`'s `missing` tiles, and the objects standing far
/// away its resident tiles want, ranked after the urgent tiles where their
/// tiles ranked; most urgent first, each unless `executor` says it waits.
fn tasks(
    state: &TerrainState,
    missing: &[(TileKey, u32)],
    executor: &impl Executor,
) -> Vec<TileTask> {
    let mut tasks: Vec<TileTask> = missing
        .iter()
        .map(|&(key, priority)| TileTask {
            key,
            priority,
            far: false,
        })
        .chain(state.far_wanted.iter().map(|(&key, &priority)| TileTask {
            key,
            priority: job_priority(VIEW_JOBS, priority & 0xff_ffff),
            far: true,
        }))
        .filter(|task| !executor.waits(state, task))
        .collect();
    tasks.sort_by_key(|task| (task.priority, task.far, task.key));
    tasks
}
