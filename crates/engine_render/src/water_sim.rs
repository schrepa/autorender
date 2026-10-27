//! Water running live (`water_sim.wgsl`): rivers running over a terrain's
//! [`WaterWindow`] by the shallow-water equations, the same water as the
//! settled water around it, starting from the water as it settled there,
//! which also stands just past its edges and which its rim is pulled back
//! to, and pushed by the solids standing in it. Water's pass draws it as one
//! with the tiles' water, reading each cell's look (`meshes/water.wgsl`'s
//! `water_live_read`). The window is one slot of the scheme's buffers
//! ([`crate::water_kernels`]) whose pool holds its blocks and as many more
//! along each side as it moves by ([`WaterWindow::snap`]), each block in the
//! slot of its place on the terrain's lattice of blocks taken around the
//! pool's side, with a ring past its edges. Every frame steps every cell by
//! the same seconds, as many steps as the simulation time passed holds, at
//! most those of a frame LONGEST_FRAME long, within a fixed number of
//! dispatches, and then finds each cell's look. As the window follows the
//! detail eye a block keeps its slot, and the water there runs on: only the
//! blocks it newly covers take their ground, their water as it settled and
//! what their beds hold, while its table, its rivers' sources and its ring
//! are written anew. The window it moves to next, gathered ahead, takes its
//! new blocks into the slots the running window leaves free a few a frame,
//! so they are there when it takes the window's place. Blocks taken are
//! staged in one write the frame's encoding copies into their slots before
//! the steps, and a new window runs once every block of it is there, taken
//! a few a frame. Before each step the solids standing in its water push it
//! toward their own velocity and churn its foam, which rides its surface,
//! dragged by the scene's wind. Where one of the device's storage bindings
//! cannot hold a buffer of the window, its middle square whose buffers they
//! hold runs instead.
//!
//! A meter cell's step is 1/112.5 of a second ([`step_seconds`]), so a frame
//! at 60 Hz steps about twice. On a view 1080 pixels tall through 55
//! degrees, the window of meter cells is 568 cells along each side (the
//! live reach of 259 meters and three quarters of a 32-meter snap past it),
//! 322,624 cells in a pool of 75 × 75 blocks; on one 1440 pixels tall, 744
//! cells (a reach of 346 meters), 553,536 cells in a pool of 97 × 97 blocks.
//! Its buffers take about 133 bytes a cell of its pool: about 48 MB and 80
//! MB.

use std::cell::Cell;
use std::sync::Arc;

use bytemuck::Zeroable;
use engine_core::glam::{Vec2, Vec3};
use engine_core::terrain_data::shallow_water::{UNHELD, step_seconds};
use engine_core::terrain_data::{BLOCK_CELLS, BLOCK_GROUND, BLOCK_SIDE, StoredBlocks};
use engine_core::{SIM_TOUCHES, WaterObstacle, WaterWindow, Wind};

use crate::gpu_layouts::constants::{SIM_LIVE, SIM_LIVE_LOOKS, SIM_STEPPING};
use crate::gpu_layouts::water::{SimSlot, SimTouch, SimTouches};
use crate::water_kernels::{
    Capacity, Kernels, SlotBuffers, encode_drawn, encode_steps, live_largest_bytes,
};

/// The longest frame whose time the live water steps through: a longer
/// frame steps as many steps as this one holds, so a slow frame costs no
/// more than this, and the water then runs slower than the clock.
const LONGEST_FRAME: f64 = 1.0 / 30.0;
/// Blocks a frame takes into its window's slots while they are not needed
/// yet, a new window's before it runs and the blocks the window moves to
/// next: about a megabyte staged, a millisecond or so of the main thread's.
/// Those a running window moves onto that are not there yet are taken at
/// once.
const TAKEN: usize = 256;
/// What a slot holding no block of the window holds as its place: past any
/// window's blocks, so its cells run no step.
const NO_PLACE: u32 = u32::MAX;
/// Bytes a block takes of the ground, of a state of the cells and of its
/// cells' looks.
const BLOCK_GROUND_BYTES: u64 = (BLOCK_GROUND * 4) as u64;
const BLOCK_CELL_BYTES: u64 = (BLOCK_CELLS * 16) as u64;
const BLOCK_LOOK_BYTES: u64 = BLOCK_CELLS as u64 * SIM_LIVE_LOOKS * 16;

/// A terrain's live window as frames hand it over: the window, the window
/// waiting ahead to take its place, the solids standing in it (nearest the
/// eye first) and how many more stand there.
pub(crate) struct Live<'a> {
    pub(crate) window: &'a Arc<WaterWindow>,
    pub(crate) ahead: Option<&'a Arc<WaterWindow>>,
    pub(crate) touches: &'a [WaterObstacle],
    pub(crate) beyond: u32,
}

/// A terrain's water running live over its window: the scheme's passes,
/// the window's buffers, the steps this frame runs, the step of the
/// simulation's time the last frame stepped to, a count advancing whenever
/// what water draws from changes, and the solids pushing its water this
/// frame with those standing in its window past the most that do.
pub(crate) struct WaterSim {
    kernels: Kernels,
    window: Option<SimWindow>,
    steps: u32,
    tick: Option<i64>,
    generation: u64,
    touches: [u32; 2],
}

/// A window as terrains placed it and the square of it that runs (all of
/// it, where the device's bindings hold its buffers); the window waiting
/// ahead and its square of the same size; the slots along each side of the
/// pool, the square's blocks (each in the slot of its place on the
/// terrain's lattice of blocks taken around the pool's side) and the slots
/// they turn by along x and z; its buffers (one slot with its ring) and the
/// count of cells rivers bring water to; the stamp of the block each slot
/// holds (0 where it holds none it knows); whether every block of the
/// square is there, so it runs; what this frame takes into its slots; and
/// the buffer it stages them in.
struct SimWindow {
    source: Arc<WaterWindow>,
    fitted: Arc<WaterWindow>,
    ahead: Option<(Arc<WaterWindow>, Arc<WaterWindow>)>,
    pool: u32,
    stored: StoredBlocks,
    wrap: [u32; 2],
    buffers: SlotBuffers,
    sourced: u32,
    stamps: Vec<u64>,
    ready: bool,
    taken: Option<Taken>,
    stage: Option<wgpu::Buffer>,
}

/// The blocks a frame takes into its window's slots, staged in their
/// slots' order, all their ground, then all their cells' start, then all
/// their cells' looks: the runs of consecutive slots they fill, each its
/// first slot, its slots and its first block staged; the blocks staged; and
/// whether the frame's encoding recorded their copies.
struct Taken {
    runs: Vec<[u32; 3]>,
    staged: u32,
    recorded: Cell<bool>,
}

impl WaterSim {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            kernels: Kernels::new(device),
            window: None,
            steps: 0,
            tick: None,
            generation: 0,
            touches: [0; 2],
        }
    }

    /// Prepares the live window `live` for this frame at simulation time
    /// `seconds` in the scene's wind `wind`, over the square of it the
    /// device's bindings hold: its buffers anew where a square of another
    /// size came; where one of its size moved, its table, sources and ring;
    /// in its slots the blocks they do not hold, and the blocks of the
    /// window waiting ahead the square leaves slots free for; the solids
    /// pushing its water; the wind dragging its surface; and as many steps,
    /// each of the same seconds, as the simulation time since the last frame
    /// holds, at most LONGEST_FRAME's. Drops it where there is no window, or
    /// no square of it fits.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        live: Option<Live<'_>>,
        (wind, seconds): (Wind, f64),
    ) {
        if let Some(current) = self.window.as_mut() {
            current.settle_taken();
        }
        // The window, with the square of it that runs where it changed.
        let came = live.and_then(|live| {
            let changed = self
                .window
                .as_ref()
                .is_none_or(|current| !Arc::ptr_eq(&current.source, live.window));
            if !changed {
                return Some((live, None));
            }
            fit(device, live.window).map(|square| (live, Some(square)))
        });
        let Some((live, fitted)) = came else {
            if self.window.take().is_some() {
                self.generation += 1;
            }
            (self.steps, self.tick, self.touches) = (0, None, [0; 2]);
            return;
        };
        self.kernels.pipelines(device);
        let layout = &self.kernels.layout;
        if let Some(fitted) = fitted {
            let same_size = self.window.as_ref().is_some_and(|current| {
                current.fitted.cells == fitted.cells
                    && current.fitted.cell == fitted.cell
                    && current.pool == pool_of(&fitted)
            });
            if same_size {
                if let Some(current) = self.window.as_mut() {
                    current.follow((device, queue, layout), live.window, fitted);
                }
            } else {
                self.window = Some(SimWindow::new((device, queue, layout), live.window, fitted));
                self.generation += 1;
            }
        }
        let Some(current) = self.window.as_mut() else {
            return;
        };
        current.wait_ahead(live.ahead);
        let ready = current.ready;
        current.take(device, queue);
        if current.ready != ready {
            self.generation += 1;
        }
        // The steps: as many of a step's seconds as the simulation's time
        // has passed through since the last frame, at most LONGEST_FRAME's.
        let (cells, cell) = (current.fitted.cells, current.fitted.cell);
        let step = step_seconds(cell);
        let tick = (seconds / f64::from(step)).floor() as i64;
        let most = (LONGEST_FRAME / f64::from(step)).ceil() as i64;
        let steps = self.tick.map_or(0, |last| (tick - last).clamp(0, most));
        self.tick = Some(tick);
        self.steps = if current.ready { steps as u32 } else { 0 };
        let pushing = touching(&current.fitted, live.touches);
        self.touches = [
            pushing.as_ref().map_or(0, |pushing| pushing.count[0]),
            live.beyond,
        ];
        let [columns, rows] = current.stored.blocks;
        let pooled = current.stored.stored() as u32;
        // The wind's mean drags the surface: its gusts come and go within
        // seconds (`Wind::at`), carrying what floats back and forth by a few
        // centimeters in a breeze.
        current.buffers.write_frames(queue, wind.velocity);
        current.buffers.write_squares(
            queue,
            &[SimSlot {
                shape: [cells, cells, SIM_STEPPING | SIM_LIVE, current.sourced],
                step: [cell, step, 0.0, 0.0],
                blocks: [columns, rows, pooled, 0],
            }],
        );
        current.buffers.write_touches(queue, pushing.as_ref());
    }

    /// Copies the blocks this frame took into the window's slots, then,
    /// once every block of its square is there, runs the steps prepared and
    /// finds each cell's look, timed by `profiler` (`water_sim`).
    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        let (Some(pipelines), Some(window)) = (self.kernels.created(), &self.window) else {
            return;
        };
        window.copy_taken(encoder);
        if !window.ready {
            return;
        }
        let blocks = window.stored.stored() as u32;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/water-sim"),
            timestamp_writes: profiler.span("water_sim").map(|span| span.compute()),
        });
        if self.steps > 0 {
            encode_steps(
                &mut pass,
                pipelines,
                &window.buffers,
                (self.steps, window.sourced, false, false),
                blocks,
            );
        }
        encode_drawn(&mut pass, pipelines, &window.buffers, blocks);
    }

    /// What water draws the live water from, once every block of its square
    /// is there: the square of the window that runs, the slots its blocks
    /// turn by, its pool's slots along each side and its cells' looks; and a
    /// count advancing whenever they change.
    pub(crate) fn drawn(&self) -> (Option<SimDrawn<'_>>, u64) {
        let drawn = self
            .window
            .as_ref()
            .filter(|window| window.ready)
            .map(|window| SimDrawn {
                window: window.fitted.as_ref(),
                wrap: window.wrap,
                pool: window.pool,
                looks: &window.buffers.averages,
            });
        (drawn, self.generation)
    }

    /// The cells along a side of the square that runs and those the
    /// device's bindings cut from the window, and the steps this frame
    /// runs.
    pub(crate) fn cells(&self) -> [u32; 3] {
        self.window.as_ref().filter(|window| window.ready).map_or([0; 3], |window| {
            [
                window.fitted.cells,
                window.source.cells - window.fitted.cells,
                self.steps,
            ]
        })
    }

    /// The solids pushing its water this frame, and those standing in its
    /// window past the most that do, which push none.
    pub(crate) fn touches(&self) -> [u32; 2] {
        self.touches
    }

    /// Bytes of the window's buffers.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.window.as_ref().map_or(0, |window| {
            window.buffers.bytes() + window.stage.as_ref().map_or(0, wgpu::Buffer::size)
        })
    }
}

impl SimWindow {
    /// The buffers `fitted`, the square of `source` that runs, runs in, its
    /// pool holding as many more blocks along each side as it moves by, with
    /// its table, sources and ring; its slots take its blocks a few a frame
    /// before it runs.
    fn new(
        (device, queue, layout): (&wgpu::Device, &wgpu::Queue, &crate::pipelines::GroupLayout),
        source: &Arc<WaterWindow>,
        fitted: Arc<WaterWindow>,
    ) -> Self {
        let pool = pool_of(&fitted);
        let (stored, wrap) = wrapped(&fitted, pool);
        let slots = stored.stored();
        let buffers = SlotBuffers::new(
            device,
            layout,
            (Capacity::of(&stored, fitted.sources.len().max(1)), 1),
            true,
        );
        let mut window = Self {
            source: source.clone(),
            fitted,
            ahead: None,
            pool,
            stored,
            wrap,
            buffers,
            sourced: 0,
            stamps: vec![0; slots],
            ready: false,
            taken: None,
            stage: None,
        };
        window.write_square((device, queue, layout));
        window
    }

    /// Takes `fitted`, the square of `source` that runs, of its square's
    /// size, in its square's place: its table turned to its place, and its
    /// sources and ring. Its slots keep the blocks both squares hold and
    /// those taken ahead, and take the rest with the frame.
    fn follow(
        &mut self,
        gpu: (&wgpu::Device, &wgpu::Queue, &crate::pipelines::GroupLayout),
        source: &Arc<WaterWindow>,
        fitted: Arc<WaterWindow>,
    ) {
        (self.stored, self.wrap) = wrapped(&fitted, self.pool);
        self.source = source.clone();
        self.fitted = fitted;
        self.write_square(gpu);
    }

    /// Writes its square's table, its rivers' sources and its ring, its
    /// inputs grown first where they hold fewer sources.
    fn write_square(
        &mut self,
        (device, queue, layout): (&wgpu::Device, &wgpu::Queue, &crate::pipelines::GroupLayout),
    ) {
        self.buffers
            .hold_sources(device, layout, self.fitted.sources.len());
        self.buffers.write_table(queue, 0, &self.stored);
        self.sourced = write_inputs(queue, &self.buffers, &self.fitted, &self.stored);
    }

    /// Keeps `ahead`, the window waiting to take its window's place, with
    /// its square of the size of its own square (its middle one, where the
    /// device's bindings cut its own); none where it waits for none, or one
    /// of other cells or another size.
    fn wait_ahead(&mut self, ahead: Option<&Arc<WaterWindow>>) {
        let Some(ahead) = ahead.filter(|ahead| ahead.cell == self.fitted.cell) else {
            self.ahead = None;
            return;
        };
        if self
            .ahead
            .as_ref()
            .is_some_and(|(source, _)| Arc::ptr_eq(source, ahead))
        {
            return;
        }
        let fitted = if ahead.cells == self.fitted.cells {
            ahead.clone()
        } else {
            Arc::new(ahead.within(self.fitted.cells))
        };
        self.ahead = (fitted.cells == self.fitted.cells).then(|| (ahead.clone(), fitted));
    }

    /// Lets go of what the frame before took: done where its encoding
    /// recorded its copies, else its slots hold no block it knows, to take
    /// again.
    fn settle_taken(&mut self) {
        let Some(taken) = self.taken.take() else {
            return;
        };
        if taken.recorded.get() {
            return;
        }
        for &[first, slots, _] in &taken.runs {
            self.stamps[first as usize..(first + slots) as usize].fill(0);
        }
    }

    /// Takes into its slots the blocks of its square they do not hold, and
    /// of the square waiting ahead those the square leaves slots free for:
    /// each one's ground (at its cells' corners, the levels lakes hold its
    /// cells at, UNHELD where none does, its cells' roughness, and the share
    /// of each covered by solids and the height their tops stand at), its
    /// cells' water to start from, into the step's first state, and its
    /// cells' looks (the water each settled to, and what its bed holds under
    /// it, its span's and reach's codes in the top bytes of its words, as
    /// water's pass reads them). A running square takes all of its own at
    /// once, a square not yet running TAKEN of them a frame, and the square
    /// waiting ahead what is left of TAKEN; it runs once all of its own are
    /// there. They are staged in one write, in their slots' order, for the
    /// frame's encoding to copy into their slots a run of consecutive slots
    /// at a time.
    fn take(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (side, pool) = (self.fitted.blocks(), self.pool);
        // Each block taken: its slot, whether it is the square's ahead, and
        // its column and row.
        let mut fresh: Vec<(u32, bool, u32, u32)> = Vec::new();
        let mut missing = 0;
        for z in 0..side {
            for x in 0..side {
                let slot = self.stored.table[(z * side + x) as usize];
                if self.stamps[slot as usize] == self.fitted.block(x, z).stamp() {
                    continue;
                }
                missing += 1;
                if self.ready || fresh.len() < TAKEN {
                    fresh.push((slot, false, x, z));
                }
            }
        }
        self.ready = fresh.len() == missing;
        if let Some((_, ahead)) = self.ahead.as_ref().filter(|_| self.ready) {
            let wrap = wrap_of(ahead, pool);
            let used = |slot: u32| {
                let (x, z) = (slot % pool, slot / pool);
                (x + pool - self.wrap[0]) % pool < side && (z + pool - self.wrap[1]) % pool < side
            };
            let mut left = TAKEN.saturating_sub(fresh.len());
            'ahead: for z in 0..side {
                for x in 0..side {
                    if left == 0 {
                        break 'ahead;
                    }
                    let slot = (z + wrap[1]) % pool * pool + (x + wrap[0]) % pool;
                    if used(slot) || self.stamps[slot as usize] == ahead.block(x, z).stamp() {
                        continue;
                    }
                    fresh.push((slot, true, x, z));
                    left -= 1;
                }
            }
        }
        if fresh.is_empty() {
            return;
        }
        fresh.sort_unstable_by_key(|&(slot, _, _, _)| slot);
        let blocks = fresh.len();
        let mut ground: Vec<f32> = Vec::with_capacity(blocks * BLOCK_GROUND);
        let mut cells: Vec<[f32; 4]> = Vec::with_capacity(blocks * BLOCK_CELLS);
        let mut looks: Vec<[f32; 4]> = Vec::with_capacity(blocks * BLOCK_CELLS * 2);
        let mut held: Vec<f32> = Vec::with_capacity(BLOCK_CELLS);
        let mut rough: Vec<f32> = Vec::with_capacity(BLOCK_CELLS);
        let mut shares: Vec<f32> = Vec::with_capacity(BLOCK_CELLS);
        let mut tops: Vec<f32> = Vec::with_capacity(BLOCK_CELLS);
        let mut stamped: Vec<(u32, u64)> = Vec::with_capacity(blocks);
        for &(slot, ahead, x, z) in &fresh {
            let square = match (&self.ahead, ahead) {
                (Some((_, waiting)), true) => waiting,
                _ => &self.fitted,
            };
            let block = square.block(x, z);
            for k in 0..=BLOCK_SIDE {
                for l in 0..=BLOCK_SIDE {
                    ground.push(block.corner(l, k));
                }
            }
            held.clear();
            rough.clear();
            shares.clear();
            tops.clear();
            for k in 0..BLOCK_SIDE {
                for l in 0..BLOCK_SIDE {
                    let level = block.held(l, k);
                    held.push(if level.is_nan() { UNHELD } else { level });
                    rough.push(block.roughness(l, k));
                    let [share, top] = block.cover(l, k);
                    shares.push(share);
                    tops.push(top);
                    cells.push(block.start(l, k));
                    let bed = block.bed_word(l, k);
                    looks.push(block.settled(l, k));
                    looks.push([
                        0.0,
                        0.0,
                        f32::from_bits(((bed >> 8) & 0xff) << 24),
                        f32::from_bits((bed & 0xff) << 24),
                    ]);
                }
            }
            ground.extend_from_slice(&held);
            ground.extend_from_slice(&rough);
            ground.extend_from_slice(&shares);
            ground.extend_from_slice(&tops);
            stamped.push((slot, block.stamp()));
        }
        for (slot, stamp) in stamped {
            self.stamps[slot as usize] = stamp;
        }
        let mut staged: Vec<u8> = Vec::with_capacity(
            blocks * (BLOCK_GROUND_BYTES + BLOCK_CELL_BYTES + BLOCK_LOOK_BYTES) as usize,
        );
        staged.extend_from_slice(bytemuck::cast_slice(&ground));
        staged.extend_from_slice(bytemuck::cast_slice(&cells));
        staged.extend_from_slice(bytemuck::cast_slice(&looks));
        let bytes = staged.len() as u64;
        if self.stage.as_ref().is_none_or(|stage| stage.size() < bytes) {
            self.stage = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine/water-sim-stage"),
                size: bytes.next_power_of_two(),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let Some(stage) = self.stage.as_ref() else {
            return;
        };
        queue.write_buffer(stage, 0, &staged);
        let mut runs: Vec<[u32; 3]> = Vec::new();
        for (at, &(slot, _, _, _)) in fresh.iter().enumerate() {
            match runs.last_mut() {
                Some(run) if run[0] + run[1] == slot => run[1] += 1,
                _ => runs.push([slot, 1, at as u32]),
            }
        }
        self.taken = Some(Taken {
            runs,
            staged: blocks as u32,
            recorded: Cell::new(false),
        });
    }

    /// Records the copies of the blocks this frame took from their stage
    /// into their slots: their ground, their cells' water into the step's
    /// first state, and their cells' looks.
    fn copy_taken(&self, encoder: &mut wgpu::CommandEncoder) {
        let (Some(taken), Some(stage)) = (&self.taken, &self.stage) else {
            return;
        };
        let staged = u64::from(taken.staged);
        let cells_staged = staged * BLOCK_GROUND_BYTES;
        let looks_staged = cells_staged + staged * BLOCK_CELL_BYTES;
        let buffers = &self.buffers;
        let (ground_at, cells_at, looks_at) = (
            buffers.ground_at(0) * 4,
            buffers.cells_at(0) * 16,
            buffers.averages_at(0) * 16,
        );
        for &[slot, slots, first] in &taken.runs {
            let (slot, slots, first) = (u64::from(slot), u64::from(slots), u64::from(first));
            encoder.copy_buffer_to_buffer(
                stage,
                first * BLOCK_GROUND_BYTES,
                &buffers.ground,
                ground_at + slot * BLOCK_GROUND_BYTES,
                slots * BLOCK_GROUND_BYTES,
            );
            encoder.copy_buffer_to_buffer(
                stage,
                cells_staged + first * BLOCK_CELL_BYTES,
                &buffers.cells,
                cells_at + slot * BLOCK_CELL_BYTES,
                slots * BLOCK_CELL_BYTES,
            );
            encoder.copy_buffer_to_buffer(
                stage,
                looks_staged + first * BLOCK_LOOK_BYTES,
                &buffers.averages,
                looks_at + slot * BLOCK_LOOK_BYTES,
                slots * BLOCK_LOOK_BYTES,
            );
        }
        taken.recorded.set(true);
    }
}

/// Slots along each side of the pool of the live square `window`: its
/// blocks', and as many more as it moves by, which the square it moves to
/// next takes its new blocks into ahead.
fn pool_of(window: &WaterWindow) -> u32 {
    window.blocks() + window.snap
}

/// The slots `window`'s blocks turn by along x and z in a pool of `pool`
/// slots along each side: its first block's place on the terrain's lattice
/// of blocks, taken around the pool's side.
fn wrap_of(window: &WaterWindow, pool: u32) -> [u32; 2] {
    window.first.map(|first| {
        first
            .div_euclid(i64::from(BLOCK_SIDE))
            .rem_euclid(i64::from(pool)) as u32
    })
}

/// The blocks of `window`'s square as its slot stores them in a pool of
/// `pool` slots along each side, and the slots they turn by along x and z:
/// every one, each in the slot of its place on the terrain's lattice of
/// blocks taken around the pool's side, so a block keeps its slot as the
/// window follows the eye, and the square it moves to takes its new blocks
/// into slots it leaves free; those slots hold NO_PLACE.
fn wrapped(window: &WaterWindow, pool: u32) -> (StoredBlocks, [u32; 2]) {
    let side = window.blocks();
    let wrap = wrap_of(window, pool);
    let mut table = vec![0; (side * side) as usize];
    let mut places = vec![NO_PLACE; (pool * pool) as usize];
    for z in 0..side {
        for x in 0..side {
            let slot = (z + wrap[1]) % pool * pool + (x + wrap[0]) % pool;
            table[(z * side + x) as usize] = slot;
            places[slot as usize] = x | (z << 16);
        }
    }
    let stored = StoredBlocks {
        cells: [window.cells; 2],
        blocks: [side; 2],
        table,
        places,
    };
    (stored, wrap)
}

/// The square of `window` whose every buffer one of `device`'s storage
/// bindings holds: the window itself where they hold its own, else its
/// middle square of the most blocks along each side whose buffers they
/// hold; none where they hold not even one.
fn fit(device: &wgpu::Device, window: &Arc<WaterWindow>) -> Option<Arc<WaterWindow>> {
    let limits = device.limits();
    let binding = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    let sources = window.sources.len();
    let fits = |blocks: u32| largest_bytes(blocks, window.snap, sources) <= binding;
    let side = window.blocks();
    if fits(side) {
        return Some(window.clone());
    }
    // The most blocks whose buffers fit lie from `low` (or none) up to
    // before `high`.
    let (mut low, mut high) = (0, side);
    while low + 1 < high {
        let middle = low + (high - low) / 2;
        if fits(middle) {
            low = middle;
        } else {
            high = middle;
        }
    }
    (low >= 1 && fits(low)).then(|| Arc::new(window.within(low * BLOCK_SIDE)))
}

/// The most bytes one buffer of a live square `blocks` blocks along each
/// side takes, in a pool of as many more along each side as it moves by
/// (`snap`), its rivers bringing water to `sources` of its cells: its
/// slot's largest, or its inputs' (an entry for each source and two for
/// each of the `4 × cells` cells of its ring).
fn largest_bytes(blocks: u32, snap: u32, sources: usize) -> u64 {
    let pool = blocks + snap;
    let cells = blocks * BLOCK_SIDE;
    let capacity = Capacity {
        blocks: pool * pool,
        mapped: blocks * blocks,
        edges: 2 * cells,
        sources: sources.max(1) as u32,
    };
    let inputs = (u64::from(capacity.sources) + 8 * u64::from(cells)) * 16;
    live_largest_bytes(capacity).max(inputs)
}

/// Writes what `window`'s rivers bring (at the cells `stored`, its blocks,
/// holds them at) and its ring into `buffers`' inputs; the cells its rivers
/// bring water to.
fn write_inputs(
    queue: &wgpu::Queue,
    buffers: &SlotBuffers,
    window: &WaterWindow,
    stored: &StoredBlocks,
) -> u32 {
    let n = window.cells;
    let sources: Vec<[f32; 4]> = window
        .sources
        .iter()
        .filter_map(|source| {
            let at = stored.at(source.cell % n, source.cell / n)?;
            Some([
                f32::from_bits(at as u32),
                source.rate,
                source.way[0],
                source.way[1],
            ])
        })
        .take(buffers.capacity.sources as usize)
        .collect();
    if !sources.is_empty() {
        queue.write_buffer(&buffers.inputs, 0, bytemuck::cast_slice(&sources));
    }
    queue.write_buffer(
        &buffers.inputs,
        buffers.ring_at(0) * 16,
        bytemuck::cast_slice(&ringing(window)),
    );
    sources.len() as u32
}

/// `window`'s ring as the shader reads it: each cell past its edges'
/// surface, flow along x and z and the level a lake holds it at
/// (`UNHELD` where none does), then its foam and the flow it brings into
/// the window.
fn ringing(window: &WaterWindow) -> Vec<[f32; 4]> {
    window
        .ghosts()
        .iter()
        .flat_map(|ghost| {
            let held = if ghost.held.is_nan() {
                UNHELD
            } else {
                ghost.held
            };
            [
                [ghost.surface, ghost.flow[0], ghost.flow[1], held],
                [ghost.foam, ghost.inflow, 0.0, 0.0],
            ]
        })
        .collect()
}

/// The solids of `touches` whose footprints reach over `window`'s square
/// as its touch pass reads them, nearest the eye first: each one's center,
/// velocity and footprint over the square's first corner and datum, the
/// inverse of its axes and how far it reaches up and down; and the blocks
/// of the square they reach. None where none reaches over it; a flat solid,
/// whose axes have no inverse, fills no water.
fn touching(window: &WaterWindow, touches: &[WaterObstacle]) -> Option<SimTouches> {
    let corner = Vec3::new(window.origin[0], window.datum, window.origin[1]);
    let (low, high) = window.square();
    let size = high - low;
    let mut pushing = SimTouches::zeroed();
    let (mut least, mut most) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
    let mut count = 0;
    for touch in touches.iter().take(SIM_TOUCHES) {
        let reach = touch.reach();
        let center = touch.center - corner;
        let (middle, half) = (Vec2::new(center.x, center.z), Vec2::new(reach.x, reach.z));
        let (from, to) = (middle - half, middle + half);
        let over = to.cmpge(Vec2::ZERO).all() && from.cmple(size).all();
        let fills = touch.axes.determinant().abs() > f32::MIN_POSITIVE;
        if !(over && fills) {
            continue;
        }
        let inverse = touch.axes.inverse();
        pushing.touch[count] = SimTouch {
            center: center
                .extend(if touch.round { 1.0 } else { 0.0 })
                .to_array(),
            inverse: [
                inverse.x_axis.extend(0.0).to_array(),
                inverse.y_axis.extend(0.0).to_array(),
                inverse.z_axis.extend(0.0).to_array(),
            ],
            velocity: touch.velocity.extend(reach.y).to_array(),
            reach: [from.x, from.y, to.x, to.y],
        };
        (least, most) = (least.min(from), most.max(to));
        count += 1;
    }
    if count == 0 {
        return None;
    }
    let block = window.cell * BLOCK_SIDE as f32;
    let last = Vec2::splat((window.blocks() - 1) as f32);
    let first = (least / block).floor().clamp(Vec2::ZERO, last);
    let end = (most / block).floor().clamp(Vec2::ZERO, last);
    pushing.count = [count as u32, 0, 0, 0];
    pushing.blocks = [
        first.x as u32,
        first.y as u32,
        (end.x - first.x) as u32 + 1,
        (end.y - first.y) as u32 + 1,
    ];
    Some(pushing)
}

/// What water draws the live water from ([`WaterSim::drawn`]).
pub(crate) struct SimDrawn<'a> {
    pub(crate) window: &'a WaterWindow,
    pub(crate) wrap: [u32; 2],
    pub(crate) pool: u32,
    pub(crate) looks: &'a wgpu::Buffer,
}
