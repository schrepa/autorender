//! Water's simulation (`water_sim.wgsl`): rivers running over a terrain's
//! [`WaterWindow`] by the shallow-water equations, a few steps a frame, from
//! the water as it settled there, which also stands just past its edges,
//! pushed by the solids standing in it; the water pass draws its water
//! (`meshes/water.wgsl`'s simulated water) in place of the tiles'. The
//! window is one slot of the scheme's buffers ([`crate::water_kernels`])
//! storing every block of its cells, each in the slot of its place on the
//! terrain's lattice of blocks taken around the window's blocks along each
//! side (water draws by the blocks they turn by), with a ring past its
//! edges, stepping every cell by the frame's seconds. As the window follows
//! the camera a block keeps its slot, and the water there runs on: only the
//! blocks it newly covers take their ground, what their beds hold and their
//! water as it settled, staged in one write the frame's encoding copies
//! into their slots before the steps, while its table, its rivers' sources
//! and its ring are written anew. Before each step the solids standing in
//! its water push it toward their own velocity and churn its foam, which
//! rides its surface, dragged by the scene's wind. Where one of the device's
//! storage bindings cannot hold a buffer of the window, its middle square
//! whose buffers they hold runs instead.

use std::cell::Cell;
use std::sync::Arc;

use bytemuck::Zeroable;
use engine_core::glam::{Vec2, Vec3};
use engine_core::math::{max, min};
use engine_core::terrain_data::shallow_water::{UNHELD, step_seconds};
use engine_core::terrain_data::{BLOCK_CELLS, BLOCK_GROUND, BLOCK_SIDE, StoredBlocks};
use engine_core::{SIM_TOUCHES, WaterObstacle, WaterWindow, Wind};

use crate::gpu_layouts::constants::SIM_STEPPING;
use crate::gpu_layouts::water::{SimSlot, SimTouch, SimTouches};
use crate::water_kernels::{Capacity, Kernels, SlotBuffers, encode_steps, slot_largest_bytes};

/// The most steps a frame runs, and the most seconds a frame simulates:
/// longer frames simulate less time than passed. Eight steps of cells half
/// a meter across ([`step_seconds`]) cover the longest frame.
const STEPS_AT_MOST: f32 = 8.0;
const LONGEST_FRAME: f32 = 1.0 / 30.0;
/// Bytes a block takes of the ground, of a state of the cells and of what
/// the beds hold under them.
const BLOCK_GROUND_BYTES: u64 = (BLOCK_GROUND * 4) as u64;
const BLOCK_CELL_BYTES: u64 = (BLOCK_CELLS * 16) as u64;
const BLOCK_BED_BYTES: u64 = (BLOCK_CELLS * 4) as u64;

/// A terrain's water simulated over its window: the scheme's passes, the
/// window's buffers, the steps this frame runs, when the last frame was, a
/// count advancing whenever the buffers water draws from change, and the
/// solids pushing its water this frame with those standing in its window
/// past the most that do.
pub(crate) struct WaterSim {
    kernels: Kernels,
    window: Option<SimWindow>,
    steps: u32,
    last: Option<f64>,
    generation: u64,
    touches: [u32; 2],
}

/// A window as terrains placed it and the square of it that runs (all of
/// it, where the device's bindings hold its buffers); the square's blocks
/// (every one, each in the slot of its place on the terrain's lattice of
/// blocks taken around its blocks along each side) and the blocks they
/// turn by along x and z; its buffers (one slot with its ring), the count
/// of cells rivers bring water to, the triangles water draws its cells'
/// middles with and what the beds hold under its cells in its slots' order,
/// which water draws them with; the stamp of the block each slot holds (0
/// where it holds none it knows); what this frame takes into its slots;
/// and the buffer it stages them in.
struct SimWindow {
    source: Arc<WaterWindow>,
    fitted: Arc<WaterWindow>,
    stored: StoredBlocks,
    wrap: [u32; 2],
    buffers: SlotBuffers,
    sourced: u32,
    indices: wgpu::Buffer,
    beds: wgpu::Buffer,
    stamps: Vec<u64>,
    taken: Option<Taken>,
    stage: Option<wgpu::Buffer>,
}

/// The blocks a frame takes into its window's slots, staged in their
/// slots' order, all their ground, then all their cells' start, then all
/// that their beds hold: the runs of consecutive slots they fill, each its
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
            last: None,
            generation: 0,
            touches: [0; 2],
        }
    }

    /// Prepares the live window `live`, with the solids standing in it
    /// (nearest the eye first) and how many more stand there, for this frame
    /// at simulation time `seconds` in the scene's wind `wind`, over the
    /// square of it the device's bindings hold: its buffers anew where a
    /// square of another size came; where one of its size moved, its table,
    /// sources and ring; in its slots the blocks they do not hold; the
    /// solids pushing its water; the wind dragging its surface; and as many
    /// steps as the time since the last frame takes. Drops it where there is
    /// no window, or no square of it fits.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        live: Option<(&Arc<WaterWindow>, &[WaterObstacle], u32)>,
        (wind, seconds): (Wind, f64),
    ) {
        if let Some(current) = self.window.as_mut() {
            current.settle_taken();
        }
        // The window, with the square of it that runs where it changed.
        let came = live.and_then(|(window, touches, beyond)| {
            let changed = self
                .window
                .as_ref()
                .is_none_or(|current| !Arc::ptr_eq(&current.source, window));
            if !changed {
                return Some((window, None, touches, beyond));
            }
            fit(device, window).map(|square| (window, Some(square), touches, beyond))
        });
        let Some((window, fitted, touches, beyond)) = came else {
            if self.window.take().is_some() {
                self.generation += 1;
            }
            (self.steps, self.last, self.touches) = (0, None, [0; 2]);
            return;
        };
        self.kernels.pipelines(device);
        let layout = &self.kernels.layout;
        if let Some(fitted) = fitted {
            let same_size = self.window.as_ref().is_some_and(|current| {
                current.fitted.cells == fitted.cells && current.fitted.cell == fitted.cell
            });
            if same_size {
                if let Some(current) = self.window.as_mut() {
                    current.follow((device, queue, layout), window, fitted);
                }
            } else {
                self.window = Some(SimWindow::new((device, queue, layout), window, fitted));
                self.generation += 1;
            }
        }
        let Some(current) = self.window.as_mut() else {
            return;
        };
        current.take(device, queue);
        let elapsed = self.last.map_or(0.0, |last| {
            min(max((seconds - last) as f32, 0.0), LONGEST_FRAME)
        });
        self.last = Some(seconds);
        let square = &current.fitted;
        let longest = step_seconds(square.cell);
        let steps = min((elapsed / longest).ceil(), STEPS_AT_MOST);
        self.steps = steps as u32;
        let pushing = touching(square, touches);
        self.touches = [
            pushing.as_ref().map_or(0, |pushing| pushing.count[0]),
            beyond,
        ];
        if self.steps == 0 {
            return;
        }
        let step = min(elapsed / max(steps, 1.0), longest);
        let stored = &current.stored;
        // The wind's mean drags the surface: its gusts come and go within
        // seconds (`Wind::at`), carrying what floats back and forth by a few
        // centimeters in a breeze.
        current.buffers.write_frames(queue, wind.velocity);
        current.buffers.write_squares(
            queue,
            &[SimSlot {
                shape: [square.cells, square.cells, SIM_STEPPING, current.sourced],
                step: [square.cell, step, 0.0, 0.0],
                blocks: [
                    stored.blocks[0],
                    stored.blocks[1],
                    stored.stored() as u32,
                    0,
                ],
            }],
        );
        current.buffers.write_touches(queue, pushing.as_ref());
    }

    /// Copies the blocks this frame took into the window's slots, then runs
    /// the steps prepared, timed by `profiler` (`water_sim`).
    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        let (Some(pipelines), Some(window)) = (self.kernels.created(), &self.window) else {
            return;
        };
        window.copy_taken(encoder);
        if self.steps == 0 {
            return;
        }
        let blocks = window.stored.stored() as u32;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/water-sim"),
            timestamp_writes: profiler.span("water_sim").map(|span| span.compute()),
        });
        encode_steps(
            &mut pass,
            pipelines,
            &window.buffers,
            (self.steps, window.sourced, false, false),
            blocks,
        );
    }

    /// What water draws the simulated water from: the square of the window
    /// that runs, the blocks its slots turn by, its cells, its ground, what
    /// the beds hold under them and the triangles drawing them; and a count
    /// advancing whenever they change.
    pub(crate) fn drawn(&self) -> (Option<SimDrawn<'_>>, u64) {
        let drawn = self.window.as_ref().map(|window| SimDrawn {
            window: window.fitted.as_ref(),
            wrap: window.wrap,
            cells: &window.buffers.cells,
            ground: &window.buffers.ground,
            beds: &window.beds,
            indices: &window.indices,
        });
        (drawn, self.generation)
    }

    /// The solids pushing its water this frame, and those standing in its
    /// window past the most that do, which push none.
    pub(crate) fn touches(&self) -> [u32; 2] {
        self.touches
    }

    /// Bytes of the window's buffers.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.window.as_ref().map_or(0, |window| {
            window.buffers.bytes()
                + window.indices.size()
                + window.beds.size()
                + window.stage.as_ref().map_or(0, wgpu::Buffer::size)
        })
    }
}

impl SimWindow {
    /// The buffers `fitted`, the square of `source` that runs, runs in,
    /// with its table, sources and ring and the triangles drawing its
    /// cells; its slots take its blocks with the frame.
    fn new(
        (device, queue, layout): (&wgpu::Device, &wgpu::Queue, &crate::pipelines::GroupLayout),
        source: &Arc<WaterWindow>,
        fitted: Arc<WaterWindow>,
    ) -> Self {
        let (stored, wrap) = wrapped(&fitted);
        let slots = stored.stored();
        let buffers = SlotBuffers::new(
            device,
            layout,
            (Capacity::of(&stored, fitted.sources.len().max(1)), 1),
            true,
        );
        let n = fitted.cells;
        let quads = n.saturating_sub(1);
        let triangles: Vec<u32> = (0..quads * quads)
            .flat_map(|quad| {
                let (i, j) = (quad % quads, quad / quads);
                let a = j * n + i;
                let (b, c, d) = (a + 1, a + n + 1, a + n);
                [a, b, c, a, c, d]
            })
            .collect();
        let indices = crate::buffers::storage(
            device,
            "engine/water-sim-indices",
            (triangles.len().max(1) * 4) as u64,
            wgpu::BufferUsages::INDEX,
        );
        if !triangles.is_empty() {
            queue.write_buffer(&indices, 0, bytemuck::cast_slice(&triangles));
        }
        let beds = crate::buffers::storage(
            device,
            "engine/water-sim-beds",
            slots as u64 * BLOCK_BED_BYTES,
            wgpu::BufferUsages::empty(),
        );
        let mut window = Self {
            source: source.clone(),
            fitted,
            stored,
            wrap,
            buffers,
            sourced: 0,
            indices,
            beds,
            stamps: vec![0; slots],
            taken: None,
            stage: None,
        };
        window.write_square((device, queue, layout));
        window
    }

    /// Takes `fitted`, the square of `source` that runs, of its square's
    /// size, in its square's place: its table turned to its place, and its
    /// sources and ring. Its slots keep the blocks both squares hold, and
    /// take the rest with the frame.
    fn follow(
        &mut self,
        gpu: (&wgpu::Device, &wgpu::Queue, &crate::pipelines::GroupLayout),
        source: &Arc<WaterWindow>,
        fitted: Arc<WaterWindow>,
    ) {
        (self.stored, self.wrap) = wrapped(&fitted);
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

    /// Takes into its slots the blocks of its square they do not hold: each
    /// one's ground (at its cells' corners, the levels lakes hold its cells
    /// at, UNHELD where none does, its cells' roughness, and the share of
    /// each covered by solids and the height their tops stand at), its
    /// cells' water to start from, into the step's first state, and what the
    /// beds hold under them. Where every slot takes one, they are written in
    /// place; else staged in one write, in their slots' order, for the
    /// frame's encoding to copy into their slots a run of consecutive slots
    /// at a time.
    fn take(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let side = self.fitted.blocks();
        let mut fresh: Vec<(u32, u32, u32)> = Vec::new();
        for z in 0..side {
            for x in 0..side {
                let slot = self.stored.table[(z * side + x) as usize];
                if self.stamps[slot as usize] != self.fitted.block(x, z).stamp() {
                    fresh.push((slot, x, z));
                }
            }
        }
        if fresh.is_empty() {
            return;
        }
        fresh.sort_unstable_by_key(|&(slot, _, _)| slot);
        let blocks = fresh.len();
        let mut ground: Vec<f32> = Vec::with_capacity(blocks * BLOCK_GROUND);
        let mut cells: Vec<[f32; 4]> = Vec::with_capacity(blocks * BLOCK_CELLS);
        let mut beds: Vec<u32> = Vec::with_capacity(blocks * BLOCK_CELLS);
        let mut held: Vec<f32> = Vec::with_capacity(BLOCK_CELLS);
        let mut rough: Vec<f32> = Vec::with_capacity(BLOCK_CELLS);
        let mut shares: Vec<f32> = Vec::with_capacity(BLOCK_CELLS);
        let mut tops: Vec<f32> = Vec::with_capacity(BLOCK_CELLS);
        for &(slot, x, z) in &fresh {
            let block = self.fitted.block(x, z);
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
                    beds.push(block.bed_word(l, k));
                }
            }
            ground.extend_from_slice(&held);
            ground.extend_from_slice(&rough);
            ground.extend_from_slice(&shares);
            ground.extend_from_slice(&tops);
            self.stamps[slot as usize] = block.stamp();
        }
        let buffers = &self.buffers;
        if blocks == self.stamps.len() {
            queue.write_buffer(
                &buffers.ground,
                buffers.ground_at(0) * 4,
                bytemuck::cast_slice(&ground),
            );
            queue.write_buffer(
                &buffers.cells,
                buffers.cells_at(0) * 16,
                bytemuck::cast_slice(&cells),
            );
            queue.write_buffer(&self.beds, 0, bytemuck::cast_slice(&beds));
            return;
        }
        let mut staged: Vec<u8> = Vec::with_capacity(
            blocks * (BLOCK_GROUND_BYTES + BLOCK_CELL_BYTES + BLOCK_BED_BYTES) as usize,
        );
        staged.extend_from_slice(bytemuck::cast_slice(&ground));
        staged.extend_from_slice(bytemuck::cast_slice(&cells));
        staged.extend_from_slice(bytemuck::cast_slice(&beds));
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
        for (at, &(slot, _, _)) in fresh.iter().enumerate() {
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
    /// first state, and what their beds hold.
    fn copy_taken(&self, encoder: &mut wgpu::CommandEncoder) {
        let (Some(taken), Some(stage)) = (&self.taken, &self.stage) else {
            return;
        };
        let staged = u64::from(taken.staged);
        let cells_staged = staged * BLOCK_GROUND_BYTES;
        let beds_staged = cells_staged + staged * BLOCK_CELL_BYTES;
        let buffers = &self.buffers;
        let (ground_at, cells_at) = (buffers.ground_at(0) * 4, buffers.cells_at(0) * 16);
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
                beds_staged + first * BLOCK_BED_BYTES,
                &self.beds,
                slot * BLOCK_BED_BYTES,
                slots * BLOCK_BED_BYTES,
            );
        }
        taken.recorded.set(true);
    }
}

/// The blocks of `window`'s square as its slot stores them, and the blocks
/// they turn by along x and z: every one, each in the slot of its place on
/// the terrain's lattice of blocks taken around the square's blocks along
/// each side, so a block keeps its slot as the window follows the camera.
fn wrapped(window: &WaterWindow) -> (StoredBlocks, [u32; 2]) {
    let side = window.blocks();
    let wrap = window.first.map(|first| {
        first
            .div_euclid(i64::from(BLOCK_SIDE))
            .rem_euclid(i64::from(side)) as u32
    });
    let count = (side * side) as usize;
    let (mut table, mut places) = (vec![0; count], vec![0; count]);
    for z in 0..side {
        for x in 0..side {
            let slot = (z + wrap[1]) % side * side + (x + wrap[0]) % side;
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
    let fits = |blocks: u32| largest_bytes(blocks * BLOCK_SIDE, sources) <= binding;
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

/// The most bytes one buffer of a window `cells` cells along each side
/// takes, its rivers bringing water to `sources` of them: its slot's
/// largest, its inputs' (an entry for each source and two for each of the
/// `4 × cells` cells of its ring) or its triangles'.
fn largest_bytes(cells: u32, sources: usize) -> u64 {
    let capacity = Capacity::of(&StoredBlocks::whole([cells; 2]), sources.max(1));
    let inputs = (u64::from(capacity.sources) + 8 * u64::from(cells)) * 16;
    let quads = u64::from(cells.saturating_sub(1));
    slot_largest_bytes(capacity)
        .max(inputs)
        .max(quads * quads * 6 * 4)
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

/// What water draws simulated water from ([`WaterSim::drawn`]).
pub(crate) struct SimDrawn<'a> {
    pub(crate) window: &'a WaterWindow,
    pub(crate) wrap: [u32; 2],
    pub(crate) cells: &'a wgpu::Buffer,
    pub(crate) ground: &'a wgpu::Buffer,
    pub(crate) beds: &'a wgpu::Buffer,
    pub(crate) indices: &'a wgpu::Buffer,
}
