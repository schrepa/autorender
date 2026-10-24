//! Water's simulation (`water_sim.wgsl`): rivers running over a terrain's
//! [`WaterWindow`] by the shallow-water equations, a few steps a frame, from
//! the water as it settled there, which also stands just past its edges;
//! the water pass draws its water (`meshes/water.wgsl`'s simulated water) in
//! place of the tiles'. As the window follows the camera, its water carries
//! over and only what it newly covers starts as it settled. The window is
//! one slot of the scheme's buffers ([`crate::water_kernels`]) storing every
//! block of its cells in order, stepping every cell by the frame's seconds,
//! with a ring past its edges. Where one of the device's storage bindings
//! cannot hold a buffer of the window, its middle square whose buffers they
//! hold runs instead.

use std::sync::Arc;

use engine_core::WaterWindow;
use engine_core::math::{max, min};
use engine_core::terrain_data::StoredBlocks;
use engine_core::terrain_data::shallow_water::{UNHELD, step_seconds};

use crate::gpu_layouts::constants::SIM_STEPPING;
use crate::gpu_layouts::water::SimSlot;
use crate::water_kernels::{
    Capacity, Kernels, SlotBuffers, encode_steps, grid, slot_largest_bytes,
};

/// The most steps a frame runs, and the most seconds a frame simulates:
/// longer frames simulate less time than passed. Eight steps of cells half
/// a meter across ([`step_seconds`]) cover the longest frame.
const STEPS_AT_MOST: f32 = 8.0;
const LONGEST_FRAME: f32 = 1.0 / 30.0;

/// A terrain's water simulated over its window: the scheme's passes, the
/// window's buffers, the steps this frame runs, the cells the window moved
/// by this frame where its water carries over, when the last frame was,
/// and a count advancing whenever the buffers water draws from change.
pub(crate) struct WaterSim {
    kernels: Kernels,
    window: Option<SimWindow>,
    steps: u32,
    carried: Option<[i32; 2]>,
    last: Option<f64>,
    generation: u64,
}

/// A window as terrains placed it and the square of it that runs (all of
/// it, where the device's bindings hold its buffers), the square's blocks
/// (every one of them, in order), its buffers (one slot with its ring), the
/// count of cells rivers bring water to, the triangles water draws its
/// cells' middles with, and what the beds hold under its cells, in the
/// blocks' order, which water draws them with.
struct SimWindow {
    source: Arc<WaterWindow>,
    fitted: Arc<WaterWindow>,
    stored: StoredBlocks,
    buffers: SlotBuffers,
    sourced: u32,
    indices: wgpu::Buffer,
    beds: wgpu::Buffer,
}

impl WaterSim {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            kernels: Kernels::new(device),
            window: None,
            steps: 0,
            carried: None,
            last: None,
            generation: 0,
        }
    }

    /// Prepares `window`'s water for this frame at simulation time
    /// `seconds`, over the square of it the device's bindings hold: its
    /// buffers anew from the water as it settled there when a square of
    /// another size came, its ground, start, sources and ring where one of
    /// its size moved (its water carrying over where it moved by whole cells
    /// on the same datum), and as many steps as the time since the last
    /// frame takes; drops it where there is no window, or no square of it
    /// fits.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        window: Option<&Arc<WaterWindow>>,
        seconds: f64,
    ) {
        // The window, with the square of it that runs where it changed.
        let came = window.and_then(|window| {
            let changed = self
                .window
                .as_ref()
                .is_none_or(|current| !Arc::ptr_eq(&current.source, window));
            if !changed {
                return Some((window, None));
            }
            fit(device, window).map(|square| (window, Some(square)))
        });
        let Some((window, fitted)) = came else {
            if self.window.take().is_some() {
                self.generation += 1;
            }
            (self.steps, self.last) = (0, None);
            return;
        };
        self.kernels.pipelines(device);
        self.carried = None;
        if let Some(fitted) = fitted {
            let same_size = self.window.as_ref().is_some_and(|current| {
                current.fitted.cells == fitted.cells
                    && current.fitted.cell == fitted.cell
                    && fitted.sources.len() <= current.buffers.capacity.sources as usize
            });
            if same_size {
                self.carried = self.refill(queue, window, fitted);
            } else {
                self.window = Some(self.buffers(device, queue, window, fitted));
                self.generation += 1;
            }
        }
        let Some(current) = self.window.as_ref() else {
            return;
        };
        let elapsed = self.last.map_or(0.0, |last| {
            min(max((seconds - last) as f32, 0.0), LONGEST_FRAME)
        });
        self.last = Some(seconds);
        let square = &current.fitted;
        let longest = step_seconds(square.cell);
        let steps = min((elapsed / longest).ceil(), STEPS_AT_MOST);
        self.steps = steps as u32;
        // The frames hold the cells the window moved by even in a frame
        // running no steps: its water carries over regardless.
        if self.steps == 0 && self.carried.is_none() {
            return;
        }
        let step = min(elapsed / max(steps, 1.0), longest);
        let stored = &current.stored;
        current
            .buffers
            .write_frames(queue, self.carried.unwrap_or_default());
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
    }

    /// The buffers `fitted`, the square of `source` that runs, runs in, its
    /// every block stored, its cells starting as its water settled, with
    /// its ground, sources and ring.
    fn buffers(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &Arc<WaterWindow>,
        fitted: Arc<WaterWindow>,
    ) -> SimWindow {
        let window = fitted.as_ref();
        let n = window.cells as usize;
        let stored = StoredBlocks::whole([window.cells; 2]);
        let sources = window.sources.len().max(1);
        let buffers = SlotBuffers::new(
            device,
            &self.kernels.layout,
            (Capacity::of(&stored, sources), 1),
            true,
        );
        buffers.write_table(queue, 0, &stored);
        let start = start(window, &stored);
        let both = [start.as_slice(), start.as_slice()].concat();
        queue.write_buffer(
            &buffers.ground,
            0,
            bytemuck::cast_slice(&ground(window, &stored)),
        );
        queue.write_buffer(&buffers.cells, 0, bytemuck::cast_slice(&both));
        let sourced = write_inputs(queue, &buffers, window, &stored);
        let quads = n.saturating_sub(1);
        let triangles: Vec<u32> = (0..quads * quads)
            .flat_map(|quad| {
                let (i, j) = ((quad % quads) as u32, (quad / quads) as u32);
                let a = j * window.cells + i;
                let (b, c, d) = (a + 1, a + window.cells + 1, a + window.cells);
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
        let words = bed_words(window, &stored);
        let beds = crate::buffers::storage(
            device,
            "engine/water-sim-beds",
            (words.len() * 8) as u64,
            wgpu::BufferUsages::empty(),
        );
        queue.write_buffer(&beds, 0, bytemuck::cast_slice(&words));
        SimWindow {
            source: source.clone(),
            fitted,
            stored,
            buffers,
            sourced,
            indices,
            beds,
        }
    }

    /// Takes `fitted`, the square of `source` that runs, of the current
    /// square's size, into the current buffers: its ground, sources and
    /// ring, and its start in the second state; where it moved by whole
    /// cells on the same datum, the cells it moved by, which the water
    /// carries over across, else its start in the first state too.
    fn refill(
        &mut self,
        queue: &wgpu::Queue,
        source: &Arc<WaterWindow>,
        fitted: Arc<WaterWindow>,
    ) -> Option<[i32; 2]> {
        let current = self.window.as_mut()?;
        let offset = (glam::Vec2::from(fitted.origin) - glam::Vec2::from(current.fitted.origin))
            / fitted.cell;
        let whole = offset.round();
        let carried = ((offset - whole).abs().max_element() < 1.0e-3
            && fitted.datum == current.fitted.datum)
            .then_some([whole.x as i32, whole.y as i32]);
        let start = start(&fitted, &current.stored);
        let buffers = &current.buffers;
        queue.write_buffer(
            &buffers.ground,
            0,
            bytemuck::cast_slice(&ground(&fitted, &current.stored)),
        );
        queue.write_buffer(
            &current.beds,
            0,
            bytemuck::cast_slice(&bed_words(&fitted, &current.stored)),
        );
        queue.write_buffer(
            &buffers.cells,
            buffers.held() * 16,
            bytemuck::cast_slice(&start),
        );
        if carried.is_none() {
            queue.write_buffer(&buffers.cells, 0, bytemuck::cast_slice(&start));
        }
        current.sourced = write_inputs(queue, buffers, &fitted, &current.stored);
        current.source = source.clone();
        current.fitted = fitted;
        carried
    }

    /// Runs the steps prepared, timed by `profiler` (`water_sim`).
    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        let (Some(pipelines), Some(window)) = (self.kernels.created(), &self.window) else {
            return;
        };
        if self.steps == 0 && self.carried.is_none() {
            return;
        }
        let blocks = window.stored.stored() as u32;
        let [groups_x, groups_z] = grid(blocks);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/water-sim"),
            timestamp_writes: profiler.span("water_sim").map(|span| span.compute()),
        });
        // The window moved: its water carries over before it runs on.
        if self.carried.is_some() {
            pass.set_bind_group(0, &window.buffers.groups[0], &[]);
            pass.set_pipeline(&pipelines.carry);
            pass.dispatch_workgroups(groups_x, groups_z, 1);
            pass.set_pipeline(&pipelines.restart);
            pass.dispatch_workgroups(groups_x, groups_z, 1);
        }
        encode_steps(
            &mut pass,
            pipelines,
            &window.buffers,
            (self.steps, window.sourced, false, false),
            blocks,
        );
    }

    /// What water draws the simulated water from: the square of the window
    /// that runs, its cells, its ground, what the beds hold under them and
    /// the triangles drawing them; and a count advancing whenever they
    /// change.
    pub(crate) fn drawn(&self) -> (Option<SimDrawn<'_>>, u64) {
        let drawn = self.window.as_ref().map(|window| SimDrawn {
            window: window.fitted.as_ref(),
            cells: &window.buffers.cells,
            ground: &window.buffers.ground,
            beds: &window.beds,
            indices: &window.indices,
        });
        (drawn, self.generation)
    }

    /// Bytes of the window's buffers.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.window.as_ref().map_or(0, |window| {
            window.buffers.bytes() + window.indices.size() + window.beds.size()
        })
    }
}

/// The square of `window` whose every buffer one of `device`'s storage
/// bindings holds: the window itself where they hold its own, else its
/// middle square of the most cells along each side whose buffers they
/// hold; none where they hold not even two.
fn fit(device: &wgpu::Device, window: &Arc<WaterWindow>) -> Option<Arc<WaterWindow>> {
    let limits = device.limits();
    let binding = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    let sources = window.sources.len();
    let fits = |cells: u32| largest_bytes(cells, sources) <= binding;
    if fits(window.cells) {
        return Some(window.clone());
    }
    // The most cells whose buffers fit lie from `low` (or fewer) up to
    // before `high`.
    let (mut low, mut high) = (1, window.cells);
    while low + 1 < high {
        let middle = low + (high - low) / 2;
        if fits(middle) {
            low = middle;
        } else {
            high = middle;
        }
    }
    (low >= 2 && fits(low)).then(|| Arc::new(window.within(low)))
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

/// `window`'s water to start from, as `stored`, its blocks, holds its
/// cells ([`WaterWindow::start`]): each one's surface, flow along x and z
/// and foam.
fn start(window: &WaterWindow, stored: &StoredBlocks) -> Vec<[f32; 4]> {
    stored.gather(|i, j| window.start(i, j))
}

/// `window`'s ground as `stored`, its blocks, holds it: the ground at its
/// cells' corners, the level a lake holds each cell at (UNHELD where none
/// does), and the roughness of each cell's ground.
fn ground(window: &WaterWindow, stored: &StoredBlocks) -> Vec<f32> {
    let n = window.cells;
    let held = stored.gather(|i, j| {
        let level = window.held(i, j);
        if level.is_nan() { UNHELD } else { level }
    });
    let roughness = stored.gather(|i, j| window.roughness[(j * n + i) as usize]);
    stored.ground(
        |i, j| window.corners[(j * (n + 1) + i) as usize],
        &held,
        &roughness,
    )
}

/// What the beds hold under `window`'s cells as `stored`, its blocks, holds
/// them ([`WaterWindow::bed_words`]): each one's reach's code and nearest
/// obstacle.
fn bed_words(window: &WaterWindow, stored: &StoredBlocks) -> Vec<[u32; 2]> {
    let n = window.cells;
    stored.gather(|i, j| window.bed_words[(j * n + i) as usize])
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

/// What water draws simulated water from ([`WaterSim::drawn`]).
pub(crate) struct SimDrawn<'a> {
    pub(crate) window: &'a WaterWindow,
    pub(crate) cells: &'a wgpu::Buffer,
    pub(crate) ground: &'a wgpu::Buffer,
    pub(crate) beds: &'a wgpu::Buffer,
    pub(crate) indices: &'a wgpu::Buffer,
}
