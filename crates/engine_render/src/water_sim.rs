//! Water's simulation (`water_sim.wgsl`): rivers running over a terrain's
//! [`WaterWindow`] by the shallow-water equations, a few steps a frame, from
//! the water as it settled there, which also stands just past its edges;
//! the water pass draws its water (`meshes/water.wgsl`'s simulated water) in
//! place of the tiles'. As the window follows the camera, its water carries
//! over and only what it newly covers starts as it settled. The window is
//! one slot of the scheme's buffers ([`crate::water_kernels`]), stepping
//! every cell by the frame's seconds, with a ring past its edges.

use std::sync::Arc;

use engine_core::WaterWindow;
use engine_core::math::{max, min};
use engine_core::terrain_data::shallow_water::{ROUGHNESS, UNHELD, step_seconds};

use crate::gpu_layouts::constants::{SIM_STEPPING, WATER_SIM_GROUP};
use crate::gpu_layouts::water::SimSlot;
use crate::water_kernels::{Kernels, SlotBuffers, encode_steps};

/// The most steps a frame runs, and the most seconds a frame simulates:
/// longer frames simulate less time than passed.
const STEPS_AT_MOST: f32 = 4.0;
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

/// A window's buffers (one slot with its ring), the count of cells rivers
/// bring water to, and the triangles water draws its cells' middles with.
struct SimWindow {
    source: Arc<WaterWindow>,
    buffers: SlotBuffers,
    sourced: u32,
    indices: wgpu::Buffer,
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
    /// `seconds`: its buffers anew from the water as it settled there when
    /// a window of another size came, its ground, start, sources and ring
    /// where one of its size moved (its water carrying over where it moved
    /// by whole cells), and as many steps as the time since the last frame
    /// takes; drops it where there is no window.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        window: Option<&Arc<WaterWindow>>,
        seconds: f64,
    ) {
        let Some(window) = window else {
            if self.window.take().is_some() {
                self.generation += 1;
            }
            (self.steps, self.last) = (0, None);
            return;
        };
        self.kernels.pipelines(device);
        self.carried = None;
        let changed = self
            .window
            .as_ref()
            .is_none_or(|current| !Arc::ptr_eq(&current.source, window));
        let same_size = self.window.as_ref().is_some_and(|current| {
            current.source.cells == window.cells
                && current.source.cell == window.cell
                && window.sources.len() <= current.buffers.sources as usize
        });
        if changed && same_size {
            self.carried = self.refill(queue, window);
        } else if changed {
            self.window = Some(self.buffers(device, queue, window));
            self.generation += 1;
        }
        let elapsed = self.last.map_or(0.0, |last| {
            min(max((seconds - last) as f32, 0.0), LONGEST_FRAME)
        });
        self.last = Some(seconds);
        let longest = step_seconds(window.cell);
        let steps = min((elapsed / longest).ceil(), STEPS_AT_MOST);
        self.steps = steps as u32;
        // The frames hold the cells the window moved by even in a frame
        // running no steps: its water carries over regardless.
        if self.steps == 0 && self.carried.is_none() {
            return;
        }
        let step = min(elapsed / max(steps, 1.0), longest);
        let Some(current) = self.window.as_ref() else {
            return;
        };
        current
            .buffers
            .write_frames(queue, self.carried.unwrap_or_default());
        current.buffers.write_squares(
            queue,
            &[SimSlot {
                shape: [window.cells, window.cells, SIM_STEPPING, current.sourced],
                step: [window.cell, step, ROUGHNESS, 0.0],
            }],
        );
    }

    /// The buffers `window`'s water runs in, its cells and foam starting as
    /// its water settled, with its sources and ring.
    fn buffers(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        window: &Arc<WaterWindow>,
    ) -> SimWindow {
        let n = window.cells as usize;
        let sources = (window.sources.len() as u32).max(1);
        let buffers = SlotBuffers::new(
            device,
            &self.kernels.layout,
            ([window.cells, window.cells], 1),
            (sources, true),
        );
        let start = start(window);
        let both = [start.as_slice(), start.as_slice()].concat();
        queue.write_buffer(&buffers.ground, 0, bytemuck::cast_slice(&window.corners));
        queue.write_buffer(&buffers.cells, 0, bytemuck::cast_slice(&both));
        let foams = [&window.foam[..], &window.foam[..]].concat();
        queue.write_buffer(&buffers.foam, 0, bytemuck::cast_slice(&foams));
        let sourced = write_inputs(queue, &buffers, window);
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
        SimWindow {
            source: window.clone(),
            buffers,
            sourced,
            indices,
        }
    }

    /// Takes `window`, of the current one's size, into the current buffers:
    /// its ground, sources and ring, and its start in the second state and
    /// foam; where it moved by whole cells, the cells it moved by, which the
    /// water carries over across, else its start in the first state too.
    fn refill(&mut self, queue: &wgpu::Queue, window: &Arc<WaterWindow>) -> Option<[i32; 2]> {
        let current = self.window.as_mut()?;
        let offset = (glam::Vec2::from(window.origin) - glam::Vec2::from(current.source.origin))
            / window.cell;
        let whole = offset.round();
        let carried = ((offset - whole).abs().max_element() < 1.0e-3)
            .then_some([whole.x as i32, whole.y as i32]);
        let n = window.cells as usize;
        let start = start(window);
        let buffers = &current.buffers;
        queue.write_buffer(&buffers.ground, 0, bytemuck::cast_slice(&window.corners));
        queue.write_buffer(
            &buffers.cells,
            (n * n * 16) as u64,
            bytemuck::cast_slice(&start),
        );
        queue.write_buffer(
            &buffers.foam,
            (n * n * 4) as u64,
            bytemuck::cast_slice(&window.foam),
        );
        if carried.is_none() {
            queue.write_buffer(&buffers.cells, 0, bytemuck::cast_slice(&start));
            queue.write_buffer(&buffers.foam, 0, bytemuck::cast_slice(&window.foam));
        }
        current.sourced = write_inputs(queue, buffers, window);
        current.source = window.clone();
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
        let cells = window.source.cells.div_ceil(WATER_SIM_GROUP);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/water-sim"),
            timestamp_writes: profiler.span("water_sim").map(|span| span.compute()),
        });
        // The window moved: its water carries over before it runs on.
        if self.carried.is_some() {
            pass.set_bind_group(0, &window.buffers.groups[0], &[]);
            pass.set_pipeline(&pipelines.carry);
            pass.dispatch_workgroups(cells, cells, 1);
            pass.set_pipeline(&pipelines.restart);
            pass.dispatch_workgroups(cells, cells, 1);
        }
        encode_steps(
            &mut pass,
            pipelines,
            &window.buffers,
            (self.steps, window.sourced, false, false),
            [window.source.cells; 2],
        );
    }

    /// What water draws the simulated water from: the window, its cells,
    /// its ground, its foam and the triangles drawing them; and a count
    /// advancing whenever they change.
    pub(crate) fn drawn(&self) -> (Option<SimDrawn<'_>>, u64) {
        let drawn = self.window.as_ref().map(|window| SimDrawn {
            window: window.source.as_ref(),
            cells: &window.buffers.cells,
            ground: &window.buffers.ground,
            foam: &window.buffers.foam,
            indices: &window.indices,
        });
        (drawn, self.generation)
    }

    /// Bytes of the window's buffers.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.window
            .as_ref()
            .map_or(0, |window| window.buffers.bytes() + window.indices.size())
    }
}

/// Writes what `window`'s rivers bring and its ring into `buffers`' inputs;
/// the cells its rivers bring water to.
fn write_inputs(queue: &wgpu::Queue, buffers: &SlotBuffers, window: &WaterWindow) -> u32 {
    let sources = sourcing(window);
    let held = buffers.held() as usize;
    let sources = &sources[..sources.len().min(held)];
    if !sources.is_empty() {
        queue.write_buffer(&buffers.inputs, 0, bytemuck::cast_slice(sources));
    }
    queue.write_buffer(
        &buffers.inputs,
        buffers.ring_at(0) * 16,
        bytemuck::cast_slice(&ringing(window)),
    );
    sources.len() as u32
}

/// `window`'s water to start from, each cell's surface, flow along x and z
/// and the level a lake holds it at (UNHELD where none does).
fn start(window: &WaterWindow) -> Vec<[f32; 4]> {
    window
        .seed
        .iter()
        .zip(window.held.iter())
        .map(|(&[surface, x, z], &held)| {
            let held = if held.is_nan() { UNHELD } else { held };
            [surface, x, z, held]
        })
        .collect()
}

/// The cells `window`'s rivers bring water to: each one's index (as bits),
/// the cubic meters a second they bring and the way it runs as it comes.
fn sourcing(window: &WaterWindow) -> Vec<[f32; 4]> {
    window
        .sources
        .iter()
        .map(|source| {
            [
                f32::from_bits(source.cell),
                source.rate,
                source.way[0],
                source.way[1],
            ]
        })
        .collect()
}

/// `window`'s ring as the shader reads it: each cell past its edges'
/// surface, flow along x and z and the level a lake holds it at
/// (`UNHELD` where none does), then its foam.
fn ringing(window: &WaterWindow) -> Vec<[f32; 4]> {
    window
        .ring
        .iter()
        .flat_map(|ghost| {
            let held = if ghost.held.is_nan() {
                UNHELD
            } else {
                ghost.held
            };
            [
                [ghost.surface, ghost.flow[0], ghost.flow[1], held],
                [ghost.foam, 0.0, 0.0, 0.0],
            ]
        })
        .collect()
}

/// What water draws simulated water from ([`WaterSim::drawn`]).
pub(crate) struct SimDrawn<'a> {
    pub(crate) window: &'a WaterWindow,
    pub(crate) cells: &'a wgpu::Buffer,
    pub(crate) ground: &'a wgpu::Buffer,
    pub(crate) foam: &'a wgpu::Buffer,
    pub(crate) indices: &'a wgpu::Buffer,
}
