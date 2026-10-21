//! Water's simulation (`water_sim.wgsl`): rivers running over a terrain's
//! [`WaterWindow`] by the shallow-water equations, a few steps a frame, from
//! the water as it settled there, which also stands just past its edges;
//! the water pass draws its water (`meshes/water.wgsl`'s simulated water) in
//! place of the tiles'. As the window follows the camera, its water carries
//! over and only what it newly covers starts as it settled.

use std::sync::Arc;

use engine_core::WaterWindow;
use engine_core::math::{max, min};
use engine_core::terrain_data::shallow_water::{ROUGHNESS, UNHELD, step_seconds};

use crate::bindings::WATER_SIM;
use crate::gpu_layouts::constants::{WATER_SIM_GROUP, WATER_SIM_SOURCE_GROUP};
use crate::gpu_layouts::water::SimFrame;
use crate::shaders::entry::water_sim;

/// The most steps a frame runs, and the most seconds a frame simulates:
/// longer frames simulate less time than passed.
const STEPS_AT_MOST: f32 = 4.0;
const LONGEST_FRAME: f32 = 1.0 / 30.0;
/// Cells rivers bring water to the sources' buffer holds at least.
const SOURCES_HELD: usize = 1024;

/// A terrain's water simulated over its window: the shader's pipelines and
/// each stage's frame, the window's buffers, the steps this frame runs, the
/// cells the window moved by this frame where its water carries over, when
/// the last frame was, and a count advancing whenever the buffers water
/// draws from change.
pub(crate) struct WaterSim {
    layout: crate::pipelines::GroupLayout,
    pipelines: Option<[wgpu::ComputePipeline; 8]>,
    frames: [wgpu::Buffer; 2],
    window: Option<SimBuffers>,
    steps: u32,
    carried: Option<[i32; 2]>,
    last: Option<f64>,
    generation: u64,
}

/// A window's buffers, the count of cells rivers bring water to and the
/// groups binding them for each stage, and the triangles water draws its
/// cells' middles with.
struct SimBuffers {
    source: Arc<WaterWindow>,
    ground: wgpu::Buffer,
    cells: wgpu::Buffer,
    faces: wgpu::Buffer,
    fluxes: wgpu::Buffer,
    sources: wgpu::Buffer,
    sourced: u32,
    ring: wgpu::Buffer,
    groups: [wgpu::BindGroup; 2],
    indices: wgpu::Buffer,
    foam: wgpu::Buffer,
}

impl WaterSim {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let bytes = size_of::<SimFrame>() as u64;
        Self {
            layout: crate::pipelines::layout(device, &WATER_SIM),
            pipelines: None,
            frames: [
                crate::buffers::uniform(device, "engine/water-sim-first", bytes),
                crate::buffers::uniform(device, "engine/water-sim-second", bytes),
            ],
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
        if self.pipelines.is_none() {
            self.pipelines = Some(pipelines(device, &self.layout));
        }
        self.carried = None;
        let changed = self
            .window
            .as_ref()
            .is_none_or(|current| !Arc::ptr_eq(&current.source, window));
        let same_size = self.window.as_ref().is_some_and(|current| {
            current.source.cells == window.cells && current.source.cell == window.cell
        });
        if changed && same_size {
            self.carried = self.refill(device, queue, window);
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
        let sourced = self.window.as_ref().map_or(0, |buffers| buffers.sourced);
        for (stage, buffer) in self.frames.iter().enumerate() {
            let [x, z] = self.carried.unwrap_or_default();
            let frame = SimFrame {
                shape: [window.cells, stage as u32, sourced, 0],
                step: [window.cell, step, ROUGHNESS, 0.0],
                shift: [x, z, 0, 0],
            };
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&frame));
        }
    }

    /// The buffers `window`'s water runs in, its cells and foam starting as
    /// its water settled, with its sources and ring, and the groups binding
    /// them for each stage.
    fn buffers(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        window: &Arc<WaterWindow>,
    ) -> SimBuffers {
        let n = window.cells as usize;
        let start = start(window);
        let both = [start.as_slice(), start.as_slice()].concat();
        let sources = sourcing(window);
        let storage = |label: &str, bytes: usize| {
            crate::buffers::storage(device, label, bytes as u64, wgpu::BufferUsages::empty())
        };
        let ground = storage("engine/water-sim-ground", window.corners.len() * 4);
        queue.write_buffer(&ground, 0, bytemuck::cast_slice(&window.corners));
        let cells = storage("engine/water-sim-cells", both.len() * 16);
        queue.write_buffer(&cells, 0, bytemuck::cast_slice(&both));
        let faces = storage("engine/water-sim-faces", n * n * 4 * 16);
        let fluxes = storage("engine/water-sim-fluxes", 2 * (n + 1) * n * 16);
        let source_buffer = storage(
            "engine/water-sim-sources",
            sources.len().max(SOURCES_HELD) * 16,
        );
        if !sources.is_empty() {
            queue.write_buffer(&source_buffer, 0, bytemuck::cast_slice(&sources));
        }
        let foam = storage("engine/water-sim-foam", 2 * n * n * 4);
        let foams = [&window.foam[..], &window.foam[..]].concat();
        queue.write_buffer(&foam, 0, bytemuck::cast_slice(&foams));
        let ring = storage("engine/water-sim-ring", 8 * n * 16);
        queue.write_buffer(&ring, 0, bytemuck::cast_slice(&ringing(window)));
        let groups = self.groups(
            device,
            [
                &ground,
                &cells,
                &faces,
                &fluxes,
                &source_buffer,
                &foam,
                &ring,
            ],
        );
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
        SimBuffers {
            source: window.clone(),
            ground,
            cells,
            faces,
            fluxes,
            sources: source_buffer,
            sourced: sources.len() as u32,
            ring,
            groups,
            indices,
            foam,
        }
    }

    /// The groups binding the simulation's buffers (its ground, cells,
    /// faces, fluxes, sources, foam and ring) for each stage.
    fn groups(&self, device: &wgpu::Device, buffers: [&wgpu::Buffer; 7]) -> [wgpu::BindGroup; 2] {
        let [ground, cells, faces, fluxes, sources, foam, ring] = buffers;
        [0, 1].map(|stage| {
            crate::pipelines::group(
                device,
                "engine/water-sim",
                &WATER_SIM,
                &self.layout,
                &[
                    crate::buffers::entry("frame", &self.frames[stage]),
                    crate::buffers::entry("sim_ground", ground),
                    crate::buffers::entry("sim_cells", cells),
                    crate::buffers::entry("sim_faces", faces),
                    crate::buffers::entry("sim_fluxes", fluxes),
                    crate::buffers::entry("sim_sources", sources),
                    crate::buffers::entry("sim_foam", foam),
                    crate::buffers::entry("sim_ring", ring),
                ],
            )
        })
    }

    /// Takes `window`, of the current one's size, into the current buffers:
    /// its ground, sources and ring, and its start in the second state and
    /// foam; where it moved by whole cells, the cells it moved by, which the
    /// water carries over across, else its start in the first state too.
    fn refill(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        window: &Arc<WaterWindow>,
    ) -> Option<[i32; 2]> {
        let mut buffers = self.window.take()?;
        let offset = (glam::Vec2::from(window.origin) - glam::Vec2::from(buffers.source.origin))
            / window.cell;
        let whole = offset.round();
        let carried = ((offset - whole).abs().max_element() < 1.0e-3)
            .then_some([whole.x as i32, whole.y as i32]);
        let n = window.cells as usize;
        let start = start(window);
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
        queue.write_buffer(&buffers.ring, 0, bytemuck::cast_slice(&ringing(window)));
        let sources = sourcing(window);
        // More cells rivers bring water to than the buffer holds: a larger
        // one, bound anew.
        if sources.len() * 16 > buffers.sources.size() as usize {
            buffers.sources = crate::buffers::storage(
                device,
                "engine/water-sim-sources",
                (sources.len() * 16) as u64,
                wgpu::BufferUsages::empty(),
            );
            buffers.groups = self.groups(
                device,
                [
                    &buffers.ground,
                    &buffers.cells,
                    &buffers.faces,
                    &buffers.fluxes,
                    &buffers.sources,
                    &buffers.foam,
                    &buffers.ring,
                ],
            );
        }
        if !sources.is_empty() {
            queue.write_buffer(&buffers.sources, 0, bytemuck::cast_slice(&sources));
        }
        buffers.sourced = sources.len() as u32;
        buffers.source = window.clone();
        self.window = Some(buffers);
        carried
    }

    /// Runs the steps prepared, timed by `profiler` (`water_sim`).
    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        let (
            Some(
                [
                    faces,
                    fluxes,
                    update,
                    source,
                    carry,
                    restart,
                    churn,
                    churned,
                ],
            ),
            Some(window),
        ) = (&self.pipelines, &self.window)
        else {
            return;
        };
        if self.steps == 0 && self.carried.is_none() {
            return;
        }
        let n = window.source.cells;
        let (cells, between) = (
            n.div_ceil(WATER_SIM_GROUP),
            (n + 1).div_ceil(WATER_SIM_GROUP),
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/water-sim"),
            timestamp_writes: profiler.span("water_sim").map(|span| span.compute()),
        });
        // The window moved: its water carries over before it runs on.
        if self.carried.is_some() {
            pass.set_bind_group(0, &window.groups[0], &[]);
            pass.set_pipeline(carry);
            pass.dispatch_workgroups(cells, cells, 1);
            pass.set_pipeline(restart);
            pass.dispatch_workgroups(cells, cells, 1);
        }
        for _ in 0..self.steps {
            for group in &window.groups {
                pass.set_bind_group(0, group, &[]);
                pass.set_pipeline(faces);
                pass.dispatch_workgroups(cells, cells, 1);
                pass.set_pipeline(fluxes);
                pass.dispatch_workgroups(between, between, 1);
                pass.set_pipeline(update);
                pass.dispatch_workgroups(cells, cells, 1);
            }
            if window.sourced > 0 {
                pass.set_pipeline(source);
                pass.dispatch_workgroups(window.sourced.div_ceil(WATER_SIM_SOURCE_GROUP), 1, 1);
            }
            pass.set_pipeline(churn);
            pass.dispatch_workgroups(cells, cells, 1);
            pass.set_pipeline(churned);
            pass.dispatch_workgroups(cells, cells, 1);
        }
    }

    /// What water draws the simulated water from: the window, its cells,
    /// its ground, its foam and the triangles drawing them; and a count
    /// advancing whenever they change.
    pub(crate) fn drawn(&self) -> (Option<SimDrawn<'_>>, u64) {
        let drawn = self.window.as_ref().map(|window| SimDrawn {
            window: window.source.as_ref(),
            cells: &window.cells,
            ground: &window.ground,
            foam: &window.foam,
            indices: &window.indices,
        });
        (drawn, self.generation)
    }

    /// Bytes of the window's buffers.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.window.as_ref().map_or(0, |window| {
            [
                &window.ground,
                &window.cells,
                &window.faces,
                &window.fluxes,
                &window.sources,
                &window.ring,
                &window.indices,
                &window.foam,
            ]
            .iter()
            .map(|buffer| buffer.size())
            .sum()
        })
    }
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

fn pipelines(
    device: &wgpu::Device,
    layout: &crate::pipelines::GroupLayout,
) -> [wgpu::ComputePipeline; 8] {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/water_sim.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::WATER_SIM_SHADER.into()),
    });
    let pipeline_layout = crate::pipelines::pipeline_layout(
        device,
        &crate::pipeline_layouts::WATER_SIM_PIPELINES,
        &[layout],
    );
    [
        water_sim::sim_reconstruct,
        water_sim::sim_exchange,
        water_sim::sim_update,
        water_sim::sim_source,
        water_sim::sim_carry,
        water_sim::sim_restart,
        water_sim::sim_churn,
        water_sim::sim_churned,
    ]
    .map(|entry| crate::pipelines::compute_entry(device, &pipeline_layout, &module, entry))
}
