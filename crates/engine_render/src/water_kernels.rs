use bytemuck::Zeroable;
use engine_core::terrain_data::shallow_water::CELL_MEANS;
use engine_core::terrain_data::{BLOCK_CELLS, BLOCK_GROUND, StoredBlocks};

use crate::bindings::WATER_SIM;
use crate::gpu_layouts::constants::{
    SIM_BLOCK_LOOKS, SIM_CELL_SUMS, SIM_LIVE_LOOKS, SIM_SLOTS, WATER_SIM_SOURCE_GROUP,
};
use crate::gpu_layouts::water::{SimFrame, SimSlot, SimSlots, SimTouches};
use crate::shaders::entry::water_sim;

const DISPATCH_ROW: u32 = 65_535;

pub(crate) struct Kernels {
    pub(crate) layout: crate::pipelines::GroupLayout,
    pipelines: Option<Pipelines>,
}

pub(crate) struct Pipelines {
    exchange: wgpu::ComputePipeline,
    waves: wgpu::ComputePipeline,
    seconds: wgpu::ComputePipeline,
    drain: wgpu::ComputePipeline,
    update: wgpu::ComputePipeline,
    source: wgpu::ComputePipeline,
    touch: wgpu::ComputePipeline,
    burst: wgpu::ComputePipeline,
    push: wgpu::ComputePipeline,
    ended: wgpu::ComputePipeline,
    tally: wgpu::ComputePipeline,
    activity: wgpu::ComputePipeline,
    look: wgpu::ComputePipeline,
    drawn: wgpu::ComputePipeline,
}

impl Kernels {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            layout: crate::pipelines::layout(device, &WATER_SIM),
            pipelines: None,
        }
    }

    pub(crate) fn pipelines(&mut self, device: &wgpu::Device) -> &Pipelines {
        let layout = &self.layout;
        self.pipelines.get_or_insert_with(|| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("engine/water_sim.wgsl"),
                source: wgpu::ShaderSource::Wgsl(crate::gpu_data::WATER_SIM_SHADER.into()),
            });
            let pipeline_layout = crate::pipelines::pipeline_layout(
                device,
                &crate::pipeline_layouts::WATER_SIM_PIPELINES,
                &[layout],
            );
            let entry =
                |entry| crate::pipelines::compute_entry(device, &pipeline_layout, &module, entry);
            Pipelines {
                exchange: entry(water_sim::sim_exchange),
                waves: entry(water_sim::sim_waves),
                seconds: entry(water_sim::sim_seconds),
                drain: entry(water_sim::sim_drain),
                update: entry(water_sim::sim_update),
                source: entry(water_sim::sim_source),
                touch: entry(water_sim::sim_touch),
                burst: entry(water_sim::sim_burst),
                push: entry(water_sim::sim_push),
                ended: entry(water_sim::sim_ended),
                tally: entry(water_sim::sim_tally),
                activity: entry(water_sim::sim_activity),
                look: entry(water_sim::sim_look),
                drawn: entry(water_sim::sim_drawn),
            }
        })
    }

    pub(crate) fn created(&self) -> Option<&Pipelines> {
        self.pipelines.as_ref()
    }
}

pub(crate) fn grid(groups: u32) -> [u32; 2] {
    [groups.min(DISPATCH_ROW), groups.div_ceil(DISPATCH_ROW)]
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Capacity {
    pub(crate) blocks: u32,
    pub(crate) mapped: u32,
    pub(crate) edges: u32,
    pub(crate) sources: u32,
}

impl Capacity {
    pub(crate) fn of(stored: &StoredBlocks, sources: usize) -> Self {
        Self {
            blocks: stored.stored() as u32,
            mapped: stored.table.len() as u32,
            edges: stored.cells[0] + stored.cells[1],
            sources: sources as u32,
        }
    }

    pub(crate) fn max(self, other: Self) -> Self {
        Self {
            blocks: self.blocks.max(other.blocks),
            mapped: self.mapped.max(other.mapped),
            edges: self.edges.max(other.edges),
            sources: self.sources.max(other.sources),
        }
    }

    pub(crate) fn holds(self, other: Self) -> bool {
        self.max(other) == self
    }
}

struct Entries {
    cells: u64,
    ground: u64,
    faces: u64,
    table: u64,
    steps: u64,
    averages: u64,
    ring: u64,
}

fn entries(capacity: Capacity, live: bool) -> Entries {
    let blocks = u64::from(capacity.blocks);
    let cells = blocks * BLOCK_CELLS as u64;
    Entries {
        cells,
        ground: blocks * BLOCK_GROUND as u64,
        faces: 2 * cells + u64::from(capacity.edges),
        table: u64::from(capacity.mapped) + blocks,
        steps: cells + 2 * blocks + 1,
        averages: if live {
            SIM_LIVE_LOOKS * cells
        } else {
            (SIM_CELL_SUMS + CELL_MEANS as u64) * cells + SIM_BLOCK_LOOKS * blocks
        },
        ring: 4 * u64::from(capacity.edges),
    }
}

fn bytes(capacity: Capacity, live: bool) -> u64 {
    let entries = entries(capacity, live);
    entries.ground * 4
        + 2 * entries.cells * 16
        + entries.cells * 8
        + entries.faces * 16
        + u64::from(capacity.sources) * 16
        + entries.table * 4
        + entries.steps * 8
        + entries.averages * 16
}

fn largest(capacity: Capacity, live: bool) -> u64 {
    let entries = entries(capacity, live);
    [
        2 * entries.cells * 16,
        entries.faces * 16,
        entries.averages * 16,
        entries.table * 4,
        entries.ground * 4,
        u64::from(capacity.sources) * 16,
    ]
    .into_iter()
    .max()
    .unwrap_or(0)
}

pub(crate) fn slot_bytes(capacity: Capacity) -> u64 {
    bytes(capacity, false)
}

pub(crate) fn slot_largest_bytes(capacity: Capacity) -> u64 {
    largest(capacity, false)
}

pub(crate) fn live_largest_bytes(capacity: Capacity) -> u64 {
    largest(capacity, true)
}

pub(crate) struct SlotBuffers {
    pub(crate) capacity: Capacity,
    pub(crate) slots: u32,
    live: bool,
    pub(crate) ground: wgpu::Buffer,
    pub(crate) cells: wgpu::Buffer,
    pub(crate) depths: wgpu::Buffer,
    pub(crate) fluxes: wgpu::Buffer,
    pub(crate) inputs: wgpu::Buffer,
    pub(crate) blocks: wgpu::Buffer,
    pub(crate) steps: wgpu::Buffer,
    pub(crate) averages: wgpu::Buffer,
    pub(crate) squares: wgpu::Buffer,
    frames: [wgpu::Buffer; 2],
    touches: wgpu::Buffer,
    groups: Vec<wgpu::BindGroup>,
    touched: Option<[u32; 2]>,
}

impl SlotBuffers {
    pub(crate) fn new(
        device: &wgpu::Device,
        layout: &crate::pipelines::GroupLayout,
        (capacity, slots): (Capacity, u32),
        live: bool,
    ) -> Self {
        let entries = entries(capacity, live);
        let all = u64::from(slots);
        let storage = |label: &str, bytes: u64, extra: wgpu::BufferUsages| {
            crate::buffers::storage(device, label, bytes, extra)
        };
        let none = wgpu::BufferUsages::empty();
        let ground = storage(
            "engine/water-sim-ground",
            all * entries.ground * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let cells = storage("engine/water-sim-cells", all * 2 * entries.cells * 16, none);
        let depths = storage("engine/water-sim-depths", all * entries.cells * 8, none);
        let fluxes = storage("engine/water-sim-fluxes", all * entries.faces * 16, none);
        let inputs = inputs(device, (capacity, slots), live);
        let blocks = storage("engine/water-sim-blocks", all * entries.table * 4, none);
        let steps = storage("engine/water-sim-steps", all * entries.steps * 8, none);
        let averages = storage(
            "engine/water-sim-averages",
            all * entries.averages * 16,
            wgpu::BufferUsages::COPY_SRC,
        );
        let uniform =
            |label: &str, bytes: usize| crate::buffers::uniform(device, label, bytes as u64);
        let squares = uniform("engine/water-sim-slots", size_of::<SimSlots>());
        let frames = [
            uniform("engine/water-sim-first", size_of::<SimFrame>()),
            uniform("engine/water-sim-second", size_of::<SimFrame>()),
        ];
        let touches = uniform("engine/water-sim-touches", size_of::<SimTouches>());
        let mut buffers = Self {
            capacity,
            slots,
            live,
            ground,
            cells,
            depths,
            fluxes,
            inputs,
            blocks,
            steps,
            averages,
            squares,
            frames,
            touches,
            groups: Vec::new(),
            touched: None,
        };
        buffers.bind(device, layout);
        buffers
    }

    fn bind(&mut self, device: &wgpu::Device, layout: &crate::pipelines::GroupLayout) {
        self.groups = self
            .frames
            .iter()
            .map(|frame| {
                crate::pipelines::group(
                    device,
                    "engine/water-sim",
                    &WATER_SIM,
                    layout,
                    &[
                        crate::buffers::entry("frame", frame),
                        crate::buffers::entry("sim_ground", &self.ground),
                        crate::buffers::entry("sim_cells", &self.cells),
                        crate::buffers::entry("sim_depths", &self.depths),
                        crate::buffers::entry("sim_fluxes", &self.fluxes),
                        crate::buffers::entry("sim_inputs", &self.inputs),
                        crate::buffers::entry("sim_blocks", &self.blocks),
                        crate::buffers::entry("sim_steps", &self.steps),
                        crate::buffers::entry("sim_slots", &self.squares),
                        crate::buffers::entry("sim_averages", &self.averages),
                        crate::buffers::entry("sim_touches", &self.touches),
                    ],
                )
            })
            .collect();
    }

    pub(crate) fn hold_sources(
        &mut self,
        device: &wgpu::Device,
        layout: &crate::pipelines::GroupLayout,
        sources: usize,
    ) -> bool {
        if sources as u32 <= self.capacity.sources {
            return false;
        }
        self.capacity.sources = (sources as u32).next_power_of_two();
        self.inputs = inputs(device, (self.capacity, self.slots), self.live);
        self.bind(device, layout);
        true
    }

    pub(crate) fn write_frames(&self, queue: &wgpu::Queue) {
        let capacity = self.capacity;
        for (stage, buffer) in self.frames.iter().enumerate() {
            let frame = SimFrame {
                stage: [stage as u32, self.slots, capacity.blocks, capacity.mapped],
                inputs: [capacity.sources, capacity.edges, 0, 0],
            };
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&frame));
        }
        let counted = self.held() + 2 * u64::from(capacity.blocks);
        for slot in 0..self.slots {
            let at = (self.steps_at(slot) + counted) * 8;
            queue.write_buffer(&self.steps, at, bytemuck::cast_slice(&[0.0_f32; 2]));
        }
    }

    pub(crate) fn write_squares(&self, queue: &wgpu::Queue, squares: &[SimSlot]) {
        let mut all = SimSlots {
            slot: [SimSlot::zeroed(); SIM_SLOTS],
        };
        for (slot, square) in all.slot.iter_mut().zip(squares) {
            *slot = *square;
        }
        queue.write_buffer(&self.squares, 0, bytemuck::bytes_of(&all));
    }

    pub(crate) fn write_touches(&mut self, queue: &wgpu::Queue, touches: Option<&SimTouches>) {
        self.touched = touches.map(|touches| [touches.blocks[2], touches.blocks[3]]);
        let none = SimTouches::zeroed();
        queue.write_buffer(
            &self.touches,
            0,
            bytemuck::bytes_of(touches.unwrap_or(&none)),
        );
    }

    pub(crate) fn write_table(&self, queue: &wgpu::Queue, slot: u32, stored: &StoredBlocks) {
        let at = self.table_at(slot);
        let places = at + u64::from(self.capacity.mapped);
        queue.write_buffer(&self.blocks, at * 4, bytemuck::cast_slice(&stored.table));
        queue.write_buffer(
            &self.blocks,
            places * 4,
            bytemuck::cast_slice(&stored.places),
        );
    }

    pub(crate) fn cells_at(&self, slot: u32) -> u64 {
        u64::from(slot) * 2 * self.held()
    }

    pub(crate) fn ground_at(&self, slot: u32) -> u64 {
        u64::from(slot) * entries(self.capacity, self.live).ground
    }

    pub(crate) fn sources_at(&self, slot: u32) -> u64 {
        u64::from(slot) * u64::from(self.capacity.sources)
    }

    pub(crate) fn ring_at(&self, slot: u32) -> u64 {
        let ring = entries(self.capacity, self.live).ring;
        u64::from(self.slots) * u64::from(self.capacity.sources) + u64::from(slot) * ring
    }

    pub(crate) fn table_at(&self, slot: u32) -> u64 {
        u64::from(slot) * entries(self.capacity, self.live).table
    }

    pub(crate) fn steps_at(&self, slot: u32) -> u64 {
        u64::from(slot) * entries(self.capacity, self.live).steps
    }

    pub(crate) fn averages_at(&self, slot: u32) -> u64 {
        u64::from(slot) * entries(self.capacity, self.live).averages
    }

    pub(crate) fn held(&self) -> u64 {
        entries(self.capacity, self.live).cells
    }

    pub(crate) fn bytes(&self) -> u64 {
        [
            &self.ground,
            &self.cells,
            &self.depths,
            &self.fluxes,
            &self.inputs,
            &self.blocks,
            &self.steps,
            &self.averages,
        ]
        .iter()
        .map(|buffer| buffer.size())
        .sum()
    }
}

fn inputs(device: &wgpu::Device, (capacity, slots): (Capacity, u32), live: bool) -> wgpu::Buffer {
    let all = u64::from(slots);
    let rings = if live {
        all * entries(capacity, live).ring
    } else {
        0
    };
    crate::buffers::storage(
        device,
        "engine/water-sim-inputs",
        (all * u64::from(capacity.sources) + rings) * 16,
        wgpu::BufferUsages::empty(),
    )
}

pub(crate) fn encode_steps(
    pass: &mut wgpu::ComputePass<'_>,
    pipelines: &Pipelines,
    buffers: &SlotBuffers,
    (steps, sourced, settling, bursting): (u32, u32, bool, bool),
    blocks: u32,
) {
    let [cells_x, cells_z] = grid(blocks);
    let [blocks_x, blocks_z] = grid(blocks.div_ceil(BLOCK_CELLS as u32));
    let slots = buffers.slots;
    for _ in 0..steps {
        if bursting {
            pass.set_bind_group(0, &buffers.groups[0], &[]);
            pass.set_pipeline(&pipelines.burst);
            pass.dispatch_workgroups(cells_x, cells_z, slots);
            pass.set_pipeline(&pipelines.push);
            pass.dispatch_workgroups(cells_x, cells_z, slots);
        }
        if let Some([touched_x, touched_z]) = buffers.touched {
            pass.set_bind_group(0, &buffers.groups[0], &[]);
            pass.set_pipeline(&pipelines.touch);
            pass.dispatch_workgroups(touched_x, touched_z, 1);
        }
        if settling {
            pass.set_bind_group(0, &buffers.groups[0], &[]);
            pass.set_pipeline(&pipelines.activity);
            pass.dispatch_workgroups(blocks_x, blocks_z, slots);
        }
        for (stage, group) in buffers.groups.iter().enumerate() {
            pass.set_bind_group(0, group, &[]);
            pass.set_pipeline(&pipelines.exchange);
            pass.dispatch_workgroups(cells_x, cells_z, slots);
            if stage == 0 && settling {
                pass.set_pipeline(&pipelines.waves);
                pass.dispatch_workgroups(cells_x, cells_z, slots);
                pass.set_pipeline(&pipelines.seconds);
                pass.dispatch_workgroups(cells_x, cells_z, slots);
            }
            pass.set_pipeline(&pipelines.drain);
            pass.dispatch_workgroups(cells_x, cells_z, slots);
            pass.set_pipeline(&pipelines.update);
            pass.dispatch_workgroups(cells_x, cells_z, slots);
        }
        if sourced > 0 {
            pass.set_pipeline(&pipelines.source);
            pass.dispatch_workgroups(sourced.div_ceil(WATER_SIM_SOURCE_GROUP), 1, slots);
        }
        pass.set_pipeline(&pipelines.ended);
        pass.dispatch_workgroups(cells_x, cells_z, slots);
        if settling {
            pass.set_pipeline(&pipelines.tally);
            pass.dispatch_workgroups(cells_x, cells_z, slots);
        }
    }
}

pub(crate) fn encode_look(
    pass: &mut wgpu::ComputePass<'_>,
    pipelines: &Pipelines,
    buffers: &SlotBuffers,
    blocks: u32,
) {
    let [groups_x, groups_z] = grid(blocks);
    pass.set_bind_group(0, &buffers.groups[0], &[]);
    pass.set_pipeline(&pipelines.look);
    pass.dispatch_workgroups(groups_x, groups_z, buffers.slots);
}

pub(crate) fn encode_drawn(
    pass: &mut wgpu::ComputePass<'_>,
    pipelines: &Pipelines,
    buffers: &SlotBuffers,
    blocks: u32,
) {
    let [groups_x, groups_z] = grid(blocks);
    pass.set_bind_group(0, &buffers.groups[0], &[]);
    pass.set_pipeline(&pipelines.drawn);
    pass.dispatch_workgroups(groups_x, groups_z, 1);
}
