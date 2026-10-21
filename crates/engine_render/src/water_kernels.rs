//! The shallow-water scheme's passes (`water_sim.wgsl`) and the buffers
//! they run over: slots of rectangles side by side, each a rectangle's
//! ground, cells, faces, fluxes, sources, foam, waves, seconds and blocks,
//! and sums. The square simulated around the camera (`water_sim.rs`) is one
//! slot with a ring past its edges; regions settling for their tiles'
//! records (`water_settle.rs`) fill as many as fit. One scheme runs both:
//! what sets them apart is each slot's flags ([`SimSlot`]).

use crate::bindings::WATER_SIM;
use crate::gpu_layouts::constants::{SIM_SLOTS, WATER_SIM_GROUP, WATER_SIM_SOURCE_GROUP};
use crate::gpu_layouts::water::{SimFrame, SimSlot, SimSlots};
use crate::shaders::entry::water_sim;

/// The scheme's passes: its group's layout and, once created, its
/// pipelines.
pub(crate) struct Kernels {
    pub(crate) layout: crate::pipelines::GroupLayout,
    pipelines: Option<Pipelines>,
}

/// The pipelines of the scheme's passes.
pub(crate) struct Pipelines {
    reconstruct: wgpu::ComputePipeline,
    exchange: wgpu::ComputePipeline,
    waves: wgpu::ComputePipeline,
    seconds: wgpu::ComputePipeline,
    update: wgpu::ComputePipeline,
    source: wgpu::ComputePipeline,
    pub(crate) carry: wgpu::ComputePipeline,
    pub(crate) restart: wgpu::ComputePipeline,
    churn: wgpu::ComputePipeline,
    churned: wgpu::ComputePipeline,
    activity: wgpu::ComputePipeline,
    look: wgpu::ComputePipeline,
}

impl Kernels {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            layout: crate::pipelines::layout(device, &WATER_SIM),
            pipelines: None,
        }
    }

    /// Its pipelines, created the first time they are asked for.
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
                reconstruct: entry(water_sim::sim_reconstruct),
                exchange: entry(water_sim::sim_exchange),
                waves: entry(water_sim::sim_waves),
                seconds: entry(water_sim::sim_seconds),
                update: entry(water_sim::sim_update),
                source: entry(water_sim::sim_source),
                carry: entry(water_sim::sim_carry),
                restart: entry(water_sim::sim_restart),
                churn: entry(water_sim::sim_churn),
                churned: entry(water_sim::sim_churned),
                activity: entry(water_sim::sim_activity),
                look: entry(water_sim::sim_look),
            }
        })
    }

    /// Its pipelines, if created.
    pub(crate) fn created(&self) -> Option<&Pipelines> {
        self.pipelines.as_ref()
    }
}

/// Workgroups, each a block of cells, along x and z of a rectangle of
/// `cells` along each.
pub(crate) fn groups_of(cells: [u32; 2]) -> [u32; 2] {
    cells.map(|count| count.div_ceil(WATER_SIM_GROUP))
}

/// The entries a slot holding rectangles of up to `capacity` cells along x
/// and z takes in each buffer: its cells, its corners, its faces between
/// columns and between rows, its blocks, and its ring where it has one.
fn slot_entries(capacity: [u32; 2]) -> [u64; 5] {
    let [x, z] = capacity.map(u64::from);
    let [blocks_x, blocks_z] = groups_of(capacity).map(u64::from);
    [
        x * z,
        (x + 1) * (z + 1),
        (x + 1) * z + x * (z + 1),
        blocks_x * blocks_z,
        4 * (x + z),
    ]
}

/// Bytes a slot holding rectangles of up to `capacity` cells along x and z
/// takes in every buffer.
pub(crate) fn slot_bytes(capacity: [u32; 2]) -> u64 {
    let [cells, corners, fluxes, blocks, _] = slot_entries(capacity);
    corners * 4
        + 2 * cells * 16
        + 4 * cells * 16
        + fluxes * 16
        + cells * 16
        + 2 * cells * 4
        + (cells + 2 * blocks) * 8
        + (2 * cells + blocks) * 16
}

/// The largest of a slot's entries in one buffer, its faces', in bytes.
pub(crate) fn slot_faces_bytes(capacity: [u32; 2]) -> u64 {
    4 * slot_entries(capacity)[0] * 16
}

/// The scheme's buffers: `slots` slots of rectangles up to `capacity`
/// cells along x and z, with a ring past each one's edges where it has
/// one; the uniforms of the slots' rectangles and of each stage's frame;
/// and the groups binding them for each stage.
pub(crate) struct SlotBuffers {
    pub(crate) capacity: [u32; 2],
    pub(crate) slots: u32,
    pub(crate) ground: wgpu::Buffer,
    pub(crate) cells: wgpu::Buffer,
    pub(crate) faces: wgpu::Buffer,
    pub(crate) fluxes: wgpu::Buffer,
    pub(crate) inputs: wgpu::Buffer,
    pub(crate) foam: wgpu::Buffer,
    pub(crate) steps: wgpu::Buffer,
    pub(crate) averages: wgpu::Buffer,
    pub(crate) squares: wgpu::Buffer,
    frames: [wgpu::Buffer; 2],
    pub(crate) groups: [wgpu::BindGroup; 2],
}

impl SlotBuffers {
    pub(crate) fn new(
        device: &wgpu::Device,
        layout: &crate::pipelines::GroupLayout,
        (capacity, slots): ([u32; 2], u32),
        ring: bool,
    ) -> Self {
        let [held, corners, faces_between, blocks, ring_entries] = slot_entries(capacity);
        let all = u64::from(slots);
        let storage = |label: &str, bytes: u64, extra: wgpu::BufferUsages| {
            crate::buffers::storage(device, label, bytes, extra)
        };
        let none = wgpu::BufferUsages::empty();
        let ground = storage("engine/water-sim-ground", all * corners * 4, none);
        let cells = storage("engine/water-sim-cells", all * 2 * held * 16, none);
        let faces = storage("engine/water-sim-faces", all * 4 * held * 16, none);
        let fluxes = storage("engine/water-sim-fluxes", all * faces_between * 16, none);
        let rings = if ring { all * ring_entries } else { 0 };
        let inputs = storage("engine/water-sim-inputs", (all * held + rings) * 16, none);
        let foam = storage("engine/water-sim-foam", all * 2 * held * 4, none);
        let steps = storage(
            "engine/water-sim-steps",
            all * (held + 2 * blocks) * 8,
            none,
        );
        let averages = storage(
            "engine/water-sim-averages",
            all * (2 * held + blocks) * 16,
            wgpu::BufferUsages::COPY_SRC,
        );
        let squares = crate::buffers::uniform(
            device,
            "engine/water-sim-slots",
            size_of::<SimSlots>() as u64,
        );
        let frames = [
            crate::buffers::uniform(
                device,
                "engine/water-sim-first",
                size_of::<SimFrame>() as u64,
            ),
            crate::buffers::uniform(
                device,
                "engine/water-sim-second",
                size_of::<SimFrame>() as u64,
            ),
        ];
        let groups = [0, 1].map(|stage| {
            crate::pipelines::group(
                device,
                "engine/water-sim",
                &WATER_SIM,
                layout,
                &[
                    crate::buffers::entry("frame", &frames[stage]),
                    crate::buffers::entry("sim_ground", &ground),
                    crate::buffers::entry("sim_cells", &cells),
                    crate::buffers::entry("sim_faces", &faces),
                    crate::buffers::entry("sim_fluxes", &fluxes),
                    crate::buffers::entry("sim_inputs", &inputs),
                    crate::buffers::entry("sim_foam", &foam),
                    crate::buffers::entry("sim_steps", &steps),
                    crate::buffers::entry("sim_slots", &squares),
                    crate::buffers::entry("sim_averages", &averages),
                ],
            )
        });
        Self {
            capacity,
            slots,
            ground,
            cells,
            faces,
            fluxes,
            inputs,
            foam,
            steps,
            averages,
            squares,
            frames,
            groups,
        }
    }

    /// Writes each stage's frame: the stage, the slots' count and capacity,
    /// and the cells the square around the camera moved by.
    pub(crate) fn write_frames(&self, queue: &wgpu::Queue, shift: [i32; 2]) {
        for (stage, buffer) in self.frames.iter().enumerate() {
            let frame = SimFrame {
                stage: [stage as u32, self.slots, self.capacity[0], self.capacity[1]],
                shift: [shift[0], shift[1], 0, 0],
            };
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&frame));
        }
    }

    /// Writes the slots' squares.
    pub(crate) fn write_squares(&self, queue: &wgpu::Queue, squares: &[SimSlot]) {
        let mut all = SimSlots {
            slot: [SimSlot {
                shape: [0; 4],
                step: [0.0; 4],
            }; SIM_SLOTS],
        };
        for (slot, square) in all.slot.iter_mut().zip(squares) {
            *slot = *square;
        }
        queue.write_buffer(&self.squares, 0, bytemuck::bytes_of(&all));
    }

    /// Where slot `slot`'s entries start in each buffer, as the shader
    /// finds them: its cells' (and foam's) states, its corners, its sources,
    /// its ring, its waves, seconds and blocks, and its sums.
    pub(crate) fn cells_at(&self, slot: u32) -> u64 {
        u64::from(slot) * 2 * self.held()
    }

    pub(crate) fn ground_at(&self, slot: u32) -> u64 {
        u64::from(slot) * slot_entries(self.capacity)[1]
    }

    pub(crate) fn sources_at(&self, slot: u32) -> u64 {
        u64::from(slot) * self.held()
    }

    pub(crate) fn ring_at(&self, slot: u32) -> u64 {
        let ring = slot_entries(self.capacity)[4];
        u64::from(self.slots) * self.held() + u64::from(slot) * ring
    }

    pub(crate) fn steps_at(&self, slot: u32) -> u64 {
        u64::from(slot) * (self.held() + 2 * self.blocks())
    }

    pub(crate) fn averages_at(&self, slot: u32) -> u64 {
        u64::from(slot) * (2 * self.held() + self.blocks())
    }

    /// Workgroups, each a block of cells, along x and z of the largest
    /// rectangle.
    pub(crate) fn groups(&self) -> [u32; 2] {
        groups_of(self.capacity)
    }

    /// Cells a slot holds at most, and blocks.
    pub(crate) fn held(&self) -> u64 {
        slot_entries(self.capacity)[0]
    }

    pub(crate) fn blocks(&self) -> u64 {
        slot_entries(self.capacity)[3]
    }

    /// Bytes of its buffers.
    pub(crate) fn bytes(&self) -> u64 {
        [
            &self.ground,
            &self.cells,
            &self.faces,
            &self.fluxes,
            &self.inputs,
            &self.foam,
            &self.steps,
            &self.averages,
        ]
        .iter()
        .map(|buffer| buffer.size())
        .sum()
    }
}

/// Encodes `steps` steps of every slot of `buffers` into `pass`, with
/// `pipelines`, over the `extent` cells along x and z the largest
/// rectangle stepping holds, whose rivers bring water to up to `sourced`
/// cells a slot: each step's blocks that step (where `sparse` slots run),
/// its two stages (each slot stepping locally finding its cells' seconds
/// from the first's waves), what rivers bring, and the foam.
pub(crate) fn encode_steps(
    pass: &mut wgpu::ComputePass<'_>,
    pipelines: &Pipelines,
    buffers: &SlotBuffers,
    (steps, sourced, local, sparse): (u32, u32, bool, bool),
    extent: [u32; 2],
) {
    let [cells_x, cells_z] = groups_of(extent);
    let [between_x, between_z] = groups_of(extent.map(|count| count + 1));
    let [blocks_x, blocks_z] = groups_of(groups_of(extent));
    let slots = buffers.slots;
    for _ in 0..steps {
        if sparse {
            pass.set_bind_group(0, &buffers.groups[0], &[]);
            pass.set_pipeline(&pipelines.activity);
            pass.dispatch_workgroups(blocks_x, blocks_z, slots);
        }
        for (stage, group) in buffers.groups.iter().enumerate() {
            pass.set_bind_group(0, group, &[]);
            pass.set_pipeline(&pipelines.reconstruct);
            pass.dispatch_workgroups(cells_x, cells_z, slots);
            pass.set_pipeline(&pipelines.exchange);
            pass.dispatch_workgroups(between_x, between_z, slots);
            if stage == 0 && local {
                pass.set_pipeline(&pipelines.waves);
                pass.dispatch_workgroups(cells_x, cells_z, slots);
                pass.set_pipeline(&pipelines.seconds);
                pass.dispatch_workgroups(cells_x, cells_z, slots);
            }
            pass.set_pipeline(&pipelines.update);
            pass.dispatch_workgroups(cells_x, cells_z, slots);
        }
        if sourced > 0 {
            pass.set_pipeline(&pipelines.source);
            pass.dispatch_workgroups(sourced.div_ceil(WATER_SIM_SOURCE_GROUP), 1, slots);
        }
        pass.set_pipeline(&pipelines.churn);
        pass.dispatch_workgroups(cells_x, cells_z, slots);
        pass.set_pipeline(&pipelines.churned);
        pass.dispatch_workgroups(cells_x, cells_z, slots);
    }
}

/// Encodes a look at the slots of `buffers` whose rectangles look this
/// dispatch into `pass`, with `pipelines`, over the `extent` cells along x
/// and z the largest of them holds.
pub(crate) fn encode_look(
    pass: &mut wgpu::ComputePass<'_>,
    pipelines: &Pipelines,
    buffers: &SlotBuffers,
    extent: [u32; 2],
) {
    let [groups_x, groups_z] = groups_of(extent);
    pass.set_bind_group(0, &buffers.groups[0], &[]);
    pass.set_pipeline(&pipelines.look);
    pass.dispatch_workgroups(groups_x, groups_z, buffers.slots);
}
