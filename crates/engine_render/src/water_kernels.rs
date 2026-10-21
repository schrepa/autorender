//! The shallow-water scheme's passes (`water_sim.wgsl`) and the buffers
//! they run over: slots of squares side by side, each a square's ground,
//! cells, faces, fluxes, sources, foam, waves and seconds, and sums. The
//! square simulated around the camera (`water_sim.rs`) is one slot with a
//! ring past its edges; tiles settling for their records
//! (`water_settle.rs`) fill as many as fit. One scheme runs both: what
//! sets them apart is each slot's flags ([`SimSlot`]).

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
            let entry = |entry| {
                crate::pipelines::compute_entry(device, &pipeline_layout, &module, entry)
            };
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
                look: entry(water_sim::sim_look),
            }
        })
    }

    /// Its pipelines, if created.
    pub(crate) fn created(&self) -> Option<&Pipelines> {
        self.pipelines.as_ref()
    }
}

/// Bytes a slot holding squares of `capacity` cells along a side takes in
/// each buffer, and the ring of one with a ring.
pub(crate) fn slot_bytes(capacity: u32) -> u64 {
    let cells = u64::from(capacity) * u64::from(capacity);
    let corners = (u64::from(capacity) + 1) * (u64::from(capacity) + 1);
    let fluxes = 2 * (u64::from(capacity) + 1) * u64::from(capacity);
    let groups = u64::from(capacity.div_ceil(WATER_SIM_GROUP));
    corners * 4
        + 2 * cells * 16
        + 4 * cells * 16
        + fluxes * 16
        + cells * 16
        + 2 * cells * 4
        + cells * 8
        + (2 * cells + groups * groups) * 16
}

/// The largest of a slot's entries in one buffer, its faces', in bytes.
pub(crate) fn slot_faces_bytes(capacity: u32) -> u64 {
    4 * u64::from(capacity) * u64::from(capacity) * 16
}

/// The scheme's buffers: `slots` slots of squares up to `capacity` cells
/// along a side, with a ring past each one's edges where it has one; the
/// uniforms of the slots' squares and of each stage's frame; and the groups
/// binding them for each stage.
pub(crate) struct SlotBuffers {
    pub(crate) capacity: u32,
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
        (capacity, slots): (u32, u32),
        ring: bool,
    ) -> Self {
        let held = u64::from(capacity) * u64::from(capacity);
        let all = u64::from(slots);
        let groups = u64::from(capacity.div_ceil(WATER_SIM_GROUP));
        let storage = |label: &str, bytes: u64, extra: wgpu::BufferUsages| {
            crate::buffers::storage(device, label, bytes, extra)
        };
        let none = wgpu::BufferUsages::empty();
        let ground = storage(
            "engine/water-sim-ground",
            all * (u64::from(capacity) + 1) * (u64::from(capacity) + 1) * 4,
            none,
        );
        let cells = storage("engine/water-sim-cells", all * 2 * held * 16, none);
        let faces = storage("engine/water-sim-faces", all * 4 * held * 16, none);
        let fluxes = storage(
            "engine/water-sim-fluxes",
            all * 2 * (u64::from(capacity) + 1) * u64::from(capacity) * 16,
            none,
        );
        let rings = if ring { all * 8 * u64::from(capacity) } else { 0 };
        let inputs = storage("engine/water-sim-inputs", (all * held + rings) * 16, none);
        let foam = storage("engine/water-sim-foam", all * 2 * held * 4, none);
        let steps = storage("engine/water-sim-steps", all * held * 8, none);
        let averages = storage(
            "engine/water-sim-averages",
            all * (2 * held + groups * groups) * 16,
            wgpu::BufferUsages::COPY_SRC,
        );
        let squares = crate::buffers::uniform(
            device,
            "engine/water-sim-slots",
            size_of::<SimSlots>() as u64,
        );
        let frames = [
            crate::buffers::uniform(device, "engine/water-sim-first", size_of::<SimFrame>() as u64),
            crate::buffers::uniform(device, "engine/water-sim-second", size_of::<SimFrame>() as u64),
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

    /// Writes each stage's frame: the stage, the slots' capacity and count,
    /// and the cells the square around the camera moved by.
    pub(crate) fn write_frames(&self, queue: &wgpu::Queue, shift: [i32; 2]) {
        for (stage, buffer) in self.frames.iter().enumerate() {
            let frame = SimFrame {
                stage: [stage as u32, self.capacity, self.slots, 0],
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
    /// its ring, and its sums.
    pub(crate) fn cells_at(&self, slot: u32) -> u64 {
        u64::from(slot) * 2 * self.held()
    }

    pub(crate) fn ground_at(&self, slot: u32) -> u64 {
        u64::from(slot) * (u64::from(self.capacity) + 1) * (u64::from(self.capacity) + 1)
    }

    pub(crate) fn sources_at(&self, slot: u32) -> u64 {
        u64::from(slot) * self.held()
    }

    pub(crate) fn ring_at(&self, slot: u32) -> u64 {
        u64::from(self.slots) * self.held() + u64::from(slot) * 8 * u64::from(self.capacity)
    }

    pub(crate) fn averages_at(&self, slot: u32) -> u64 {
        let groups = u64::from(self.capacity.div_ceil(WATER_SIM_GROUP));
        u64::from(slot) * (2 * self.held() + groups * groups)
    }

    /// Workgroups along a side of the largest square.
    pub(crate) fn groups_along(&self) -> u32 {
        self.capacity.div_ceil(WATER_SIM_GROUP)
    }

    fn held(&self) -> u64 {
        u64::from(self.capacity) * u64::from(self.capacity)
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
/// `pipelines`, whose rivers bring water to up to `sourced` cells a slot:
/// each step's two stages (each slot stepping locally finding its cells'
/// seconds from the first's waves), what rivers bring, and the foam.
pub(crate) fn encode_steps(
    pass: &mut wgpu::ComputePass<'_>,
    pipelines: &Pipelines,
    buffers: &SlotBuffers,
    (steps, sourced, local): (u32, u32, bool),
) {
    let cells = buffers.capacity.div_ceil(WATER_SIM_GROUP);
    let between = (buffers.capacity + 1).div_ceil(WATER_SIM_GROUP);
    let slots = buffers.slots;
    for _ in 0..steps {
        for (stage, group) in buffers.groups.iter().enumerate() {
            pass.set_bind_group(0, group, &[]);
            pass.set_pipeline(&pipelines.reconstruct);
            pass.dispatch_workgroups(cells, cells, slots);
            pass.set_pipeline(&pipelines.exchange);
            pass.dispatch_workgroups(between, between, slots);
            if stage == 0 && local {
                pass.set_pipeline(&pipelines.waves);
                pass.dispatch_workgroups(cells, cells, slots);
                pass.set_pipeline(&pipelines.seconds);
                pass.dispatch_workgroups(cells, cells, slots);
            }
            pass.set_pipeline(&pipelines.update);
            pass.dispatch_workgroups(cells, cells, slots);
        }
        if sourced > 0 {
            pass.set_pipeline(&pipelines.source);
            pass.dispatch_workgroups(sourced.div_ceil(WATER_SIM_SOURCE_GROUP), 1, slots);
        }
        pass.set_pipeline(&pipelines.churn);
        pass.dispatch_workgroups(cells, cells, slots);
        pass.set_pipeline(&pipelines.churned);
        pass.dispatch_workgroups(cells, cells, slots);
    }
}

/// Encodes a look at the slots of `buffers` whose squares look this
/// dispatch into `pass`, with `pipelines`.
pub(crate) fn encode_look(
    pass: &mut wgpu::ComputePass<'_>,
    pipelines: &Pipelines,
    buffers: &SlotBuffers,
) {
    let groups = buffers.groups_along();
    pass.set_bind_group(0, &buffers.groups[0], &[]);
    pass.set_pipeline(&pipelines.look);
    pass.dispatch_workgroups(groups, groups, buffers.slots);
}
