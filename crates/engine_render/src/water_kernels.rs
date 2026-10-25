//! The shallow-water scheme's passes (`water_sim.wgsl`) and the buffers
//! they run over: slots side by side, each a pool of blocks of cells
//! ([`StoredBlocks`]) with its table, and the blocks' ground, cells, faces'
//! depths, fluxes, sources, waves, outflow's shares, seconds and sums. The
//! square simulated around the camera (`water_sim.rs`) is one slot storing
//! every block of it, with a ring past its edges and the solids standing in
//! its water, which push it before each step; regions settling for their
//! tiles' records (`water_settle.rs`) fill as many as fit, each storing the
//! blocks its water can reach, and no solid touches them. One scheme runs
//! both: what sets them apart is each slot's flags ([`SimSlot`]).

use engine_core::terrain_data::shallow_water::CELL_MEANS;
use engine_core::terrain_data::{BLOCK_CELLS, BLOCK_GROUND, StoredBlocks};

use crate::bindings::WATER_SIM;
use crate::gpu_layouts::constants::{
    SIM_BLOCK_LOOKS, SIM_CELL_SUMS, SIM_SLOTS, WATER_SIM_SOURCE_GROUP,
};
use crate::gpu_layouts::water::{SimFrame, SimSlot, SimSlots, SimTouches};
use crate::shaders::entry::water_sim;

/// Workgroups along x a dispatch takes at most, as every device allows
/// them: a pool of more blocks dispatches rows of them.
const DISPATCH_ROW: u32 = 65_535;

/// The scheme's passes: its group's layout and, once created, its
/// pipelines.
pub(crate) struct Kernels {
    pub(crate) layout: crate::pipelines::GroupLayout,
    pipelines: Option<Pipelines>,
}

/// The pipelines of the scheme's passes.
pub(crate) struct Pipelines {
    exchange: wgpu::ComputePipeline,
    waves: wgpu::ComputePipeline,
    seconds: wgpu::ComputePipeline,
    drain: wgpu::ComputePipeline,
    update: wgpu::ComputePipeline,
    source: wgpu::ComputePipeline,
    touch: wgpu::ComputePipeline,
    churn: wgpu::ComputePipeline,
    churned: wgpu::ComputePipeline,
    tally: wgpu::ComputePipeline,
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
                exchange: entry(water_sim::sim_exchange),
                waves: entry(water_sim::sim_waves),
                seconds: entry(water_sim::sim_seconds),
                drain: entry(water_sim::sim_drain),
                update: entry(water_sim::sim_update),
                source: entry(water_sim::sim_source),
                touch: entry(water_sim::sim_touch),
                churn: entry(water_sim::sim_churn),
                churned: entry(water_sim::sim_churned),
                tally: entry(water_sim::sim_tally),
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

/// Workgroups along x and z dispatching `groups` of them, in rows.
pub(crate) fn grid(groups: u32) -> [u32; 2] {
    [groups.min(DISPATCH_ROW), groups.div_ceil(DISPATCH_ROW)]
}

/// What each slot of the scheme's buffers holds at most: the blocks of its
/// pool, the blocks its table maps (its rectangle's), the faces along its
/// rectangle's west and south edges (its cells along x and z), and the
/// cells its rivers bring water to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Capacity {
    pub(crate) blocks: u32,
    pub(crate) mapped: u32,
    pub(crate) edges: u32,
    pub(crate) sources: u32,
}

impl Capacity {
    /// What a slot holding `stored` blocks, with `sources` cells its rivers
    /// bring water to, takes.
    pub(crate) fn of(stored: &StoredBlocks, sources: usize) -> Self {
        Self {
            blocks: stored.stored() as u32,
            mapped: stored.table.len() as u32,
            edges: stored.cells[0] + stored.cells[1],
            sources: sources as u32,
        }
    }

    /// The least capacity holding both it and `other`.
    pub(crate) fn max(self, other: Self) -> Self {
        Self {
            blocks: self.blocks.max(other.blocks),
            mapped: self.mapped.max(other.mapped),
            edges: self.edges.max(other.edges),
            sources: self.sources.max(other.sources),
        }
    }

    /// Whether it holds `other`.
    pub(crate) fn holds(self, other: Self) -> bool {
        self.max(other) == self
    }
}

/// The entries a slot of `capacity` takes in each buffer: its cells, its
/// ground, its faces, its table, its steps, its sums and, where it has
/// one, its ring.
struct Entries {
    cells: u64,
    ground: u64,
    faces: u64,
    table: u64,
    steps: u64,
    averages: u64,
    ring: u64,
}

fn entries(capacity: Capacity) -> Entries {
    let blocks = u64::from(capacity.blocks);
    let cells = blocks * BLOCK_CELLS as u64;
    Entries {
        cells,
        ground: blocks * BLOCK_GROUND as u64,
        faces: 2 * cells + u64::from(capacity.edges),
        table: u64::from(capacity.mapped) + blocks,
        steps: cells + 2 * blocks,
        averages: (SIM_CELL_SUMS + CELL_MEANS as u64) * cells + SIM_BLOCK_LOOKS * blocks,
        ring: 4 * u64::from(capacity.edges),
    }
}

/// Bytes a slot of `capacity` takes in every buffer but the ring's: about
/// 206 a cell its pool holds, and 4 for each block its table maps.
pub(crate) fn slot_bytes(capacity: Capacity) -> u64 {
    let entries = entries(capacity);
    entries.ground * 4
        + 2 * entries.cells * 16
        + entries.cells * 8
        + entries.faces * 16
        + u64::from(capacity.sources) * 16
        + entries.table * 4
        + entries.steps * 8
        + entries.averages * 16
}

/// The most bytes a slot of `capacity` takes in one buffer: its cells' two
/// states', its fluxes', its sums' (about 113 a cell its pool holds), its
/// table's, its ground's or its sources'.
pub(crate) fn slot_largest_bytes(capacity: Capacity) -> u64 {
    let entries = entries(capacity);
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

/// The scheme's buffers: `slots` slots of `capacity`, each with a ring past
/// its rectangle's edges where it has one; the uniforms of the slots'
/// rectangles, of each stage's frame and of the solids standing in the live
/// square's water; the groups binding them for each stage; and the
/// workgroups along x and z the touch pass dispatches each step, a block's
/// each of the blocks those solids reach (none without any).
pub(crate) struct SlotBuffers {
    pub(crate) capacity: Capacity,
    pub(crate) slots: u32,
    ring: bool,
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
        ring: bool,
    ) -> Self {
        let entries = entries(capacity);
        let all = u64::from(slots);
        let storage = |label: &str, bytes: u64, extra: wgpu::BufferUsages| {
            crate::buffers::storage(device, label, bytes, extra)
        };
        let none = wgpu::BufferUsages::empty();
        let ground = storage("engine/water-sim-ground", all * entries.ground * 4, none);
        let cells = storage("engine/water-sim-cells", all * 2 * entries.cells * 16, none);
        let depths = storage("engine/water-sim-depths", all * entries.cells * 8, none);
        let fluxes = storage("engine/water-sim-fluxes", all * entries.faces * 16, none);
        let inputs = inputs(device, (capacity, slots), ring);
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
            ring,
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

    /// Binds its buffers into each stage's group, the stage's frame first.
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

    /// Holds `sources` cells rivers bring water to a slot: where it holds
    /// fewer, its inputs grow to the next power of two of them and its
    /// groups bind them, its other buffers keeping what they hold; grown,
    /// they hold nothing, so each slot's sources and ring are written anew.
    pub(crate) fn hold_sources(
        &mut self,
        device: &wgpu::Device,
        layout: &crate::pipelines::GroupLayout,
        sources: usize,
    ) {
        if sources as u32 <= self.capacity.sources {
            return;
        }
        self.capacity.sources = (sources as u32).next_power_of_two();
        self.inputs = inputs(device, (self.capacity, self.slots), self.ring);
        self.bind(device, layout);
    }

    /// Writes each stage's frame: the stage, the slots' count, the blocks
    /// a slot's pool holds and those its table maps, and the cells rivers
    /// bring water to and the faces along its west and south edges a slot
    /// holds.
    pub(crate) fn write_frames(&self, queue: &wgpu::Queue) {
        let capacity = self.capacity;
        for (stage, buffer) in self.frames.iter().enumerate() {
            let frame = SimFrame {
                stage: [stage as u32, self.slots, capacity.blocks, capacity.mapped],
                inputs: [capacity.sources, capacity.edges, 0, 0],
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
                blocks: [0; 4],
            }; SIM_SLOTS],
        };
        for (slot, square) in all.slot.iter_mut().zip(squares) {
            *slot = *square;
        }
        queue.write_buffer(&self.squares, 0, bytemuck::bytes_of(&all));
    }

    /// Writes the solids standing in the live square's water this frame,
    /// which push it before each step its frame runs; none touch it where
    /// `touches` holds none.
    pub(crate) fn write_touches(&mut self, queue: &wgpu::Queue, touches: Option<&SimTouches>) {
        self.touched = touches.map(|touches| [touches.blocks[2], touches.blocks[3]]);
        if let Some(touches) = touches {
            queue.write_buffer(&self.touches, 0, bytemuck::bytes_of(touches));
        }
    }

    /// Writes `stored`, the blocks slot `slot` stores, into its table.
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

    /// Where slot `slot`'s entries start in each buffer, as the shader
    /// finds them: its cells' states, its ground, its sources, its ring,
    /// its table, its waves, seconds and blocks, and its sums.
    pub(crate) fn cells_at(&self, slot: u32) -> u64 {
        u64::from(slot) * 2 * self.held()
    }

    pub(crate) fn ground_at(&self, slot: u32) -> u64 {
        u64::from(slot) * entries(self.capacity).ground
    }

    pub(crate) fn sources_at(&self, slot: u32) -> u64 {
        u64::from(slot) * u64::from(self.capacity.sources)
    }

    pub(crate) fn ring_at(&self, slot: u32) -> u64 {
        let ring = entries(self.capacity).ring;
        u64::from(self.slots) * u64::from(self.capacity.sources) + u64::from(slot) * ring
    }

    pub(crate) fn table_at(&self, slot: u32) -> u64 {
        u64::from(slot) * entries(self.capacity).table
    }

    pub(crate) fn steps_at(&self, slot: u32) -> u64 {
        u64::from(slot) * entries(self.capacity).steps
    }

    pub(crate) fn averages_at(&self, slot: u32) -> u64 {
        u64::from(slot) * entries(self.capacity).averages
    }

    /// Cells a slot's pool holds at most.
    pub(crate) fn held(&self) -> u64 {
        entries(self.capacity).cells
    }

    /// Bytes of its buffers.
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

/// The inputs of `slots` slots of `capacity`: each one's sources, then,
/// where they have one, each one's ring.
fn inputs(device: &wgpu::Device, (capacity, slots): (Capacity, u32), ring: bool) -> wgpu::Buffer {
    let all = u64::from(slots);
    let rings = if ring {
        all * entries(capacity).ring
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

/// Encodes `steps` steps of every slot of `buffers` into `pass`, with
/// `pipelines`, over the `blocks` the pool of the slot stepping storing the
/// most holds, whose rivers bring water to up to `sourced` cells a slot:
/// each step, the solids standing in the live square's water pushing it
/// where any do; its blocks that step (where `sparse` slots run), its two
/// stages (each finding the faces' fluxes, the share of its outflow each
/// cell can give and updating the cells, each slot stepping locally
/// finding its cells' seconds from the first's waves), what rivers bring,
/// and the foam; and, where slots step locally (settling, which sums its
/// water), each cell's water into its sums.
pub(crate) fn encode_steps(
    pass: &mut wgpu::ComputePass<'_>,
    pipelines: &Pipelines,
    buffers: &SlotBuffers,
    (steps, sourced, local, sparse): (u32, u32, bool, bool),
    blocks: u32,
) {
    let [cells_x, cells_z] = grid(blocks);
    let [blocks_x, blocks_z] = grid(blocks.div_ceil(BLOCK_CELLS as u32));
    let slots = buffers.slots;
    for _ in 0..steps {
        if let Some([touched_x, touched_z]) = buffers.touched {
            pass.set_bind_group(0, &buffers.groups[0], &[]);
            pass.set_pipeline(&pipelines.touch);
            pass.dispatch_workgroups(touched_x, touched_z, 1);
        }
        if sparse {
            pass.set_bind_group(0, &buffers.groups[0], &[]);
            pass.set_pipeline(&pipelines.activity);
            pass.dispatch_workgroups(blocks_x, blocks_z, slots);
        }
        for (stage, group) in buffers.groups.iter().enumerate() {
            pass.set_bind_group(0, group, &[]);
            pass.set_pipeline(&pipelines.exchange);
            pass.dispatch_workgroups(cells_x, cells_z, slots);
            if stage == 0 && local {
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
        pass.set_pipeline(&pipelines.churn);
        pass.dispatch_workgroups(cells_x, cells_z, slots);
        pass.set_pipeline(&pipelines.churned);
        pass.dispatch_workgroups(cells_x, cells_z, slots);
        if local {
            pass.set_pipeline(&pipelines.tally);
            pass.dispatch_workgroups(cells_x, cells_z, slots);
        }
    }
}

/// Encodes a look at the slots of `buffers` whose rectangles look this
/// dispatch into `pass`, with `pipelines`, over the `blocks` the pool of
/// the one storing the most holds.
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
