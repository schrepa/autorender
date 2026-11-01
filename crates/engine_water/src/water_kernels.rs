use bytemuck::Zeroable;
use engine_core::Error;
use engine_render::{
    Bind, Buffer, ComputePipeline, Count, Encode, Install, Lifetime, Pass, Prepare,
};

use crate::gpu::{
    SIM_BLOCK_LOOKS, SIM_CELL_SUMS, SIM_LIVE_LOOKS, SIM_SLOTS, SimFrame, SimSlot, SimSlots,
    SimTouches, WATER_SIM_SOURCE_GROUP,
};
use crate::shallow_water::CELL_MEANS;
use crate::water::WaterGpu;
use crate::{BLOCK_CELLS, BLOCK_GROUND, StoredBlocks};

const DISPATCH_ROW: u32 = 65_535;

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
            u64::from(SIM_LIVE_LOOKS) * cells
        } else {
            (u64::from(SIM_CELL_SUMS) + CELL_MEANS as u64) * cells
                + u64::from(SIM_BLOCK_LOOKS) * blocks
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    pub(crate) capacity: Capacity,
    pub(crate) slots: u32,
    pub(crate) live: bool,
}

impl Layout {
    fn entries(&self) -> Entries {
        entries(self.capacity, self.live)
    }

    pub(crate) fn cells_at(&self, slot: u32) -> u64 {
        u64::from(slot) * 2 * self.held()
    }

    pub(crate) fn ground_at(&self, slot: u32) -> u64 {
        u64::from(slot) * self.entries().ground
    }

    pub(crate) fn sources_at(&self, slot: u32) -> u64 {
        u64::from(slot) * u64::from(self.capacity.sources)
    }

    pub(crate) fn ring_at(&self, slot: u32) -> u64 {
        u64::from(self.slots) * u64::from(self.capacity.sources)
            + u64::from(slot) * self.entries().ring
    }

    pub(crate) fn table_at(&self, slot: u32) -> u64 {
        u64::from(slot) * self.entries().table
    }

    pub(crate) fn steps_at(&self, slot: u32) -> u64 {
        u64::from(slot) * self.entries().steps
    }

    pub(crate) fn averages_at(&self, slot: u32) -> u64 {
        u64::from(slot) * self.entries().averages
    }

    pub(crate) fn held(&self) -> u64 {
        self.entries().cells
    }

    pub(crate) fn frames(&self) -> [SimFrame; 2] {
        let capacity = self.capacity;
        [0, 1].map(|stage| SimFrame {
            stage: [stage, self.slots, capacity.blocks, capacity.mapped],
            inputs: [capacity.sources, capacity.edges, 0, 0],
        })
    }

    pub(crate) fn step_tails(&self) -> Vec<u64> {
        let counted = self.held() + 2 * u64::from(self.capacity.blocks);
        (0..self.slots)
            .map(|slot| (self.steps_at(slot) + counted) * 8)
            .collect()
    }
}

pub(crate) fn squares(written: &[SimSlot]) -> SimSlots {
    let mut all = SimSlots {
        slot: [SimSlot::zeroed(); SIM_SLOTS as usize],
    };
    for (slot, square) in all.slot.iter_mut().zip(written) {
        *slot = *square;
    }
    all
}

pub(crate) struct SlotHandles {
    pub(crate) ground: Buffer<WaterGpu>,
    pub(crate) cells: Buffer<WaterGpu>,
    pub(crate) depths: Buffer<WaterGpu>,
    pub(crate) fluxes: Buffer<WaterGpu>,
    pub(crate) inputs: Buffer<WaterGpu>,
    pub(crate) blocks: Buffer<WaterGpu>,
    pub(crate) steps: Buffer<WaterGpu>,
    pub(crate) averages: Buffer<WaterGpu>,
    pub(crate) squares: Buffer<WaterGpu>,
    pub(crate) frames: [Buffer<WaterGpu>; 2],
    pub(crate) touches: Buffer<WaterGpu>,
    layout: Option<Layout>,
}

impl SlotHandles {
    pub(crate) fn install(cx: &mut Install<'_, WaterGpu>, labels: [&'static str; 12]) -> Self {
        let kept = |cx: &mut Install<'_, WaterGpu>, label| cx.buffer(label, Lifetime::Kept);
        let ground = kept(cx, labels[0]);
        let cells = kept(cx, labels[1]);
        let depths = kept(cx, labels[2]);
        let fluxes = kept(cx, labels[3]);
        let inputs = kept(cx, labels[4]);
        let blocks = kept(cx, labels[5]);
        let steps = kept(cx, labels[6]);
        let averages = kept(cx, labels[7]);
        let squares = kept(cx, labels[8]);
        let frames = [kept(cx, labels[9]), kept(cx, labels[10])];
        let touches = kept(cx, labels[11]);
        Self {
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
            layout: None,
        }
    }

    pub(crate) fn hold(&mut self, layout: Layout, fresh: bool, cx: &mut Prepare<'_, WaterGpu>) {
        let entries = layout.entries();
        let all = u64::from(layout.slots);
        let rings = if layout.live { all * entries.ring } else { 0 };
        let sizes = [
            (self.ground, all * entries.ground * 4),
            (self.cells, all * 2 * entries.cells * 16),
            (self.depths, all * entries.cells * 8),
            (self.fluxes, all * entries.faces * 16),
            (
                self.inputs,
                (all * u64::from(layout.capacity.sources) + rings) * 16,
            ),
            (self.blocks, all * entries.table * 4),
            (self.steps, all * entries.steps * 8),
            (self.averages, all * entries.averages * 16),
        ];
        let inputs_grew = self
            .layout
            .is_some_and(|held| held.capacity.sources != layout.capacity.sources);
        for (at, (buffer, bytes)) in sizes.into_iter().enumerate() {
            if fresh || (at == 4 && inputs_grew) {
                cx.release_buffer(buffer);
            }
            cx.allocate_buffer(buffer, bytes);
        }
        cx.allocate_buffer(self.squares, size_of::<SimSlots>() as u64);
        for frame in self.frames {
            cx.allocate_buffer(frame, size_of::<SimFrame>() as u64);
        }
        cx.allocate_buffer(self.touches, size_of::<SimTouches>() as u64);
        self.layout = Some(layout);
    }

    pub(crate) fn release(&mut self, cx: &mut Prepare<'_, WaterGpu>) {
        for buffer in [
            self.ground,
            self.cells,
            self.depths,
            self.fluxes,
            self.inputs,
            self.blocks,
            self.steps,
            self.averages,
        ] {
            cx.allocate_buffer(buffer, 16);
        }
        cx.allocate_buffer(self.squares, size_of::<SimSlots>() as u64);
        for frame in self.frames {
            cx.allocate_buffer(frame, size_of::<SimFrame>() as u64);
        }
        cx.allocate_buffer(self.touches, size_of::<SimTouches>() as u64);
        self.layout = None;
    }

    pub(crate) fn write_frames(&self, layout: &Layout, cx: &mut Prepare<'_, WaterGpu>) {
        for (buffer, frame) in self.frames.iter().zip(layout.frames()) {
            cx.write(*buffer, 0, &[frame]);
        }
        for at in layout.step_tails() {
            cx.write(self.steps, at, &[0.0_f32; 2]);
        }
    }

    pub(crate) fn write_table(
        &self,
        layout: &Layout,
        slot: u32,
        stored: &StoredBlocks,
        cx: &mut Prepare<'_, WaterGpu>,
    ) {
        let at = layout.table_at(slot);
        let places = at + u64::from(layout.capacity.mapped);
        cx.write(self.blocks, at * 4, &stored.table);
        cx.write(self.blocks, places * 4, &stored.places);
    }

    fn bound(&self, stage: usize) -> Vec<(engine_core::Var, Bind<WaterGpu>)> {
        use crate::shaders::water::sim;
        vec![
            (sim::FRAME, Bind::Buffer(self.frames[stage])),
            (sim::SIM_GROUND, Bind::Buffer(self.ground)),
            (sim::SIM_CELLS, Bind::Buffer(self.cells)),
            (sim::SIM_DEPTHS, Bind::Buffer(self.depths)),
            (sim::SIM_FLUXES, Bind::Buffer(self.fluxes)),
            (sim::SIM_INPUTS, Bind::Buffer(self.inputs)),
            (sim::SIM_BLOCKS, Bind::Buffer(self.blocks)),
            (sim::SIM_STEPS, Bind::Buffer(self.steps)),
            (sim::SIM_SLOTS, Bind::Buffer(self.squares)),
            (sim::SIM_AVERAGES, Bind::Buffer(self.averages)),
            (sim::SIM_TOUCHES, Bind::Buffer(self.touches)),
        ]
    }
}

pub(crate) struct Kernels {
    exchange: [ComputePipeline<WaterGpu>; 2],
    drain: [ComputePipeline<WaterGpu>; 2],
    update: [ComputePipeline<WaterGpu>; 2],
    waves: ComputePipeline<WaterGpu>,
    seconds: ComputePipeline<WaterGpu>,
    source: ComputePipeline<WaterGpu>,
    touch: ComputePipeline<WaterGpu>,
    burst: ComputePipeline<WaterGpu>,
    push: ComputePipeline<WaterGpu>,
    ended: ComputePipeline<WaterGpu>,
    tally: ComputePipeline<WaterGpu>,
    activity: ComputePipeline<WaterGpu>,
    look: ComputePipeline<WaterGpu>,
    drawn: ComputePipeline<WaterGpu>,
}

impl Kernels {
    pub(crate) fn install(
        cx: &mut Install<'_, WaterGpu>,
        pass: Pass<WaterGpu>,
        handles: &SlotHandles,
    ) -> Result<Self, Error> {
        use crate::shaders::water::sim;
        let mut entry = |entry, stage: usize| {
            let bound = crate::water::reached(cx, &[entry], handles.bound(stage));
            cx.compute_pipeline(pass, entry, &bound)
        };
        Ok(Self {
            exchange: [entry(sim::SIM_EXCHANGE, 0)?, entry(sim::SIM_EXCHANGE, 1)?],
            drain: [entry(sim::SIM_DRAIN, 0)?, entry(sim::SIM_DRAIN, 1)?],
            update: [entry(sim::SIM_UPDATE, 0)?, entry(sim::SIM_UPDATE, 1)?],
            waves: entry(sim::SIM_WAVES, 0)?,
            seconds: entry(sim::SIM_SECONDS, 0)?,
            source: entry(sim::SIM_SOURCE, 1)?,
            touch: entry(sim::SIM_TOUCH, 0)?,
            burst: entry(sim::SIM_BURST, 0)?,
            push: entry(sim::SIM_PUSH, 0)?,
            ended: entry(sim::SIM_ENDED, 1)?,
            tally: entry(sim::SIM_TALLY, 1)?,
            activity: entry(sim::SIM_ACTIVITY, 0)?,
            look: entry(sim::SIM_LOOK, 0)?,
            drawn: entry(sim::SIM_DRAWN, 0)?,
        })
    }

    pub(crate) fn encode_steps(
        &self,
        cx: &mut Encode<'_, WaterGpu>,
        (steps, sourced, settling, bursting): (u32, u32, bool, bool),
        (blocks, slots, touched): (u32, u32, Option<[u32; 2]>),
    ) {
        let [cells_x, cells_z] = grid(blocks);
        let [blocks_x, blocks_z] = grid(blocks.div_ceil(BLOCK_CELLS as u32));
        let run = |cx: &mut Encode<'_, WaterGpu>, pipeline: &ComputePipeline<WaterGpu>, groups| {
            cx.dispatch(pipeline, Count::Groups(groups));
        };
        for _ in 0..steps {
            if bursting {
                run(cx, &self.burst, [cells_x, cells_z, slots]);
                run(cx, &self.push, [cells_x, cells_z, slots]);
            }
            if let Some([touched_x, touched_z]) = touched {
                run(cx, &self.touch, [touched_x, touched_z, 1]);
            }
            if settling {
                run(cx, &self.activity, [blocks_x, blocks_z, slots]);
            }
            for stage in 0..2 {
                run(cx, &self.exchange[stage], [cells_x, cells_z, slots]);
                if stage == 0 && settling {
                    run(cx, &self.waves, [cells_x, cells_z, slots]);
                    run(cx, &self.seconds, [cells_x, cells_z, slots]);
                }
                run(cx, &self.drain[stage], [cells_x, cells_z, slots]);
                run(cx, &self.update[stage], [cells_x, cells_z, slots]);
            }
            if sourced > 0 {
                let groups = [sourced.div_ceil(WATER_SIM_SOURCE_GROUP), 1, slots];
                run(cx, &self.source, groups);
            }
            run(cx, &self.ended, [cells_x, cells_z, slots]);
            if settling {
                run(cx, &self.tally, [cells_x, cells_z, slots]);
            }
        }
    }

    pub(crate) fn encode_look(&self, cx: &mut Encode<'_, WaterGpu>, (blocks, slots): (u32, u32)) {
        let [groups_x, groups_z] = grid(blocks);
        cx.dispatch(&self.look, Count::Groups([groups_x, groups_z, slots]));
    }

    pub(crate) fn encode_drawn(&self, cx: &mut Encode<'_, WaterGpu>, blocks: u32) {
        let [groups_x, groups_z] = grid(blocks);
        cx.dispatch(&self.drawn, Count::Groups([groups_x, groups_z, 1]));
    }
}

pub(crate) struct SetupHandles {
    pub(crate) frame: Buffer<WaterGpu>,
    pub(crate) list: Buffer<WaterGpu>,
}

impl SetupHandles {
    pub(crate) fn install(cx: &mut Install<'_, WaterGpu>, labels: [&'static str; 2]) -> Self {
        let frame = cx.buffer(labels[0], Lifetime::Kept);
        let list = cx.buffer(labels[1], Lifetime::Kept);
        Self { frame, list }
    }

    pub(crate) fn live_at(room: usize) -> u64 {
        room as u64 * 16
    }

    pub(crate) fn hold(
        &self,
        (room, live_room): (usize, usize),
        fresh: bool,
        cx: &mut Prepare<'_, WaterGpu>,
    ) {
        if fresh {
            cx.release_buffer(self.list);
        }
        cx.allocate_buffer(self.frame, size_of::<crate::gpu::SimSetup>() as u64);
        cx.allocate_buffer(self.list, Self::live_at(room) + live_room as u64 * 4);
    }

    pub(crate) fn release(&self, cx: &mut Prepare<'_, WaterGpu>) {
        cx.allocate_buffer(self.frame, size_of::<crate::gpu::SimSetup>() as u64);
        cx.allocate_buffer(self.list, 16);
    }

    pub(crate) fn bound(&self, slots: &SlotHandles) -> Vec<(engine_core::Var, Bind<WaterGpu>)> {
        use crate::shaders::water::setup;
        vec![
            (setup::SETUP_FRAME, Bind::Buffer(self.frame)),
            (setup::SETUP_LIST, Bind::Buffer(self.list)),
            (setup::SETUP_GROUND, Bind::Buffer(slots.ground)),
            (setup::SETUP_CELLS, Bind::Buffer(slots.cells)),
            (setup::SETUP_LOOKS, Bind::Buffer(slots.averages)),
            (setup::SETUP_EDGES, Bind::Buffer(slots.inputs)),
        ]
    }
}
