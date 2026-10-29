use std::cell::Cell;
use std::sync::Arc;

use bytemuck::Zeroable;
use engine_core::WaterObstacle;
use engine_core::glam::{Mat3, Vec2, Vec3};
use engine_faces::heightfield::HeightfieldFace;
use engine_render::bindings::EMPTY;
use engine_render::pipelines::GroupLayout;
use engine_terrain::shallow_water::step_seconds;
use engine_terrain::{
    BLOCK_SIDE, Footprint, NO_BLOCK, SIM_TOUCHES, StoredBlocks, WaterRead, WaterWindow,
    live_steps,
};

use crate::bindings::WATER_SETUP;
use crate::gpu_layouts::constants::{
    LIVE_CELL, LIVE_HEAD, LIVE_HELD, LIVE_NONE, NO_WATER_SURFACE, SETUP_RISEN, SETUP_SOLID_ENTRIES,
    SETUP_STANDS, SETUP_WHOLE, SIM_BURSTING, SIM_LIVE, SIM_STANDING, SIM_STEPPING,
    WATER_SETUP_RING_GROUP,
};
use crate::gpu_layouts::water::{SimSetup, SimSlot, SimTouch, SimTouches};
use crate::shaders::entry::meshes;
use crate::water_kernels::{
    Capacity, Kernels, SlotBuffers, encode_drawn, encode_steps, grid, live_largest_bytes,
};

pub(crate) use engine_terrain::LONGEST_FRAME;

const RISEN: usize = 256;

pub(crate) struct Live<'a> {
    pub(crate) window: &'a Arc<WaterWindow>,
    pub(crate) touches: &'a [WaterObstacle],
    pub(crate) beyond: u32,
    pub(crate) ground: Option<LiveGround<'a>>,
}

pub(crate) struct LiveGround<'a> {
    pub(crate) group: &'a wgpu::BindGroup,
    pub(crate) uniform: u32,
    pub(crate) terrain: u64,
    pub(crate) held: &'a dyn HeightfieldFace,
}

impl LiveGround<'_> {
    fn resident(&self, read: &WaterRead) -> bool {
        read.ready
            && read.tile.as_ref().is_some_and(|tile| {
                self.held
                    .holds(self.terrain, read.key, Arc::as_ptr(tile).cast())
            })
    }
}

pub(crate) struct Setup {
    empty: wgpu::BindGroup,
    pub(crate) layout: GroupLayout,
    grounded: bool,
    pipelines: Option<[wgpu::ComputePipeline; 3]>,
    ground: Option<(wgpu::BindGroup, u32)>,
}

impl Setup {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let empty_layout = engine_render::pipelines::layout(device, &EMPTY);
        let empty = engine_render::pipelines::group(
            device,
            "engine/water-setup-empty",
            &EMPTY,
            &empty_layout,
            &[],
        );
        Self {
            empty,
            layout: engine_render::pipelines::layout(device, &WATER_SETUP),
            grounded: false,
            pipelines: None,
            ground: None,
        }
    }

    pub(crate) fn hold(&mut self, ground: Option<&LiveGround<'_>>) {
        self.ground = ground.map(|ground| {
            self.grounded = true;
            (ground.group.clone(), ground.uniform)
        });
    }

    pub(crate) fn ensure(&mut self, device: &wgpu::Device) {
        if self.pipelines.is_some() || !self.grounded {
            return;
        }
        let layout = engine_render::pipelines::layout_of(
            device,
            &crate::pipeline_layouts::WATER_SETUP_PIPELINES,
        );
        let entry = |entry| engine_render::pipelines::compute_entry(device, &layout, entry);
        self.pipelines = Some([
            entry(meshes::setup_blocks),
            entry(meshes::setup_ring),
            entry(meshes::setup_raise),
        ]);
    }

    pub(crate) fn sets_up(&self) -> bool {
        self.pipelines.is_some() && self.ground.is_some()
    }

    fn bind<'a>(
        &'a self,
        pass: &mut wgpu::ComputePass<'_>,
        group: &wgpu::BindGroup,
    ) -> Option<&'a [wgpu::ComputePipeline; 3]> {
        let (Some(pipelines), Some((terrain, uniform))) = (&self.pipelines, &self.ground) else {
            return None;
        };
        pass.set_bind_group(0, &self.empty, &[]);
        pass.set_bind_group(1, terrain, &[0, *uniform]);
        pass.set_bind_group(2, group, &[]);
        Some(pipelines)
    }

    pub(crate) fn encode_raise(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        group: &wgpu::BindGroup,
        blocks: u32,
    ) -> bool {
        let Some([_, _, raise]) = self.bind(pass, group) else {
            return false;
        };
        let [groups_x, groups_z] = grid(blocks);
        pass.set_pipeline(raise);
        pass.dispatch_workgroups(groups_x, groups_z, 1);
        true
    }

    fn encode(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        group: &wgpu::BindGroup,
        (blocks, ghosts): (u32, u32),
    ) -> bool {
        let Some([set_blocks, set_ring, _]) = self.bind(pass, group) else {
            return false;
        };
        if blocks > 0 {
            let [groups_x, groups_z] = grid(blocks);
            pass.set_pipeline(set_blocks);
            pass.dispatch_workgroups(groups_x, groups_z, 1);
        }
        if ghosts > 0 {
            pass.set_pipeline(set_ring);
            pass.dispatch_workgroups(ghosts.div_ceil(WATER_SETUP_RING_GROUP), 1, 1);
        }
        true
    }
}

pub(crate) struct SetupBuffers {
    pub(crate) frame: wgpu::Buffer,
    pub(crate) list: wgpu::Buffer,
    pub(crate) room: usize,
    pub(crate) live: wgpu::Buffer,
    pub(crate) live_room: usize,
    pub(crate) group: wgpu::BindGroup,
}

impl SetupBuffers {
    pub(crate) fn new(
        device: &wgpu::Device,
        layout: &GroupLayout,
        buffers: &SlotBuffers,
        (entries, words): (usize, usize),
    ) -> Self {
        let room = entries.max(1).next_power_of_two();
        let live_room = words.max(4).next_power_of_two();
        let frame = engine_render::buffers::uniform(
            device,
            "engine/water-setup",
            size_of::<SimSetup>() as u64,
        );
        let list = engine_render::buffers::storage(
            device,
            "engine/water-setup-list",
            room as u64 * 16,
            wgpu::BufferUsages::empty(),
        );
        let live = engine_render::buffers::storage(
            device,
            "engine/water-setup-live",
            live_room as u64 * 4,
            wgpu::BufferUsages::empty(),
        );
        let group = engine_render::pipelines::group(
            device,
            "engine/water-setup",
            &WATER_SETUP,
            layout,
            &[
                engine_render::buffers::entry("setup_frame", &frame),
                engine_render::buffers::entry("setup_list", &list),
                engine_render::buffers::entry("setup_ground", &buffers.ground),
                engine_render::buffers::entry("setup_cells", &buffers.cells),
                engine_render::buffers::entry("setup_looks", &buffers.averages),
                engine_render::buffers::entry("setup_edges", &buffers.inputs),
                engine_render::buffers::entry("setup_live", &live),
            ],
        );
        Self {
            frame,
            list,
            room,
            live,
            live_room,
            group,
        }
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.frame.size() + self.list.size() + self.live.size()
    }
}

pub(crate) struct WaterSim {
    kernels: Kernels,
    setup: Setup,
    window: Option<SimWindow>,
    steps: u32,
    behind: u32,
    tick: Option<i64>,
    generation: u64,
    touches: [u32; 2],
}

struct SimWindow {
    source: Arc<WaterWindow>,
    fitted: Arc<WaterWindow>,
    pool: u32,
    stored: StoredBlocks,
    wrap: [u32; 2],
    buffers: SlotBuffers,
    setup: SetupBuffers,
    sourced: u32,
    solids: u32,
    tiles: [u32; 4],
    held: Vec<Option<Held>>,
    sweep: usize,
    changed: Vec<[Vec2; 2]>,
    restand: bool,
    ring: bool,
    table: bool,
    ready: bool,
    waiting: u32,
    listed: Option<Listed>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Held {
    place: [i64; 2],
    runs: bool,
}

struct Listed {
    slots: Vec<u32>,
    whole: u32,
    blocks: u32,
    ghosts: u32,
    recorded: Cell<bool>,
}

impl WaterSim {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            kernels: Kernels::new(device),
            setup: Setup::new(device),
            window: None,
            steps: 0,
            behind: 0,
            tick: None,
            generation: 0,
            touches: [0; 2],
        }
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &engine_render::gpu_calls::Queue,
        live: Option<Live<'_>>,
        seconds: f64,
    ) {
        if let Some(current) = self.window.as_mut() {
            current.settle();
        }
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
            self.setup.hold(None);
            (self.steps, self.behind, self.tick, self.touches) = (0, 0, None, [0; 2]);
            return;
        };
        self.setup.hold(live.ground.as_ref());
        let sets_up = self.setup.sets_up();
        self.kernels.pipelines(device);
        let layouts = (&self.kernels.layout, &self.setup.layout);
        if let Some(fitted) = fitted {
            let same_size = self.window.as_ref().is_some_and(|current| {
                current.fitted.cells == fitted.cells
                    && current.fitted.cell == fitted.cell
                    && current.pool == pool_of(&fitted)
            });
            if same_size {
                if let Some(current) = self.window.as_mut() {
                    current.follow((device, queue, layouts), live.window, fitted);
                }
            } else {
                self.window = Some(SimWindow::new(
                    (device, queue, layouts),
                    live.window,
                    fitted,
                ));
                self.generation += 1;
            }
        }
        let Some(current) = self.window.as_mut() else {
            return;
        };
        let ready = current.ready;
        let resident = |read: &WaterRead| {
            live.ground
                .as_ref()
                .is_some_and(|ground| ground.resident(read))
        };
        current.set_up(queue, sets_up, &resident);
        if current.ready != ready {
            self.generation += 1;
        }
        let (cells, cell) = (current.fitted.cells, current.fitted.cell);
        let step = step_seconds(cell);
        let tick = (seconds / f64::from(step)).floor() as i64;
        let most = i64::from(live_steps(cell));
        let passed = self.tick.map_or(0, |last| (tick - last).max(0));
        let starting = self.tick.map_or(tick, |last| last + 1);
        self.tick = Some(tick);
        (self.steps, self.behind) = if current.ready {
            (passed.min(most) as u32, (passed - most).max(0) as u32)
        } else {
            (0, 0)
        };
        let pushing = touching(&current.fitted, live.touches);
        self.touches = [
            pushing.as_ref().map_or(0, |pushing| pushing.count[0]),
            live.beyond,
        ];
        let [columns, rows] = current.stored.blocks;
        let pooled = current.stored.stored() as u32;
        let [x, z] = current.fitted.first;
        current.buffers.write_frames(queue);
        current.buffers.write_squares(
            queue,
            &[SimSlot {
                shape: [
                    cells,
                    cells,
                    SIM_STEPPING | SIM_LIVE | SIM_BURSTING,
                    current.sourced,
                ],
                step: [cell, step, 0.0, 0.0],
                blocks: [columns, rows, pooled, 0],
                place: [x as u32, z as u32, starting as u32, 0],
            }],
        );
        current.buffers.write_touches(queue, pushing.as_ref());
    }

    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &engine_render::profiling::GpuProfiler,
    ) {
        let (Some(pipelines), Some(window)) = (self.kernels.created(), &self.window) else {
            return;
        };
        if !window.ready {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/water-sim"),
            timestamp_writes: profiler.span("water_sim").map(|span| span.compute()),
        });
        if let Some(listed) = window.listed.as_ref() {
            let recorded = self.setup.encode(
                &mut pass,
                &window.setup.group,
                (listed.blocks, listed.ghosts),
            );
            listed.recorded.set(recorded);
        }
        let blocks = window.stored.stored() as u32;
        if self.steps > 0 {
            encode_steps(
                &mut pass,
                pipelines,
                &window.buffers,
                (self.steps, window.sourced, false, true),
                blocks,
            );
        }
        encode_drawn(&mut pass, pipelines, &window.buffers, blocks);
    }

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

    pub(crate) fn cells(&self) -> [u32; 7] {
        self.window
            .as_ref()
            .filter(|window| window.ready)
            .map_or([0; 7], |window| {
                [
                    window.fitted.cells,
                    window.source.asked.saturating_sub(window.source.cells),
                    window.source.cells - window.fitted.cells,
                    self.steps,
                    self.behind,
                    window.listed.as_ref().map_or(0, |listed| listed.whole),
                    window.waiting,
                ]
            })
    }

    pub(crate) fn touches(&self) -> [u32; 2] {
        self.touches
    }

    pub(crate) fn ensure_setup(&mut self, device: &wgpu::Device) {
        self.setup.ensure(device);
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.window
            .as_ref()
            .map_or(0, |window| window.buffers.bytes() + window.setup.bytes())
    }
}

impl SimWindow {
    fn new(
        (device, queue, layouts): (
            &wgpu::Device,
            &engine_render::gpu_calls::Queue,
            (&GroupLayout, &GroupLayout),
        ),
        source: &Arc<WaterWindow>,
        fitted: Arc<WaterWindow>,
    ) -> Self {
        let pool = pool_of(&fitted);
        let (stored, wrap) = wrapped(&fitted, pool);
        let slots = stored.stored();
        let buffers = SlotBuffers::new(
            device,
            layouts.0,
            (Capacity::of(&stored, fitted.sources.len().max(1)), 1),
            true,
        );
        let setup = SetupBuffers::new(device, layouts.1, &buffers, (entries(&fitted, slots), 0));
        let mut window = Self {
            source: source.clone(),
            fitted,
            pool,
            stored,
            wrap,
            buffers,
            setup,
            sourced: 0,
            solids: 0,
            tiles: [0; 4],
            held: vec![None; slots],
            sweep: 0,
            changed: Vec::new(),
            restand: true,
            ring: true,
            table: true,
            ready: false,
            waiting: 0,
            listed: None,
        };
        window.write_square((device, queue, layouts));
        window
    }

    fn follow(
        &mut self,
        gpu: (
            &wgpu::Device,
            &engine_render::gpu_calls::Queue,
            (&GroupLayout, &GroupLayout),
        ),
        source: &Arc<WaterWindow>,
        fitted: Arc<WaterWindow>,
    ) {
        self.changed = changed(&self.source.reads, &source.reads);
        if (fitted.datum, fitted.split) != (self.fitted.datum, self.fitted.split) {
            self.held.fill(None);
        }
        (self.stored, self.wrap) = wrapped(&fitted, self.pool);
        self.source = source.clone();
        self.fitted = fitted;
        self.write_square(gpu);
    }

    fn write_square(
        &mut self,
        (device, queue, (layout, setup)): (
            &wgpu::Device,
            &engine_render::gpu_calls::Queue,
            (&GroupLayout, &GroupLayout),
        ),
    ) {
        let grown = self
            .buffers
            .hold_sources(device, layout, self.fitted.sources.len());
        let entries = entries(&self.fitted, self.held.len());
        let (tiles, words) = live_words(&self.fitted);
        if grown || entries > self.setup.room || words.len() > self.setup.live_room {
            let room = (
                entries.max(self.setup.room),
                words.len().max(self.setup.live_room),
            );
            self.setup = SetupBuffers::new(device, setup, &self.buffers, room);
        }
        if !words.is_empty() {
            queue.write_buffer(&self.setup.live, 0, bytemuck::cast_slice(&words));
        }
        self.tiles = tiles;
        self.sourced = write_sources(queue, &self.buffers, &self.fitted, &self.stored);
        self.solids = write_solids(queue, &self.setup.list, &self.fitted.solids);
        (self.restand, self.ring, self.table) = (true, true, true);
    }

    fn settle(&mut self) {
        let Some(listed) = self.listed.take() else {
            return;
        };
        if listed.recorded.get() {
            return;
        }
        for &slot in &listed.slots {
            self.held[slot as usize] = None;
        }
        if listed.ghosts > 0 {
            self.ring = true;
        }
    }

    fn set_up(
        &mut self,
        queue: &engine_render::gpu_calls::Queue,
        sets_up: bool,
        resident: &dyn Fn(&WaterRead) -> bool,
    ) {
        let side = self.fitted.blocks();
        let corner = self
            .fitted
            .first
            .map(|first| first.div_euclid(i64::from(BLOCK_SIDE)));
        let cell = self.fitted.cell;
        let block = cell * BLOCK_SIDE as f32;
        let changed = std::mem::take(&mut self.changed);
        let missing: Vec<[Vec2; 2]> = self
            .source
            .reads
            .iter()
            .filter(|read| !resident(read))
            .map(|read| read.square)
            .collect();
        let reaches = |place: [i64; 2], squares: &[[Vec2; 2]]| {
            let low = Vec2::new(place[0] as f32, place[1] as f32) * block - Vec2::splat(cell);
            let high = low + Vec2::splat(block + 2.0 * cell);
            squares
                .iter()
                .any(|[least, most]| high.cmpge(*least).all() && low.cmple(*most).all())
        };
        let restand = sets_up && std::mem::take(&mut self.restand);
        let mut listed: Vec<[u32; 4]> = Vec::new();
        let mut slots: Vec<u32> = Vec::new();
        let (mut whole, mut waiting, mut deferred) = (0, 0, false);
        for z in 0..side {
            for x in 0..side {
                let slot = self.stored.table[(z * side + x) as usize];
                let place = [corner[0] + i64::from(x), corner[1] + i64::from(z)];
                let touched = reaches(place, &changed);
                let held = &mut self.held[slot as usize];
                if !sets_up {
                    if held.is_none_or(|held| held.place != place) || touched {
                        *held = None;
                        deferred = true;
                    }
                    waiting += u32::from(held.is_none_or(|held| !held.runs));
                    continue;
                }
                let runs = !reaches(place, &missing);
                waiting += u32::from(!runs);
                let set = Some(Held { place, runs });
                if *held == set && (runs || !restand) && !touched {
                    continue;
                }
                self.table |= held.is_none_or(|held| held.runs != runs);
                *held = set;
                slots.push(slot);
                if runs {
                    whole += 1;
                    listed.push(entry(slot, place, SETUP_WHOLE));
                } else {
                    listed.push(entry(slot, place, SETUP_STANDS));
                }
            }
        }
        self.ready = !deferred;
        self.waiting = waiting;
        if !sets_up {
            return;
        }
        if std::mem::take(&mut self.table) {
            self.write_table(queue);
        }
        let set = slots.len();
        slots.sort_unstable();
        let pooled = self.held.len();
        let mut risen = 0;
        for _ in 0..pooled {
            if risen == RISEN || set == pooled {
                break;
            }
            let slot = self.sweep;
            self.sweep = (slot + 1) % pooled;
            let Some(Held { place, runs: true }) = self.held[slot] else {
                continue;
            };
            if slots.binary_search(&(slot as u32)).is_ok() {
                continue;
            }
            listed.push(entry(slot as u32, place, SETUP_RISEN));
            risen += 1;
        }
        let ghosts = if self.ring || set > 0 {
            4 * self.fitted.cells
        } else {
            0
        };
        self.ring = false;
        if listed.is_empty() && ghosts == 0 {
            return;
        }
        let solids = u64::from(self.solids) * SETUP_SOLID_ENTRIES as u64 * 16;
        queue.write_buffer(&self.setup.list, solids, bytemuck::cast_slice(&listed));
        let [first_x, first_z] = self.fitted.first;
        let frame = SimSetup {
            counts: [listed.len() as u32, 0, ghosts, self.solids],
            square: [
                first_x as u32,
                first_z as u32,
                self.fitted.cells,
                self.fitted.split,
            ],
            meters: [cell, self.fitted.datum, 0.0, 0.0],
            ring: [self.buffers.ring_at(0) as u32, 0, 0, 0],
            tiles: self.tiles,
        };
        queue.write_buffer(&self.setup.frame, 0, bytemuck::bytes_of(&frame));
        self.listed = Some(Listed {
            slots,
            whole,
            blocks: listed.len() as u32,
            ghosts,
            recorded: Cell::new(false),
        });
    }

    fn write_table(&self, queue: &engine_render::gpu_calls::Queue) {
        let side = self.fitted.blocks();
        let corner = self
            .fitted
            .first
            .map(|first| first.div_euclid(i64::from(BLOCK_SIDE)));
        let mut stored = self.stored.clone();
        for z in 0..side {
            for x in 0..side {
                let at = (z * side + x) as usize;
                let slot = stored.table[at];
                let place = [corner[0] + i64::from(x), corner[1] + i64::from(z)];
                let runs = self.held[slot as usize] == Some(Held { place, runs: true });
                if !runs {
                    stored.table[at] = slot | SIM_STANDING;
                    stored.places[slot as usize] = NO_BLOCK;
                }
            }
        }
        self.buffers.write_table(queue, 0, &stored);
    }
}

fn entry(slot: u32, place: [i64; 2], what: u32) -> [u32; 4] {
    let side = i64::from(BLOCK_SIDE);
    [
        slot,
        (place[0] * side) as u32,
        (place[1] * side) as u32,
        what,
    ]
}

fn live_words(window: &WaterWindow) -> ([u32; 4], Vec<u32>) {
    let Some(read) = window.reads.first() else {
        return ([0; 4], Vec::new());
    };
    let (mut least, mut most) = ([read.key.x, read.key.z], [read.key.x, read.key.z]);
    for read in window.reads.iter() {
        least = [least[0].min(read.key.x), least[1].min(read.key.z)];
        most = [most[0].max(read.key.x), most[1].max(read.key.z)];
    }
    let [columns, rows] = [0, 1].map(|axis| u32::from(most[axis] - least[axis]) + 1);
    let side = i64::from(BLOCK_SIDE);
    let span = i64::from(window.blocks()) + 2;
    let low = window.first.map(|first| first.div_euclid(side) - 1);
    let mut words = vec![LIVE_NONE; (columns * rows) as usize];
    for read in window.reads.iter() {
        let Some(record) = read.settled.as_ref() else {
            continue;
        };
        let live = &record.live;
        let from = [0, 1].map(|axis| live.first[axis].max(low[axis]));
        let to = [0, 1]
            .map(|axis| (live.first[axis] + i64::from(live.blocks[axis])).min(low[axis] + span));
        let blocks = [0, 1].map(|axis| (to[axis] - from[axis]).max(0) as u32);
        let column = u32::from(read.key.x - least[0]);
        let row = u32::from(read.key.z - least[1]);
        words[(row * columns + column) as usize] = words.len() as u32;
        let head = [
            from[0] as i32 as u32,
            from[1] as i32 as u32,
            blocks[0],
            blocks[1],
        ];
        words.extend_from_slice(&head[..LIVE_HEAD as usize]);
        let table = words.len();
        words.resize(table + (blocks[0] * blocks[1]) as usize, LIVE_NONE);
        for z in 0..blocks[1] {
            for x in 0..blocks[0] {
                let Some(cells) = live.block(from[0] + i64::from(x), from[1] + i64::from(z)) else {
                    continue;
                };
                words[table + (z * blocks[0] + x) as usize] = words.len() as u32;
                for &[surface, along_x, along_z] in cells {
                    let held = if surface.is_nan() {
                        [NO_WATER_SURFACE, 0.0, 0.0]
                    } else if along_x.is_nan() {
                        [surface, LIVE_HELD, 0.0]
                    } else {
                        [surface, along_x, along_z]
                    };
                    words.extend_from_slice(&held.map(f32::to_bits)[..LIVE_CELL as usize]);
                }
            }
        }
    }
    (
        [u32::from(least[0]), u32::from(least[1]), columns, rows],
        words,
    )
}

fn entries(window: &WaterWindow, slots: usize) -> usize {
    SETUP_SOLID_ENTRIES * window.solids.len() + slots
}

fn changed(before: &[WaterRead], reads: &[WaterRead]) -> Vec<[Vec2; 2]> {
    reads
        .iter()
        .filter(|read| {
            before
                .iter()
                .any(|held| held.key == read.key && !held.same_tile(read))
        })
        .map(|read| read.square)
        .collect()
}

fn pool_of(window: &WaterWindow) -> u32 {
    window.blocks()
}

fn wrap_of(window: &WaterWindow, pool: u32) -> [u32; 2] {
    window.first.map(|first| {
        first
            .div_euclid(i64::from(BLOCK_SIDE))
            .rem_euclid(i64::from(pool)) as u32
    })
}

fn wrapped(window: &WaterWindow, pool: u32) -> (StoredBlocks, [u32; 2]) {
    let side = window.blocks();
    let wrap = wrap_of(window, pool);
    let mut table = vec![0; (side * side) as usize];
    let mut places = vec![0; (pool * pool) as usize];
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

fn fit(device: &wgpu::Device, window: &Arc<WaterWindow>) -> Option<Arc<WaterWindow>> {
    let limits = device.limits();
    let binding = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    let held = (window.sources.len(), window.solids.len());
    let fits = |blocks: u32| largest_bytes(blocks, held) <= binding;
    let side = window.blocks();
    if fits(side) {
        return Some(window.clone());
    }
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

fn largest_bytes(blocks: u32, (sources, solids): (usize, usize)) -> u64 {
    let cells = blocks * BLOCK_SIDE;
    let capacity = Capacity {
        blocks: blocks * blocks,
        mapped: blocks * blocks,
        edges: 2 * cells,
        sources: sources.max(1) as u32,
    };
    let inputs = (u64::from(capacity.sources) + 8 * u64::from(cells)) * 16;
    let list = (SETUP_SOLID_ENTRIES * solids) as u64 + u64::from(capacity.blocks);
    live_largest_bytes(capacity)
        .max(inputs)
        .max((list * 16).next_power_of_two())
}

fn write_sources(
    queue: &engine_render::gpu_calls::Queue,
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
    sources.len() as u32
}

fn write_solids(
    queue: &engine_render::gpu_calls::Queue,
    list: &wgpu::Buffer,
    solids: &[Footprint],
) -> u32 {
    let mut entries: Vec<[f32; 4]> = Vec::with_capacity(SETUP_SOLID_ENTRIES * solids.len());
    for solid in solids {
        let axes = Mat3::from_cols_array(&solid.axes);
        if axes.determinant() == 0.0 {
            continue;
        }
        let inverse = axes.inverse();
        let reach = solid.reach();
        let [x, y, z] = solid.center;
        entries.extend([
            [x, y, z, if solid.round { 1.0 } else { 0.0 }],
            inverse.x_axis.extend(reach.x).to_array(),
            inverse.y_axis.extend(reach.y).to_array(),
            inverse.z_axis.extend(0.0).to_array(),
        ]);
    }
    if !entries.is_empty() {
        queue.write_buffer(list, 0, bytemuck::cast_slice(&entries));
    }
    (entries.len() / SETUP_SOLID_ENTRIES) as u32
}

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

pub(crate) struct SimDrawn<'a> {
    pub(crate) window: &'a WaterWindow,
    pub(crate) wrap: [u32; 2],
    pub(crate) pool: u32,
    pub(crate) looks: &'a wgpu::Buffer,
}
