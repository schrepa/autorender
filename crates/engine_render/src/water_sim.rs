//! Water running live (`water_sim.wgsl`): rivers running over a terrain's
//! [`WaterWindow`] by the shallow-water equations, in cells finer than the
//! settled water's, the same water as the settled water around it,
//! starting from the water as it settled there, which also stands just past
//! its edges and which its rim is pulled back to, stirred by its own
//! turbulence's bursts and pushed by the solids standing in it. Water's
//! pass draws it as one with the tiles' water, reading each cell's look
//! (`meshes/water.wgsl`'s `water_live_read`). The window is one slot of the
//! scheme's buffers ([`crate::water_kernels`]) whose pool holds its blocks,
//! each block in the slot of its place on the terrain's lattice of blocks
//! taken around the pool's side, with a ring past its edges. Every frame
//! steps every cell by the same seconds, as many steps as the simulation
//! time passed holds, at most those of a frame LONGEST_FRAME long
//! ([`live_steps`]), within a fixed number of dispatches, and then finds
//! each cell's look. As the window follows the detail eye a block keeps
//! its slot, and the water there runs on; each block it newly covers takes
//! the slot of a block it left. Before each step turbulence bursts in its
//! water, each burst drawn by the one rule for bursts (`water_bursts.wgsl`)
//! from its cell and the step's index, and pushes it; then the solids
//! standing in its water push it toward their own velocity and churn its
//! foam, which rides its surface, dragged by the scene's wind. Where one of
//! the device's storage bindings cannot hold a buffer of the window, its
//! middle square whose buffers they hold runs instead. Nothing is read
//! back.
//!
//! The GPU sets the window up from what it holds already
//! (`meshes/water_setup.wgsl`, through the terrain group of the terrain the
//! window stands on), in the frame a block comes into it or the tiles under
//! the block change (a finer level, or its settled water, arriving), before
//! any step: one dispatch, a workgroup a block, sets up each such block's
//! ground from the finest resident heights, raised by the relief of the
//! ground's materials at the level whose texels span a live cell, so its
//! fast water breaks over the stones and cobbles the ground draws; the
//! lakes holding its cells, which let go of those that relief raises to
//! their level; its bed's roughness and what its bed holds, from the tiles'
//! water; the solids' cover; its water to start from, the settled water's
//! smooth surface at each cell matched over each settled cell to that cell's
//! water, off that steady state as unsteadily as the water runs there; and
//! that steady state, which its rim is pulled back to. A second dispatch,
//! an invocation a cell, sets up the cells past its edges and what they
//! bring in, in each frame the window changed. So a window runs in the
//! frame it comes, reading the coarser level shading reads where the
//! tiles' settled water is not resident yet; and every frame RISEN slots'
//! ground rises again, round the pool, so the bed follows the ground's
//! materials as they arrive and the tiles under it fade in. The CPU hands
//! over only where the window stands, the blocks a frame sets up, the
//! solids and where its rivers bring water.
//!
//! The cook's meter cells over a meter's samples run live in half-meter
//! cells (`engine_core`'s `live_cell`), each step 1/225 of a second
//! ([`step_seconds`]), so a frame at 60 Hz steps about four times and one
//! LONGEST_FRAME long eight, each step thirteen dispatches at most, 107 a
//! frame with the looks' and the setup's two. On a view 1080 pixels tall
//! through 55 degrees,
//! their live reach of 130 meters asks for a window of 616 cells along each
//! side (with three quarters of a 32-meter snap past it), and the default
//! budget of 2,097,152 cell steps over the longest frame's eight holds 512
//! (256 meters, reaching 104 meters or more from the eye), 262,144 cells in
//! a pool of 64 × 64 blocks, about a million cell steps a frame at 60 Hz;
//! on one 1440 pixels tall the reach of 173 meters asks for 792 cells, and
//! the budget holds the same 512. Its buffers take about 133 bytes a cell
//! of its pool: about 35 MB. A window coming anew sets up all 4,096 blocks
//! in its first frame; a move of a snap along both axes sets up the 960
//! blocks it uncovers (61,440 cells and 77,760 corners, each corner a
//! height and its materials' relief, each cell a smooth surface through
//! 4 × 4 samples, its settled cell's four, and the solids reaching it at
//! 16 points each) and the 2,048 cells past its edges, two dispatches; and
//! every frame RISEN blocks' ground rises again (16,384 cells and 20,736
//! corners). The CPU writes 16 bytes a block set up and 64 a solid.

use std::cell::Cell;
use std::sync::Arc;

use bytemuck::Zeroable;
use engine_core::glam::{Mat3, Vec2, Vec3};
use engine_core::terrain_data::shallow_water::step_seconds;
use engine_core::terrain_data::{BLOCK_SIDE, Footprint, StoredBlocks};
use engine_core::{SIM_TOUCHES, WaterObstacle, WaterRead, WaterWindow, Wind, live_steps};

use crate::bindings::{EMPTY, WATER_SETUP};
use crate::gpu_layouts::constants::{
    SETUP_SOLID_ENTRIES, SIM_LIVE, SIM_STEPPING, WATER_SETUP_RING_GROUP,
};
use crate::gpu_layouts::water::{SimSetup, SimSlot, SimTouch, SimTouches};
use crate::pipelines::GroupLayout;
use crate::shaders::entry::meshes;
use crate::water_kernels::{
    Capacity, Kernels, SlotBuffers, encode_drawn, encode_steps, grid, live_largest_bytes,
};

// The longest frame whose time the live water steps through, whose steps
// a longer frame steps no more of, nor water's particles.
pub(crate) use engine_core::LONGEST_FRAME;

/// Slots whose ground a frame sets up again, round the pool, from the tiles
/// under them as they stand then, so the live water's bed follows the
/// ground's materials as they arrive and the tiles under it fade in: a pool
/// of 64 × 64 slots within 16 frames.
const RISEN: usize = 256;

/// A terrain's live window as frames hand it over: the window, the solids
/// standing in it (nearest the eye first) and how many more stand there,
/// and the terrain group its setup reads the ground and the tiles' water
/// through, once made.
pub(crate) struct Live<'a> {
    pub(crate) window: &'a Arc<WaterWindow>,
    pub(crate) touches: &'a [WaterObstacle],
    pub(crate) beyond: u32,
    pub(crate) ground: Option<LiveGround<'a>>,
}

/// The terrain group through which the live water's setup reads the ground,
/// its materials and the tiles' water (`meshes/water_setup.wgsl`): its
/// layout, the group as the subdivisions' passes bind it, and the offset of
/// the uniform of the terrain the window stands on.
pub(crate) struct LiveGround<'a> {
    pub(crate) layout: &'a GroupLayout,
    pub(crate) group: &'a wgpu::BindGroup,
    pub(crate) uniform: u32,
}

/// What sets the live water up on the GPU (`meshes/water_setup.wgsl`): the
/// empty group its pipelines bind first and that group's layout, its own
/// group's layout, the terrain group's layout once a frame brought it and,
/// once the mesh module came, its pipelines, setting up blocks and the cells
/// past the window's edges; and this frame's terrain group with the offset
/// of the uniform of the terrain the window stands on.
struct Setup {
    empty_layout: GroupLayout,
    empty: wgpu::BindGroup,
    layout: GroupLayout,
    terrain: Option<GroupLayout>,
    pipelines: Option<[wgpu::ComputePipeline; 2]>,
    ground: Option<(wgpu::BindGroup, u32)>,
}

impl Setup {
    fn new(device: &wgpu::Device) -> Self {
        let empty_layout = crate::pipelines::layout(device, &EMPTY);
        let empty = crate::pipelines::group(
            device,
            "engine/water-setup-empty",
            &EMPTY,
            &empty_layout,
            &[],
        );
        Self {
            empty_layout,
            empty,
            layout: crate::pipelines::layout(device, &WATER_SETUP),
            terrain: None,
            pipelines: None,
            ground: None,
        }
    }

    /// Keeps `ground`, the terrain group this frame's setup reads, and its
    /// layout the first time one comes; none where it does not.
    fn hold(&mut self, ground: Option<&LiveGround<'_>>) {
        self.ground = ground.map(|ground| {
            if self.terrain.is_none() {
                self.terrain = Some(ground.layout.clone());
            }
            (ground.group.clone(), ground.uniform)
        });
    }

    /// Creates its pipelines from the mesh module `shader` unless they
    /// exist, once a frame brought the terrain group's layout.
    fn ensure(&mut self, device: &wgpu::Device, shader: &wgpu::ShaderModule) {
        if self.pipelines.is_some() {
            return;
        }
        let Some(terrain) = self.terrain.as_ref() else {
            return;
        };
        let layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::WATER_SETUP_PIPELINES,
            &[&self.empty_layout, terrain, &self.layout],
        );
        let entry = |entry| crate::pipelines::compute_entry(device, &layout, shader, entry);
        self.pipelines = Some([entry(meshes::setup_blocks), entry(meshes::setup_ring)]);
    }

    /// Whether it sets up this frame: its pipelines made and the terrain
    /// group there.
    fn sets_up(&self) -> bool {
        self.pipelines.is_some() && self.ground.is_some()
    }

    /// Records into `pass` the setup of `blocks` blocks a window's group
    /// `group` lists, a workgroup each, then of `ghosts` cells past its
    /// edges, through the terrain group at the uniform of the terrain the
    /// window stands on; whether it recorded them.
    fn encode(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        group: &wgpu::BindGroup,
        (blocks, ghosts): (u32, u32),
    ) -> bool {
        let (Some([set_blocks, set_ring]), Some((terrain, uniform))) =
            (&self.pipelines, &self.ground)
        else {
            return false;
        };
        // Both pipelines bind the same groups: the terrain's uniform at its
        // offset, the views' at their first, which they read none of.
        pass.set_bind_group(0, &self.empty, &[]);
        pass.set_bind_group(1, terrain, &[0, *uniform]);
        pass.set_bind_group(2, group, &[]);
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

/// What sets up a window's blocks and the cells past its edges: its
/// uniform; its list, the solids standing in the settled water over the
/// window first (SETUP_SOLID_ENTRIES entries each), then the blocks a frame
/// sets up (each one's slot and the cell of the terrain's lattice of live
/// cells its first corner stands at), with the entries it holds room for;
/// and the group binding them with the window's buffers.
struct SetupBuffers {
    frame: wgpu::Buffer,
    list: wgpu::Buffer,
    room: usize,
    group: wgpu::BindGroup,
}

impl SetupBuffers {
    /// Room for `entries` entries of the list, the next power of two of
    /// them, and the group of the layout `layout` binding them with
    /// `buffers`.
    fn new(
        device: &wgpu::Device,
        layout: &GroupLayout,
        buffers: &SlotBuffers,
        entries: usize,
    ) -> Self {
        let room = entries.max(1).next_power_of_two();
        let frame = crate::buffers::uniform(
            device,
            "engine/water-setup",
            size_of::<SimSetup>() as u64,
        );
        let list = crate::buffers::storage(
            device,
            "engine/water-setup-list",
            room as u64 * 16,
            wgpu::BufferUsages::empty(),
        );
        let group = crate::pipelines::group(
            device,
            "engine/water-setup",
            &WATER_SETUP,
            layout,
            &[
                crate::buffers::entry("setup_frame", &frame),
                crate::buffers::entry("setup_list", &list),
                crate::buffers::entry("setup_ground", &buffers.ground),
                crate::buffers::entry("setup_cells", &buffers.cells),
                crate::buffers::entry("setup_looks", &buffers.averages),
                crate::buffers::entry("setup_ring", &buffers.inputs),
            ],
        );
        Self {
            frame,
            list,
            room,
            group,
        }
    }

    /// Bytes of its buffers.
    fn bytes(&self) -> u64 {
        self.frame.size() + self.list.size()
    }
}

/// A terrain's water running live over its window: the scheme's passes,
/// what sets it up, the window's buffers, the steps this frame runs and
/// those the simulation's time held past them, the step of the simulation's
/// time the last frame stepped to, a count advancing whenever what water
/// draws from changes, and the solids pushing its water this frame with
/// those standing in its window past the most that do.
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

/// A window as terrains placed it and the square of it that runs (all of
/// it, where the device's bindings hold its buffers); the slots along each
/// side of the pool, the square's blocks (each in the slot of its place on
/// the terrain's lattice of blocks taken around the pool's side) and the
/// slots they turn by along x and z; its buffers (one slot with its ring),
/// what sets them up, the count of cells rivers bring water to and of the
/// solids its setup covers cells with; the block each slot holds as it was
/// set up (its place on the terrain's lattice of blocks; none where it
/// holds none set up on the square's datum and settled cells), and the
/// slot the next frame's ground rises again from; the squares of the
/// level-0 tiles set up from another tile than the window before was
/// (terrain-local least and greatest corners), and whether the cells past
/// its edges are to be set up; whether every block of the square is set
/// up, so it runs; and what this frame sets up.
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
    held: Vec<Option<[i64; 2]>>,
    sweep: usize,
    changed: Vec<[Vec2; 2]>,
    ring: bool,
    ready: bool,
    listed: Option<Listed>,
}

/// What a frame sets up: the slots of the blocks it sets up whole, the
/// blocks it sets up (those whole, and those whose ground alone rises
/// again), the cells past the edges it sets up (none where it sets up none)
/// and whether the frame's encoding recorded it.
struct Listed {
    whole: Vec<u32>,
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

    /// Prepares the live window `live` for this frame at simulation time
    /// `seconds` in the scene's wind `wind`, over the square of it the
    /// device's bindings hold: its buffers anew where a square of another
    /// size came; where one of its size moved, its table and sources; the
    /// terrain group its setup reads the ground and the tiles' water
    /// through; the setup of the blocks its slots do not hold, of those whose
    /// tiles changed and of the cells past its edges where it changed, and
    /// the ground of RISEN slots rising again, wherever the setup can run;
    /// the solids pushing its water; the wind dragging its surface; and as
    /// many steps, each of the same seconds, as the simulation time since
    /// the last frame holds, at most LONGEST_FRAME's, each of its index in
    /// the simulation's time, which its bursts are drawn by. Drops it where
    /// there is no window, or no square of it fits.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        live: Option<Live<'_>>,
        (wind, seconds): (Wind, f64),
    ) {
        if let Some(current) = self.window.as_mut() {
            current.settle();
        }
        // The window, with the square of it that runs where it changed.
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
        current.set_up(queue, sets_up);
        if current.ready != ready {
            self.generation += 1;
        }
        // The steps: as many of a step's seconds as the simulation's time
        // has passed through since the last frame, at most LONGEST_FRAME's,
        // those past them falling behind the clock, the first of them the
        // step after the last frame's.
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
        // The wind's mean drags the surface: its gusts come and go within
        // seconds (`Wind::at`), carrying what floats back and forth by a few
        // centimeters in a breeze. Bursts are drawn by the cell on the
        // terrain's lattice of the square's cells, an `i32`'s bits each, and
        // the step's index.
        let [x, z] = current.fitted.first;
        current
            .buffers
            .write_frames(queue, wind.velocity, [x as u32, z as u32, starting as u32]);
        current.buffers.write_squares(
            queue,
            &[SimSlot {
                shape: [cells, cells, SIM_STEPPING | SIM_LIVE, current.sourced],
                step: [cell, step, 0.0, 0.0],
                blocks: [columns, rows, pooled, 0],
            }],
        );
        current.buffers.write_touches(queue, pushing.as_ref());
    }

    /// Sets up the blocks and the cells past the edges this frame lists,
    /// then, once every block of its square is set up, runs the steps
    /// prepared and finds each cell's look, timed by `profiler`
    /// (`water_sim`).
    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
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
                (self.steps, window.sourced, false, false),
                blocks,
            );
        }
        encode_drawn(&mut pass, pipelines, &window.buffers, blocks);
    }

    /// What water draws the live water from, once every block of its square
    /// is set up: the square of the window that runs, the slots its blocks
    /// turn by, its pool's slots along each side and its cells' looks; and a
    /// count advancing whenever they change.
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

    /// The cells along a side of the square that runs, those the budget of
    /// its cell steps cut from the square its reach asks for, and those the
    /// device's bindings cut from the window; the steps this frame runs and
    /// those the simulation's time held past them; and the blocks this frame
    /// sets up whole.
    pub(crate) fn cells(&self) -> [u32; 6] {
        self.window
            .as_ref()
            .filter(|window| window.ready)
            .map_or([0; 6], |window| {
                [
                    window.fitted.cells,
                    window.source.asked.saturating_sub(window.source.cells),
                    window.source.cells - window.fitted.cells,
                    self.steps,
                    self.behind,
                    window
                        .listed
                        .as_ref()
                        .map_or(0, |listed| listed.whole.len() as u32),
                ]
            })
    }

    /// The solids pushing its water this frame, and those standing in its
    /// window past the most that do, which push none.
    pub(crate) fn touches(&self) -> [u32; 2] {
        self.touches
    }

    /// Creates the passes setting the live water up from the mesh module
    /// `shader` unless they exist, once a frame brought the terrain group
    /// its setup reads the ground and the tiles' water through.
    pub(crate) fn ensure_setup(&mut self, device: &wgpu::Device, shader: &wgpu::ShaderModule) {
        self.setup.ensure(device, shader);
    }

    /// Bytes of the window's buffers.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.window
            .as_ref()
            .map_or(0, |window| window.buffers.bytes() + window.setup.bytes())
    }
}

impl SimWindow {
    /// The buffers `fitted`, the square of `source` that runs, runs in, its
    /// pool holding its blocks, with its table and sources, of the scheme's
    /// group layout `layout`, and what sets them up, of the layout `setup`;
    /// every block of it, and the cells past its edges, are set up with the
    /// frame.
    fn new(
        (device, queue, layouts): (&wgpu::Device, &wgpu::Queue, (&GroupLayout, &GroupLayout)),
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
        let setup = SetupBuffers::new(device, layouts.1, &buffers, entries(&fitted, slots));
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
            held: vec![None; slots],
            sweep: 0,
            changed: Vec::new(),
            ring: true,
            ready: false,
            listed: None,
        };
        window.write_square((device, queue, layouts));
        window
    }

    /// Takes `fitted`, the square of `source` that runs, of its square's
    /// size, in its square's place: its table turned to its place, its
    /// sources and solids, the squares of the tiles set up from another tile
    /// than the window before was, and the cells past its edges to set up
    /// again. Its slots keep the blocks both squares hold, unless the datum
    /// or the settled cells changed, and the frame sets up the rest.
    fn follow(
        &mut self,
        gpu: (&wgpu::Device, &wgpu::Queue, (&GroupLayout, &GroupLayout)),
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

    /// Writes its square's table, its rivers' sources and the solids its
    /// setup covers cells with, its inputs grown first where they hold fewer
    /// sources and its setup's list where it holds fewer solids, and has the
    /// cells past its edges set up.
    fn write_square(
        &mut self,
        (device, queue, (layout, setup)): (
            &wgpu::Device,
            &wgpu::Queue,
            (&GroupLayout, &GroupLayout),
        ),
    ) {
        let grown = self
            .buffers
            .hold_sources(device, layout, self.fitted.sources.len());
        let entries = entries(&self.fitted, self.held.len());
        if grown || entries > self.setup.room {
            self.setup = SetupBuffers::new(
                device,
                setup,
                &self.buffers,
                entries.max(self.setup.room),
            );
        }
        self.buffers.write_table(queue, 0, &self.stored);
        self.sourced = write_sources(queue, &self.buffers, &self.fitted, &self.stored);
        self.solids = write_solids(queue, &self.setup.list, &self.fitted.solids);
        self.ring = true;
    }

    /// Lets go of what the frame before set up: done where its encoding
    /// recorded it, else the slots it set up whole hold no block set up,
    /// and the cells past its edges, where it set them up, are set up again.
    fn settle(&mut self) {
        let Some(listed) = self.listed.take() else {
            return;
        };
        if listed.recorded.get() {
            return;
        }
        for &slot in &listed.whole {
            self.held[slot as usize] = None;
        }
        if listed.ghosts > 0 {
            self.ring = true;
        }
    }

    /// Lists the frame's setup where it can run (`sets_up`): whole, each
    /// block of its square its slot does not hold as set up, and each over
    /// whose tiles another tile stands than the window before set it up
    /// from, a cell past it reaching into one; its ground alone, RISEN of
    /// the other slots holding blocks round the pool from where the frame
    /// before stopped; and the cells past its edges where the window
    /// changed. Each is listed by its slot and the cell of the terrain's
    /// lattice of live cells its first corner stands at, the blocks whole
    /// first, after the solids in its setup's list. The square runs once
    /// every block of it is set up; where the setup cannot run, the blocks
    /// it would set up wait for a frame where it can, and the square runs
    /// not.
    fn set_up(&mut self, queue: &wgpu::Queue, sets_up: bool) {
        let side = self.fitted.blocks();
        let corner = self
            .fitted
            .first
            .map(|first| first.div_euclid(i64::from(BLOCK_SIDE)));
        let cell = self.fitted.cell;
        let block = cell * BLOCK_SIDE as f32;
        let changed = std::mem::take(&mut self.changed);
        // Whether another tile stands under block `place` than it was set up
        // from: a cell past it reaches into such a level-0 tile.
        let touched = |place: [i64; 2]| {
            let low = Vec2::new(place[0] as f32, place[1] as f32) * block - Vec2::splat(cell);
            let high = low + Vec2::splat(block + 2.0 * cell);
            changed
                .iter()
                .any(|[least, most]| high.cmpge(*least).all() && low.cmple(*most).all())
        };
        let mut listed: Vec<[u32; 4]> = Vec::new();
        let mut whole: Vec<u32> = Vec::new();
        let mut waiting = false;
        for z in 0..side {
            for x in 0..side {
                let slot = self.stored.table[(z * side + x) as usize];
                let place = [corner[0] + i64::from(x), corner[1] + i64::from(z)];
                let held = &mut self.held[slot as usize];
                if *held == Some(place) && !touched(place) {
                    continue;
                }
                if !sets_up {
                    *held = None;
                    waiting = true;
                    continue;
                }
                *held = Some(place);
                whole.push(slot);
                listed.push(entry(slot, place));
            }
        }
        self.ready = !waiting;
        if !sets_up {
            return;
        }
        // Then as many of the other slots holding blocks as RISEN, round the
        // pool from where the frame before stopped, their ground rising
        // again.
        let wholes = whole.len();
        whole.sort_unstable();
        let slots = self.held.len();
        let mut risen = 0;
        for _ in 0..slots {
            if risen == RISEN || wholes == slots {
                break;
            }
            let slot = self.sweep;
            self.sweep = (slot + 1) % slots;
            let Some(place) = self.held[slot] else {
                continue;
            };
            if whole.binary_search(&(slot as u32)).is_ok() {
                continue;
            }
            listed.push(entry(slot as u32, place));
            risen += 1;
        }
        let ghosts = if self.ring { 4 * self.fitted.cells } else { 0 };
        self.ring = false;
        if listed.is_empty() && ghosts == 0 {
            return;
        }
        let solids = u64::from(self.solids) * SETUP_SOLID_ENTRIES as u64 * 16;
        queue.write_buffer(&self.setup.list, solids, bytemuck::cast_slice(&listed));
        // Its first cell on the terrain's lattice of live cells, an `i32`'s
        // bits each.
        let [first_x, first_z] = self.fitted.first;
        let frame = SimSetup {
            counts: [listed.len() as u32, wholes as u32, ghosts, self.solids],
            square: [
                first_x as u32,
                first_z as u32,
                self.fitted.cells,
                self.fitted.split,
            ],
            meters: [cell, self.fitted.datum, 0.0, 0.0],
            ring: [self.buffers.ring_at(0) as u32, 0, 0, 0],
        };
        queue.write_buffer(&self.setup.frame, 0, bytemuck::bytes_of(&frame));
        self.listed = Some(Listed {
            whole,
            blocks: listed.len() as u32,
            ghosts,
            recorded: Cell::new(false),
        });
    }
}

/// The entry of the setup's list of the block at `place` on the terrain's
/// lattice of blocks in slot `slot`: the slot, and the cell of the
/// terrain's lattice of live cells its first corner stands at, an `i32`'s
/// bits along x and z.
fn entry(slot: u32, place: [i64; 2]) -> [u32; 4] {
    let side = i64::from(BLOCK_SIDE);
    [slot, (place[0] * side) as u32, (place[1] * side) as u32, 0]
}

/// Entries the setup's list of the live square `window` holds at most, in
/// a pool of `slots` slots: those of its solids, and one for each slot.
fn entries(window: &WaterWindow, slots: usize) -> usize {
    SETUP_SOLID_ENTRIES * window.solids.len() + slots
}

/// The squares of the level-0 tiles `reads` sets the window up from that
/// another tile than `before` set it up from stands over (terrain-local
/// least and greatest corners): those both read. A tile one of them reads
/// alone stands under no block both windows hold.
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

/// Slots along each side of the pool of the live square `window`: its
/// blocks', each block of the square it moves to taking the slot of the
/// block it leaves.
fn pool_of(window: &WaterWindow) -> u32 {
    window.blocks()
}

/// The slots `window`'s blocks turn by along x and z in a pool of `pool`
/// slots along each side: its first block's place on the terrain's lattice
/// of blocks, taken around the pool's side.
fn wrap_of(window: &WaterWindow, pool: u32) -> [u32; 2] {
    window.first.map(|first| {
        first
            .div_euclid(i64::from(BLOCK_SIDE))
            .rem_euclid(i64::from(pool)) as u32
    })
}

/// The blocks of `window`'s square as its slot stores them in a pool of
/// `pool` slots along each side, and the slots they turn by along x and z:
/// every one, each in the slot of its place on the terrain's lattice of
/// blocks taken around the pool's side, so a block keeps its slot as the
/// window follows the eye, and the square it moves to takes its new blocks
/// into the slots of those it leaves.
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

/// The square of `window` whose every buffer one of `device`'s storage
/// bindings holds: the window itself where they hold its own, else its
/// middle square of the most blocks along each side whose buffers they
/// hold; none where they hold not even one.
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

/// The most bytes one buffer of a live square `blocks` blocks along each
/// side takes, its pool holding as many, its rivers bringing water to
/// `sources` of its cells and `solids` standing in the settled water over
/// it: its slot's largest, its inputs' (an entry for each source and two
/// for each of the `4 × cells` cells of its ring) or its setup's list's.
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

/// Writes what `window`'s rivers bring (at the cells `stored`, its blocks,
/// holds them at) into `buffers`' inputs; the cells its rivers bring water
/// to.
fn write_sources(
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
    sources.len() as u32
}

/// Writes `solids`, those standing in the settled water over a window
/// (terrain-local), first into its setup's list `list`, as the setup reads
/// them: each one's center and 1 where it is a ball, else 0, then the
/// inverse of its axes a column each, the first two beside how far it
/// reaches along x and z. A flat solid, whose axes have no inverse, covers
/// nothing and is left out. The solids written.
fn write_solids(queue: &wgpu::Queue, list: &wgpu::Buffer, solids: &[Footprint]) -> u32 {
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

/// What water draws the live water from ([`WaterSim::drawn`]).
pub(crate) struct SimDrawn<'a> {
    pub(crate) window: &'a WaterWindow,
    pub(crate) wrap: [u32; 2],
    pub(crate) pool: u32,
    pub(crate) looks: &'a wgpu::Buffer,
}
