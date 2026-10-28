//! Water running live (`water_sim.wgsl`): rivers running over a terrain's
//! [`WaterWindow`] by the shallow-water equations, the same water as the
//! settled water around it in the same model: on the cells settling ran on,
//! finer than the settled water's, over the same bed, stirred by the same
//! bursts of turbulence. Each block of it starts from the water exactly as
//! it settled there, on those cells, as the settled water's records keep it
//! (`engine_core::terrain_data::LiveWater`), which also stands just past its
//! edges and which its rim is pulled back to, and the solids standing in it
//! push it. Water's pass draws it as one with the tiles' water, reading each
//! cell's look (`meshes/water.wgsl`'s `water_live_read`). The window is one
//! slot of the scheme's buffers ([`crate::water_kernels`]) whose pool holds
//! its blocks, each block in the slot of its place on the terrain's lattice
//! of blocks taken around the pool's side, with a ring past its edges.
//! Every frame steps every cell whose block runs by the same seconds, as
//! many steps as the simulation time passed holds, at most those of a frame
//! LONGEST_FRAME long ([`live_steps`]), within a fixed number of dispatches,
//! and then finds each cell's look. As the window follows the detail eye a
//! block keeps its slot, and the water there runs on; each block it newly
//! covers takes the slot of a block it left. Before each step turbulence
//! bursts in its water, each burst drawn by the one rule for bursts
//! (`water_bursts.wgsl`) from its cell and the step's index, and pushes it;
//! then the solids standing in its water push it toward their own velocity.
//! Where one of the device's storage bindings cannot hold a buffer of the
//! window, its middle square whose buffers they hold runs instead. Nothing
//! is read back.
//!
//! A block runs once its data is resident: the level-0 tiles under it and a
//! cell past it on every side, holding their settled water, on the GPU, and
//! those tiles' records of settled water with it. The CPU lays out the live
//! cells of the records of the level-0 tiles the window reads over its
//! blocks and a block past them on every side (`live_words`) and hands
//! them over as the window comes, moves or reads other records; the GPU
//! sets every block up from them and from what it holds already
//! (`meshes/water_setup.wgsl`, through the terrain group of the terrain the
//! window stands on), before any step, in one dispatch, a workgroup a block.
//! A block whose data came to be resident is set up whole in that frame,
//! once: its ground from level 0's heights, raised by the relief of the
//! ground's materials at the level whose texels span a live cell as settling
//! raises its own (`terrain_ground_rise`), so its fast water breaks over the
//! stones and cobbles the ground draws; the lakes holding its cells as the
//! records hold them, which let go of those that relief raises to their
//! level; its bed's roughness by settling's rule, and what its bed holds,
//! from the tiles' water; the solids' cover; its water to start from, each
//! cell's as its record holds it, off that steady state as unsteadily as
//! the water runs there; and that steady state, which its rim is pulled
//! back to. A block still waiting for its data stands as the settled water
//! does there, as the window comes and again as what it reads changes: its
//! water does not step, the solve takes it as it takes the cells past the
//! square's edges, the live water beside it pulled back to the settled
//! water as at the rim, and water's pass draws the tiles' water over it. A
//! second dispatch, an invocation a cell, sets up the cells past the
//! square's edges and what they bring in, from the same records, in each
//! frame the window changed or a block was set up. And every frame RISEN
//! running slots' ground rises again, round the pool, so the bed follows
//! the ground's materials as their maps arrive after their tiles. The CPU
//! hands over only where the window stands, the blocks a frame sets up, the
//! records' live cells, the solids and where its rivers bring water.
//!
//! The cook's meter cells over a meter's samples run live in half-meter
//! cells (`engine_core`'s `live_cell`), each step 1/225 of a second
//! ([`step_seconds`]), so a frame at 60 Hz steps about four times and one
//! LONGEST_FRAME long eight, each step eleven dispatches at most, 91 a frame
//! with the looks' and the setup's two. On a view 1080 pixels tall through
//! 55 degrees,
//! their live reach of 130 meters asks for a window of 616 cells along each
//! side (with three quarters of a 32-meter snap past it), and the default
//! budget of 2,097,152 cell steps over the longest frame's eight holds 512
//! (256 meters, reaching 104 meters or more from the eye), 262,144 cells in
//! a pool of 64 × 64 blocks, about a million cell steps a frame at 60 Hz;
//! on one 1440 pixels tall the reach of 173 meters asks for 792 cells, and
//! the budget holds the same 512. Its buffers take about 133 bytes a cell
//! of its pool: about 35 MB. The records' live cells over it take 772 bytes
//! a block holding a wet cell (its cells' 768 and its table's word) and 4 a
//! block holding none: at most 3.4 MB over its 66 × 66 blocks, about 0.35
//! MB where rivers wet a tenth of them. A window coming anew sets up all
//! 4,096 blocks in the first frame their data is resident; a move of a snap
//! along both axes sets up the 960 blocks it uncovers (61,440 cells and
//! 77,760 corners, each corner a height and its materials' relief, each cell
//! its record's water, the tiles' water's sample nearest it and the solids
//! reaching it at 16 points each) and the 2,048 cells past its edges, two
//! dispatches; and every frame RISEN blocks' ground rises again (16,384
//! cells and 20,736 corners). The CPU writes 16 bytes a block set up and 64
//! a solid.

use std::cell::Cell;
use std::sync::Arc;

use bytemuck::Zeroable;
use engine_core::collections::HashMap;
use engine_core::glam::{Mat3, Vec2, Vec3};
use engine_core::terrain_data::shallow_water::step_seconds;
use engine_core::terrain_data::{BLOCK_SIDE, Footprint, NO_BLOCK, StoredBlocks};
use engine_core::{SIM_TOUCHES, Tile, TileKey, WaterObstacle, WaterRead, WaterWindow, live_steps};

use crate::bindings::{EMPTY, WATER_SETUP};
use crate::gpu_layouts::constants::{
    LIVE_CELL, LIVE_HEAD, LIVE_HELD, LIVE_NONE, NO_WATER_SURFACE, SETUP_RISEN, SETUP_SOLID_ENTRIES,
    SETUP_STANDS, SETUP_WHOLE, SIM_BURSTING, SIM_LIVE, SIM_STANDING, SIM_STEPPING,
    WATER_SETUP_RING_GROUP,
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

/// Running slots whose ground a frame sets up again, round the pool, from
/// the tiles under them as they stand then, so the live water's bed follows
/// the ground's materials as their maps arrive after their tiles and the
/// tiles' own materials fade in over the coarser level's
/// (`meshes/terrain.wgsl`'s `terrain_layer_fades`): a pool of 64 × 64 slots
/// within 16 frames.
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
/// the uniform of the terrain the window stands on; with that terrain's id
/// and the tiles the GPU holds, by terrain and key, with their water, which
/// a block's data is resident in.
pub(crate) struct LiveGround<'a> {
    pub(crate) layout: &'a GroupLayout,
    pub(crate) group: &'a wgpu::BindGroup,
    pub(crate) uniform: u32,
    pub(crate) terrain: u64,
    pub(crate) held: &'a HashMap<(u64, TileKey), Arc<Tile>>,
}

impl LiveGround<'_> {
    /// Whether the window's cells over `read` can start from what the GPU
    /// holds there: its level-0 tile, holding its settled water, the very
    /// tile the terrain group holds (`WaterRead::ready`).
    fn resident(&self, read: &WaterRead) -> bool {
        read.ready
            && read.tile.as_ref().is_some_and(|tile| {
                self.held
                    .get(&(self.terrain, read.key))
                    .is_some_and(|held| Arc::ptr_eq(held, tile))
            })
    }
}

/// What sets the water's beds and the live water up on the GPU
/// (`meshes/water_setup.wgsl`): the empty group its pipelines bind first and
/// that group's layout, its own group's layout, the terrain group's layout
/// once a frame brought it and, once the mesh module came, its pipelines,
/// setting up the live water's blocks and the cells past its window's edges
/// and raising settling water's ground; and this frame's terrain group with
/// the offset of the uniform of the terrain the water stands on. The live
/// water and settling ([`crate::water_settle`]) each hold one.
pub(crate) struct Setup {
    empty_layout: GroupLayout,
    empty: wgpu::BindGroup,
    pub(crate) layout: GroupLayout,
    terrain: Option<GroupLayout>,
    pipelines: Option<[wgpu::ComputePipeline; 3]>,
    ground: Option<(wgpu::BindGroup, u32)>,
}

impl Setup {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
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
    pub(crate) fn hold(&mut self, ground: Option<&LiveGround<'_>>) {
        self.ground = ground.map(|ground| {
            if self.terrain.is_none() {
                self.terrain = Some(ground.layout.clone());
            }
            (ground.group.clone(), ground.uniform)
        });
    }

    /// Creates its pipelines from the mesh module `shader` unless they
    /// exist, once a frame brought the terrain group's layout.
    pub(crate) fn ensure(&mut self, device: &wgpu::Device, shader: &wgpu::ShaderModule) {
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
        self.pipelines = Some([
            entry(meshes::setup_blocks),
            entry(meshes::setup_ring),
            entry(meshes::setup_raise),
        ]);
    }

    /// Whether it sets up this frame: its pipelines made and the terrain
    /// group there.
    pub(crate) fn sets_up(&self) -> bool {
        self.pipelines.is_some() && self.ground.is_some()
    }

    /// Binds into `pass` the groups its pipelines read through: the empty
    /// one, the terrain group at the uniform of the terrain the water stands
    /// on and `group`; the pipelines, where it sets up this frame.
    fn bind<'a>(
        &'a self,
        pass: &mut wgpu::ComputePass<'_>,
        group: &wgpu::BindGroup,
    ) -> Option<&'a [wgpu::ComputePipeline; 3]> {
        let (Some(pipelines), Some((terrain, uniform))) = (&self.pipelines, &self.ground) else {
            return None;
        };
        // Its pipelines bind the same groups: the terrain's uniform at its
        // offset, the views' at their first, which they read none of.
        pass.set_bind_group(0, &self.empty, &[]);
        pass.set_bind_group(1, terrain, &[0, *uniform]);
        pass.set_bind_group(2, group, &[]);
        Some(pipelines)
    }

    /// Records into `pass` the raising of `blocks` blocks of settling water's
    /// ground a settling slot's group `group` lists, a workgroup each,
    /// through the terrain group at the uniform of the terrain it settles on;
    /// whether it recorded them.
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

/// What sets up a window's blocks and the cells past its edges, or raises
/// settling water's ground: its uniform; its list (the live water's, the
/// solids standing in the settled water over the window first,
/// SETUP_SOLID_ENTRIES entries each, then the blocks a frame sets up, each
/// one's slot, the cell of the terrain's lattice of live cells its first
/// corner stands at and what it sets up of it; settling water's, its blocks
/// and their samples), with the entries it holds room for; the live cells
/// of the settled water's records the live water starts from
/// (`live_words`; none settling), with the words it holds room for; and
/// the group binding them with the scheme's buffers.
pub(crate) struct SetupBuffers {
    pub(crate) frame: wgpu::Buffer,
    pub(crate) list: wgpu::Buffer,
    pub(crate) room: usize,
    pub(crate) live: wgpu::Buffer,
    pub(crate) live_room: usize,
    pub(crate) group: wgpu::BindGroup,
}

impl SetupBuffers {
    /// Room for `entries` entries of the list and `words` words of the
    /// records' live cells, the next power of two of each, and the group of
    /// the layout `layout` binding them with `buffers`.
    pub(crate) fn new(
        device: &wgpu::Device,
        layout: &GroupLayout,
        buffers: &SlotBuffers,
        (entries, words): (usize, usize),
    ) -> Self {
        let room = entries.max(1).next_power_of_two();
        let live_room = words.max(4).next_power_of_two();
        let frame =
            crate::buffers::uniform(device, "engine/water-setup", size_of::<SimSetup>() as u64);
        let list = crate::buffers::storage(
            device,
            "engine/water-setup-list",
            room as u64 * 16,
            wgpu::BufferUsages::empty(),
        );
        let live = crate::buffers::storage(
            device,
            "engine/water-setup-live",
            live_room as u64 * 4,
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
                crate::buffers::entry("setup_edges", &buffers.inputs),
                crate::buffers::entry("setup_live", &live),
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

    /// Bytes of its buffers.
    pub(crate) fn bytes(&self) -> u64 {
        self.frame.size() + self.list.size() + self.live.size()
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
/// solids its setup covers cells with, and the level-0 tiles whose records'
/// live cells its setup holds (the first along x and z, and how many along
/// each); the block each slot holds as it was set up (none where it holds
/// none set up on the square's datum and settled cells), and the slot the
/// next frame's ground rises again from; the squares of the level-0 tiles
/// set up from another tile than the window before was (terrain-local least
/// and greatest corners), whether the blocks standing as the settled water
/// does are to stand anew (what they read changed), whether the cells past
/// its edges are to be set up and whether its table is to be written again;
/// whether every block of the square is set up, running or standing, so it
/// runs; the blocks waiting for their data; and what this frame sets up.
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

/// A block a slot holds as it was set up: its place on the terrain's
/// lattice of blocks, and whether its water runs (its data resident when it
/// was set up), or it stands as the settled water does there.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Held {
    place: [i64; 2],
    runs: bool,
}

/// What a frame sets up: the slots of the blocks it sets up whole or as
/// they stand, those it sets up whole, the blocks it sets up (those, and
/// those whose ground alone rises again), the cells past the edges it sets
/// up (none where it sets up none) and whether the frame's encoding
/// recorded it.
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

    /// Prepares the live window `live` for this frame at simulation time
    /// `seconds`, over the square of it the device's bindings hold: its
    /// buffers anew where a square of another size came; where one of its
    /// size moved, or it read other tiles or records, its sources and the
    /// records' live cells; the terrain group its setup reads the ground and
    /// the tiles' water through; the setup of the blocks whose data came to
    /// be resident and of those standing as the settled water does until
    /// it is, of those whose tiles changed and of the cells past its edges
    /// where it or they changed, the ground of RISEN running slots rising
    /// again and its table, wherever the setup can run; the solids pushing
    /// its water; and
    /// as many steps, each of the same seconds, as the simulation time since
    /// the last frame holds, at most LONGEST_FRAME's, each of its index in
    /// the simulation's time, which its bursts are drawn by. Drops it where
    /// there is no window, or no square of it fits.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        live: Option<Live<'_>>,
        seconds: f64,
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
        let resident = |read: &WaterRead| {
            live.ground
                .as_ref()
                .is_some_and(|ground| ground.resident(read))
        };
        current.set_up(queue, sets_up, &resident);
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
        // Bursts are drawn by the cell on the terrain's lattice of the
        // square's cells, an `i32`'s bits each, and the step's index.
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

    /// Sets up the blocks and the cells past the edges this frame lists,
    /// then, once every block of its square is set up, running or standing
    /// as the settled water does, runs the steps prepared over those running
    /// and finds each of their cells' look, timed by `profiler`
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
                (self.steps, window.sourced, false, true),
                blocks,
            );
        }
        encode_drawn(&mut pass, pipelines, &window.buffers, blocks);
    }

    /// What water draws the live water from, once every block of its square
    /// is set up, running or standing as the settled water does: the square
    /// of the window that runs, the slots its blocks turn by, its pool's
    /// slots along each side and its cells' looks (a standing block's
    /// holding no surface); and a count advancing whenever they change.
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
    /// those the simulation's time held past them; the blocks this frame
    /// sets up whole; and the blocks waiting for their data, standing as the
    /// settled water does.
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
    /// pool holding its blocks, with its sources and its records' live
    /// cells, of the scheme's group layout `layout`, and what sets them up,
    /// of the layout `setup`; every block of it, running or standing, the
    /// cells past its edges and its table are set up with the frame.
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

    /// Takes `fitted`, the square of `source` that runs, of its square's
    /// size, in its square's place: its table turned to its place, its
    /// sources, solids and records' live cells, the squares of the tiles set
    /// up from another tile than the window before was, and the blocks
    /// standing as the settled water does, the cells past its edges and its
    /// table to set up again. Its slots keep the blocks both squares hold,
    /// unless the datum or the settled cells changed, and the frame sets up
    /// the rest.
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

    /// Writes its rivers' sources, the solids its setup covers cells with
    /// and its records' live cells, its inputs grown first where they hold
    /// fewer sources and what sets it up where that holds fewer solids or
    /// words of them, and has the blocks standing as the settled water does,
    /// the cells past its edges and its table set up again.
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

    /// Lets go of what the frame before set up: done where its encoding
    /// recorded it, else the slots it set up whole or as they stand hold no
    /// block set up, and the cells past its edges, where it set them up, are
    /// set up again.
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

    /// Lists the frame's setup where it can run (`sets_up`), each block of
    /// its square running where its data is resident (`resident` says so of
    /// every level-0 tile it and a cell past it reach), else standing as the
    /// settled water does: whole, each running block its slot does not hold
    /// as set up running there; as it stands, each standing block its slot
    /// does not hold as set up standing there, and every one where what it
    /// reads changed; whole or as it stands again, each over whose tiles
    /// another tile stands than the window before set it up from; its
    /// ground alone, RISEN of the other slots holding running blocks round
    /// the pool from where the frame before stopped; and the cells past its
    /// edges where the window changed or any block was set up. Each is
    /// listed by its slot, the cell of the terrain's lattice of live cells
    /// its first corner stands at and what it sets up of it, after the
    /// solids in its setup's list; and the table, where it changed, marks
    /// the standing blocks (SIM_STANDING), placing them nowhere. The square
    /// runs once every block of it is set up; where the setup cannot run, the
    /// blocks it would set up wait for a frame where it can, and the square
    /// runs not.
    fn set_up(
        &mut self,
        queue: &wgpu::Queue,
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
        // The squares of the level-0 tiles the window reads whose data is
        // not all resident.
        let missing: Vec<[Vec2; 2]> = self
            .source
            .reads
            .iter()
            .filter(|read| !resident(read))
            .map(|read| read.square)
            .collect();
        // Whether block `place`, with a cell past it on every side, reaches
        // into one of `squares`.
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
                // Where the setup cannot run, each block keeps what its slot
                // holds of it, and the others wait.
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
        // Then as many of the other slots holding running blocks as RISEN,
        // round the pool from where the frame before stopped, their ground
        // rising again.
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
        // Its first cell on the terrain's lattice of live cells, an `i32`'s
        // bits each.
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

    /// Writes its square's table as the solve reads it: each block's slot,
    /// marked SIM_STANDING where the slot does not hold it set up running
    /// there, and each slot's place, none (NO_BLOCK) for such a block, so
    /// none of its water runs and the solve takes it as it takes the cells
    /// past the square's edges.
    fn write_table(&self, queue: &wgpu::Queue) {
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

/// The entry of the setup's list of the block at `place` on the terrain's
/// lattice of blocks in slot `slot`, of which it sets up `what`
/// (`SETUP_WHOLE`, `SETUP_RISEN` or `SETUP_STANDS`): the slot, the cell of
/// the terrain's lattice of live cells its first corner stands at, an
/// `i32`'s bits along x and z, and `what`.
fn entry(slot: u32, place: [i64; 2], what: u32) -> [u32; 4] {
    let side = i64::from(BLOCK_SIDE);
    [
        slot,
        (place[0] * side) as u32,
        (place[1] * side) as u32,
        what,
    ]
}

/// The live cells of the records of settled water `window` reads, as its
/// setup reads them (`meshes/water_setup.wgsl`'s `setup_live`), over its
/// blocks and a block past them on every side, where its ring lies: the
/// level-0 tiles it reads (the first along x and z and how many along
/// each), and its words. For each of those tiles, row by row, the word its
/// table starts at (LIVE_NONE where its record is not held); each table
/// its first block on the terrain's lattice of blocks of live cells (an
/// `i32`'s bits along x and z) and its blocks along each, LIVE_HEAD words
/// of the record's table over those blocks, then the word each of its
/// blocks' cells start at (LIVE_NONE where none of them is wet); and each
/// block's cells, row by row, LIVE_CELL words each, a float's bits: the
/// surface its water stands at (terrain-local; NO_WATER_SURFACE where it is
/// dry) and its flow along x and z (LIVE_HELD along x, and none along z,
/// where a lake holds it).
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
        // An `i32`'s bits each, as the setup reads them.
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
