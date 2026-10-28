//! Water's live windows: the square of ground around the detail eye over
//! which the renderer runs rivers by the shallow-water equations (its
//! `water_sim`), the same water as the settled water around it
//! ([`super::settled`]'s, in cells [`Water::simulation_cell`] across), in
//! live cells finer than the settled ones, as few as put LIVE_ACROSS across
//! the narrowest channel its streams fill ([`live_cell`]): out from the eye
//! as far as those cells' unsteady features stay [`LIVE_PIXELS`] across or
//! more ([`live_reach`]), no farther than a frame's budget of its cells'
//! steps holds ([`Water::live_budget`]), and as far again as the eye goes
//! from its middle before it follows, a whole number of the scheme's blocks
//! of cells along each side. Wherever water draws, a window runs. It holds
//! its ground and the roughness of its rivers' beds, the solids standing in
//! its water, the water to start from, where its rivers bring theirs, the
//! water standing just past its edges, and what the beds hold under its
//! water (their reach and span), which it draws with as the settled water
//! does. Its bed is the terrain's ground under each of its cells, the beds'
//! roughness and what they hold under each one's middle, and the solids
//! standing in the water of the settled records held whole, covering its
//! cells by their share as they cover any cells ([`cover_cells`]); the
//! relief of the ground's materials, which the GPU's materials alone hold,
//! raises that ground as the renderer takes its blocks, at the level whose
//! texels span a live cell (the renderer's `meshes/water_bed.wgsl`), and
//! lets go of the lakes it raises cells out of, so each cell's lake comes
//! as its settled cell holds it. Its water comes from the terrain's settled
//! water through that water's smooth surface ([`SteadyCells::matched`]):
//! each live cell's at the uniform cubic B-spline through the settled cells
//! around it, as water's shading takes its smooth surface, its depth its
//! own over its own ground, held by a lake where the settled cell holding
//! it is, and the live cells over each flowing settled cell raised and
//! turned alike so their mean is that cell's water; it starts off that
//! steady state as unsteadily as the water settled there; the cells past
//! its edges stand as that same water does there, giving and taking water
//! as the steady state does; and its rivers bring what the settled cells
//! bring, shared by area, so water left alone stays as it settled. Its
//! heights stand above a datum, its lowest ground as probed in whole
//! meters, which keeps them small for the GPU's single precision.
//!
//! A window holds its cells by the scheme's blocks of them, each made from
//! its place on the terrain's lattice of blocks, the window's cells' size
//! and datum and what the terrain holds there alone, so whichever way the
//! eye came a window holds what one gathered at once would. It follows the
//! detail eye a snap of whole blocks at a time ([`WaterWindow::snap`]),
//! keeping its datum while its ground lies near it: once the eye stands
//! nearer another snap than the window's, the window there is gathered
//! ahead, GATHERED cells a frame while the window keeps running, taking
//! over the blocks the window and the windows gathered before it hold and
//! gathering the strips the move uncovers, then the cells just past its
//! edges and its rivers' sources, taken over too where the window held
//! them. Gathered, it waits ahead, renderers taking its blocks into the
//! slots the running window leaves free, and takes the window's place once
//! the eye goes FOLLOW of a snap from the window's middle. It waits for its
//! ground's finest tiles and the settled water of the tiles it and the
//! cells past its edges read. Each frame the solids standing in its square
//! reach the renderer with it, nearest the eye first
//! ([`WaterWindow::touching`]), to push its water.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy_ecs::prelude::{DetectChangesMut, Res, ResMut};
use glam::{Vec2, Vec3};

use super::residency::TerrainState;
use super::{TerrainDebug, Terrains, Viewport, Water};
use crate::math::{clamp, max, min, sin_cos};
use crate::terrain_data::shallow_water::{DRY, roughness, step_seconds};
use crate::terrain_data::{
    BLOCK_CELLS, BLOCK_SIDE, Footprint, NARROWEST, SteadyCell, TileKey, cover_cells,
};
use crate::{Camera, WaterObstacle};

/// Pixels across the view a live cell spans, at the view's middle, as far
/// out as water runs live: its smallest unsteady features, an eddy or a
/// surge's front a couple of cells across, stand twice that wide there, and
/// past it they blur under the pixels into the settled water's look, which
/// water's pass hands the live water over to on the way out
/// (`meshes/water.wgsl`'s `water_live_share`).
pub const LIVE_PIXELS: f32 = 4.0;
/// Live cells across the narrowest water whose flow the live solve turns
/// ([`live_cell`]): four, the fewest over which its cells' water, linear
/// across each of them (`water_sim.wgsl`'s reconstruction), carries a
/// velocity rising from a bank or a rock's side to the stream's thread and
/// falling back, two cells each side, so its flow bends around what stands
/// in it and sheds eddies in its lee.
const LIVE_ACROSS: f32 = 4.0;
/// The longest frame whose time the live water steps through: a longer
/// frame steps as many steps as this one holds, so a slow frame costs no
/// more than this ([`live_steps`]), and the water then runs slower than the
/// clock; the foam field steps no more of it either.
pub const LONGEST_FRAME: f64 = 1.0 / 30.0;
/// Settled cells a live cell's settled water reads past the one holding
/// its middle, along each axis: the uniform cubic B-spline's control points
/// around it reach two of them.
const SPLINE_REACH: f32 = 2.0;
/// Meters a window's first corner snaps to, as the whole number of the
/// scheme's blocks of its cells nearest them, and the share of a snap the
/// eye goes from its middle, along x or z, before it follows: a window
/// reaches that far past the live reach on every side, so the reach never
/// leaves it.
const SNAP: f32 = 32.0;
const FOLLOW: f32 = 0.75;
/// Meters apart a window's ground is checked for its finest tiles, and
/// probed for its lowest, before it is gathered: finer than a tile, so none
/// is missed.
const PROBED: f32 = 32.0;
/// What a cell dry past a window's edge stands at: below any ground.
pub const GHOST_DRY: f32 = -1.0e30;
/// Meters a window's lowest ground may lie from the datum of the window
/// before it before it takes its own: heights within a few hundred meters
/// of their datum keep single precision's steps (2^-23 of 512 m is 0.06 mm)
/// under the depth water dries at.
const DATUM_DRIFT: f32 = 256.0;
/// Uniform noise over this either side of zero has unit variance: the
/// square root of 3.
const NOISE_REACH: f32 = 1.732_050_8;
/// Cells a frame gathers at most for the window the eye nears while the
/// window keeps running: a few milliseconds of the main thread's at most. A
/// move of a snap along both axes of a window 256 meters across in
/// half-meter cells (the default budget's, on a view 1080 pixels tall or
/// taller) uncovers some 61,000 cells, gathered over 15 frames as the eye
/// crosses the quarter of a snap between nearing the next snap and
/// following.
const GATHERED: usize = 4096;
/// The most solids standing in a window's square that push its water each
/// frame, nearest the eye first: every cell they reach reads them all, so
/// they bound the work of the pass pushing it; those past them push none.
pub const SIM_TOUCHES: usize = 32;
/// Corners along a side of a block, and a block's corners.
const SIDE: usize = BLOCK_SIDE as usize;
const CORNERS: usize = (SIDE + 1) * (SIDE + 1);
/// Stamps handed out so far: every block gathered takes one of its own.
static STAMPS: AtomicU64 = AtomicU64::new(0);

/// The square over which water runs live: its first corner (world x and
/// z) and the cell there on the terrain's lattice of its cells, its cells
/// along each side, a whole number of the scheme's blocks of them, those
/// its live reach asks for, which a frame's budget of their steps cuts it
/// to ([`Water::live_budget`]), the pixels across the view a cell spans, at
/// the view's middle, as far out as it holds the live reach about the eye
/// wherever the eye stands in it, where water's pass hands its water back
/// to the settled water ([`LIVE_PIXELS`] where it holds all the reach asks,
/// more where the budget cuts it), and the meters across each cell (a live
/// cell's, `live_cell`'s), the blocks it moves by along x or z as it
/// follows the eye (renderers keep as many more slots along each side,
/// which the window it moves to takes its new blocks into ahead), and the
/// height its heights stand above (world meters, whole: its lowest ground
/// as probed, or the datum of the window before it while that lies near
/// it); its blocks of cells, row by row, each as any window holding it
/// gathers it (`WaterBlock`); the cells just past its edges, each as the
/// settled water stands there (along its south edge, its north, its west
/// and its east, each from its first corner); and where its rivers bring
/// water.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterWindow {
    pub origin: [f32; 2],
    pub first: [i64; 2],
    pub cells: u32,
    pub asked: u32,
    pub pixels: f32,
    pub cell: f32,
    pub snap: u32,
    pub datum: f32,
    blocks: Arc<[Arc<WaterBlock>]>,
    ring: Arc<[WaterCell]>,
    pub sources: Arc<[WaterSourceCell]>,
}

/// A block of a window's cells, [`BLOCK_SIDE`] along each side, as any
/// window holding it gathers it: the stamp it took, which no other block
/// holds; the ground at its cells' corners over the datum of the windows
/// holding it (`BLOCK_SIDE + 1` a row, rows along z), which the renderer
/// raises by its materials' relief as it takes the block; and, cells row
/// by row, each one's water as the settled water stands at it
/// (`SteadyCells::matched`), the roughness of its ground as the scheme
/// counts it (the D84 of the river bed under it in meters, negated where
/// steps or boulders are carved into the bed, whose friction then counts
/// its grains alone, and a hillside's stones and turf outside rivers'
/// channels), what the bed holds under its middle as the settled water's
/// sample nearest it holds it (its reach's code beside its span's,
/// [`crate::terrain_data::Tile::bed_word`]) and the share of it covered by
/// the solids standing in the settled water, with the height their tops
/// stand at over the datum ([`cover_cells`]; 0 each where none covers it);
/// and whether its ground came from the finest tiles alone, as every block
/// within the live reach's does: one beyond it may stand on a coarser
/// level's, and is gathered again once the finest tiles stand under it.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterBlock {
    stamp: u64,
    fine: bool,
    corners: [f32; CORNERS],
    water: [WaterCell; BLOCK_CELLS],
    roughness: [f32; BLOCK_CELLS],
    bed_words: [u32; BLOCK_CELLS],
    cover: [[f32; 2]; BLOCK_CELLS],
}

/// A cell of a window, or one just past its edges, as the settled water
/// stands at it: the surface it stands at (meters above the window's datum;
/// NaN where dry), its velocity along x and z (meters a second), the level
/// a lake holds it at (NaN where none does) and its foam; and how far it
/// starts off that steady state, as unsteadily as the water settled there:
/// its surface (meters) and its velocity along x and z (meters a second),
/// each of zero mean.
#[derive(Clone, Copy, Debug, PartialEq)]
struct WaterCell {
    surface: f32,
    velocity: [f32; 2],
    held: f32,
    foam: f32,
    unsteady: [f32; 3],
}

impl WaterCell {
    /// A cell no water stands in.
    const DRY: Self = Self {
        surface: f32::NAN,
        velocity: [0.0; 2],
        held: f32::NAN,
        foam: 0.0,
        unsteady: [0.0; 3],
    };
}

impl WaterBlock {
    /// The stamp it took as it was gathered: blocks of one stamp are one.
    pub fn stamp(&self) -> u64 {
        self.stamp
    }

    /// The ground at its corner (x, z), each from 0 to [`BLOCK_SIDE`].
    pub fn corner(&self, x: u32, z: u32) -> f32 {
        self.corners[z as usize * (SIDE + 1) + x as usize]
    }

    /// The ground at its cell (x, z)'s middle: its corners' mean.
    fn bed(&self, x: u32, z: u32) -> f32 {
        0.25 * (self.corner(x, z)
            + self.corner(x + 1, z)
            + self.corner(x, z + 1)
            + self.corner(x + 1, z + 1))
    }

    /// Its cell (x, z)'s water.
    fn water(&self, x: u32, z: u32) -> WaterCell {
        self.water[z as usize * SIDE + x as usize]
    }

    /// The level a lake holds its cell (x, z) at, where the settled cell
    /// holding its middle is a lake's; NaN where it is not. The live bed
    /// lets the lake go where the cell's ground, raised by the relief of the
    /// ground's materials, stands at or over that level, as a lake holds no
    /// cell it does not stand over (the renderer's `meshes/water_bed.wgsl`).
    pub fn held(&self, x: u32, z: u32) -> f32 {
        self.water(x, z).held
    }

    /// What its cell (x, z) starts from: its surface, its flow along x and
    /// z (square meters a second) and its foam. A cell a lake holds stands
    /// still at its level, below its ground where the lake stands under it,
    /// which the live water dries there; one holding flowing water starts
    /// off its steady state by its unsteadiness, its surface by at most half
    /// its depth so it stays wet, its flow its depth over its ground at its
    /// velocity; a dry one stands at its ground.
    pub fn start(&self, x: u32, z: u32) -> [f32; 4] {
        self.standing(x, z, true)
    }

    /// Its cell (x, z)'s water at its steady state, as [`Self::start`]
    /// holds it without starting off it: what the live window's rim is
    /// pulled back to, and what its swings are measured from.
    pub fn settled(&self, x: u32, z: u32) -> [f32; 4] {
        self.standing(x, z, false)
    }

    /// Its cell (x, z)'s water as [`Self::start`] holds it, off its steady
    /// state by its unsteadiness where `unsteady`.
    fn standing(&self, x: u32, z: u32, unsteady: bool) -> [f32; 4] {
        let water = self.water(x, z);
        let level = self.held(x, z);
        if !level.is_nan() {
            return [level, 0.0, 0.0, water.foam];
        }
        let bed = self.bed(x, z);
        let depth = water.surface - bed;
        if depth.is_nan() || depth <= DRY {
            return [bed, 0.0, 0.0, 0.0];
        }
        let [heave, along_x, along_z] = if unsteady { water.unsteady } else { [0.0; 3] };
        let depth = depth + clamp(heave, -0.5 * depth, 0.5 * depth);
        let velocity = Vec2::from(water.velocity) + Vec2::new(along_x, along_z);
        [
            bed + depth,
            depth * velocity.x,
            depth * velocity.y,
            water.foam,
        ]
    }

    /// The roughness of its cell (x, z)'s ground.
    pub fn roughness(&self, x: u32, z: u32) -> f32 {
        self.roughness[z as usize * SIDE + x as usize]
    }

    /// What the bed holds under its cell (x, z)'s middle.
    pub fn bed_word(&self, x: u32, z: u32) -> u32 {
        self.bed_words[z as usize * SIDE + x as usize]
    }

    /// The share of its cell (x, z) covered by solids, and the height their
    /// tops stand at over the datum.
    pub fn cover(&self, x: u32, z: u32) -> [f32; 2] {
        self.cover[z as usize * SIDE + x as usize]
    }
}

impl WaterWindow {
    /// Its middle (world x and z).
    pub fn middle(&self) -> Vec2 {
        Vec2::from(self.origin) + Vec2::splat(0.5 * self.cells as f32 * self.cell)
    }

    /// Its square's least and greatest corners (world x and z).
    pub fn square(&self) -> (Vec2, Vec2) {
        let low = Vec2::from(self.origin);
        (low, low + Vec2::splat(self.cells as f32 * self.cell))
    }

    /// Its blocks along each side.
    pub fn blocks(&self) -> u32 {
        self.cells / BLOCK_SIDE
    }

    /// Its block (x, z), counted from its first corner.
    pub fn block(&self, x: u32, z: u32) -> &WaterBlock {
        &self.blocks[(z * self.blocks() + x) as usize]
    }

    /// Its block at `at` on the terrain's lattice of blocks, where it holds
    /// one there.
    fn block_on(&self, at: [i64; 2]) -> Option<&Arc<WaterBlock>> {
        let side = i64::from(BLOCK_SIDE);
        let blocks = i64::from(self.blocks());
        let x = at[0] - self.first[0].div_euclid(side);
        let z = at[1] - self.first[1].div_euclid(side);
        ((0..blocks).contains(&x) && (0..blocks).contains(&z))
            .then(|| &self.blocks[(z * blocks + x) as usize])
    }

    /// The water of the terrain's lattice cell `at`, where it covers it or
    /// holds it just past its edges (its corners' cells aside).
    fn water_on(&self, [i, j]: [i64; 2]) -> Option<WaterCell> {
        let n = i64::from(self.cells);
        let (x, z) = (i - self.first[0], j - self.first[1]);
        let inside = |at: i64| (0..n).contains(&at);
        let edge = match (inside(x), inside(z)) {
            (true, true) => {
                let side = i64::from(BLOCK_SIDE);
                let block = self.block((x / side) as u32, (z / side) as u32);
                return Some(block.water((x % side) as u32, (z % side) as u32));
            }
            (true, false) if z == -1 => x,
            (true, false) if z == n => n + x,
            (false, true) if x == -1 => 2 * n + z,
            (false, true) if x == n => 3 * n + z,
            _ => return None,
        };
        Some(self.ring[edge as usize])
    }

    /// The ground at its corner (i, j), each from 0 to its cells along a
    /// side.
    fn corner(&self, i: u32, j: u32) -> f32 {
        let last = self.blocks() - 1;
        let (x, z) = ((i / BLOCK_SIDE).min(last), (j / BLOCK_SIDE).min(last));
        self.block(x, z)
            .corner(i - x * BLOCK_SIDE, j - z * BLOCK_SIDE)
    }

    /// The cells just past its edges as their water stands
    /// ([`WaterGhost`]): south, then north, of each column, then west, then
    /// east, of each row. A flowing one's flow is its depth over the ground
    /// at the face it borders times its velocity.
    pub fn ghosts(&self) -> Vec<WaterGhost> {
        let n = self.cells;
        let ghost = |water: WaterCell, ground: f32, inward: Vec2| {
            if !water.held.is_nan() {
                return WaterGhost {
                    surface: water.held,
                    flow: [0.0; 2],
                    held: water.held,
                    foam: water.foam,
                    inflow: 0.0,
                };
            }
            if water.surface.is_nan() {
                return WaterGhost {
                    surface: GHOST_DRY,
                    flow: [0.0; 2],
                    held: f32::NAN,
                    foam: 0.0,
                    inflow: 0.0,
                };
            }
            let flow = Vec2::from(water.velocity) * max(water.surface - ground, 0.0);
            WaterGhost {
                surface: water.surface,
                flow: flow.to_array(),
                held: f32::NAN,
                foam: water.foam,
                inflow: max(flow.dot(inward), 0.0),
            }
        };
        let ring = |edge: u32, k: u32| self.ring[(edge * n + k) as usize];
        let mut ghosts = Vec::with_capacity(4 * n as usize);
        for k in 0..n {
            ghosts.push(ghost(
                ring(0, k),
                0.5 * (self.corner(k, 0) + self.corner(k + 1, 0)),
                Vec2::Y,
            ));
            ghosts.push(ghost(
                ring(1, k),
                0.5 * (self.corner(k, n) + self.corner(k + 1, n)),
                Vec2::NEG_Y,
            ));
        }
        for k in 0..n {
            ghosts.push(ghost(
                ring(2, k),
                0.5 * (self.corner(0, k) + self.corner(0, k + 1)),
                Vec2::X,
            ));
            ghosts.push(ghost(
                ring(3, k),
                0.5 * (self.corner(n, k) + self.corner(n, k + 1)),
                Vec2::NEG_X,
            ));
        }
        ghosts
    }

    /// The square of its middle `cells` cells along each side, whole blocks
    /// of them (all of it, where it has no more): its blocks there, the
    /// cells just past that square's edges standing as it holds them, and
    /// its sources there, on its datum, its cells spanning as many more
    /// pixels where it hands back as it holds less of the reach.
    #[must_use]
    pub fn within(&self, cells: u32) -> Self {
        let side = self.blocks();
        let kept = (cells / BLOCK_SIDE).clamp(1, side);
        let off = (side - kept) / 2;
        let (shift, n) = (off * BLOCK_SIDE, kept * BLOCK_SIDE);
        let first = [
            self.first[0] + i64::from(shift),
            self.first[1] + i64::from(shift),
        ];
        // The reach it holds about the eye, the eye FOLLOW of a snap from its
        // middle at most, and the pixels a cell spans there.
        let follows = FOLLOW * self.snap as f32 * BLOCK_SIDE as f32 * self.cell;
        let holds = |cells: u32| max(0.5 * cells as f32 * self.cell - follows, self.cell);
        let mut blocks = Vec::with_capacity((kept * kept) as usize);
        for z in off..off + kept {
            for x in off..off + kept {
                blocks.push(Arc::clone(&self.blocks[(z * side + x) as usize]));
            }
        }
        let ring = ring_cells(first, n)
            .map(|at| self.water_on(at).unwrap_or(WaterCell::DRY))
            .collect();
        let sources = self
            .sources
            .iter()
            .filter_map(|source| {
                let (i, j) = (source.cell % self.cells, source.cell / self.cells);
                let inside = i >= shift && j >= shift && i < shift + n && j < shift + n;
                inside.then(|| WaterSourceCell {
                    cell: (j - shift) * n + i - shift,
                    ..*source
                })
            })
            .collect();
        Self {
            origin: (Vec2::from(self.origin) + Vec2::splat(shift as f32 * self.cell)).to_array(),
            first,
            cells: n,
            asked: self.asked,
            pixels: self.pixels * holds(self.cells) / holds(n),
            cell: self.cell,
            snap: self.snap,
            datum: self.datum,
            blocks: blocks.into(),
            ring,
            sources,
        }
    }

    /// The solids of `obstacles` whose footprints reach over its square,
    /// nearest `eye` first, into `touching`, as many as [`SIM_TOUCHES`]:
    /// how many more stand there.
    pub fn touching(
        &self,
        obstacles: &[WaterObstacle],
        eye: Vec3,
        touching: &mut Vec<WaterObstacle>,
    ) -> u32 {
        let (low, high) = self.square();
        touching.clear();
        touching.extend(obstacles.iter().filter(|obstacle| {
            let reach = obstacle.reach();
            let (at, half) = (
                Vec2::new(obstacle.center.x, obstacle.center.z),
                Vec2::new(reach.x, reach.z),
            );
            (at + half).cmpge(low).all() && (at - half).cmple(high).all()
        }));
        touching.sort_by(|a, b| {
            a.center
                .distance_squared(eye)
                .total_cmp(&b.center.distance_squared(eye))
        });
        let beyond = touching.len().saturating_sub(SIM_TOUCHES);
        touching.truncate(SIM_TOUCHES);
        beyond as u32
    }
}

/// Where a window's rivers bring water to one of its cells (its index, row
/// by row): the cubic meters a second, and the way the water runs as it
/// comes, as a share of the source speed along x and z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSourceCell {
    pub cell: u32,
    pub rate: f32,
    pub way: [f32; 2],
}

/// A cell just past a window's edge, as its water settled: the surface it
/// stands at (meters above the window's datum; [`GHOST_DRY`] where dry),
/// its flow along x and z (square meters a second), the level a lake holds
/// it at (NaN where none does), its foam, and the flow it brings into the
/// window across the face it borders (square meters a second; none where it
/// runs out or along).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterGhost {
    pub surface: f32,
    pub flow: [f32; 2],
    pub held: f32,
    pub foam: f32,
    pub inflow: f32,
}

impl Terrains {
    /// The squares around the detail eye over which terrains' water runs
    /// live: each one's least and greatest corners (world x and z).
    pub fn water_windows(&self) -> impl Iterator<Item = (Vec2, Vec2)> + '_ {
        self.entries
            .iter()
            .filter_map(|state| state.hydrology.window.as_deref())
            .map(WaterWindow::square)
    }
}

/// The meters from the eye out to which a cell `cell` meters across spans
/// [`LIVE_PIXELS`] pixels or more across a view `height` pixels tall
/// through a vertical field of view of `fov` radians, at its middle, as
/// the detail eye's measures count a pixel (`scene.wgsl`'s
/// `detail_across`): how far water runs live, where its budget holds as
/// far. Half-meter cells reach 130 meters on a view 1080 pixels tall
/// through 55 degrees, and 173 meters on one 1440 pixels tall.
fn live_reach(cell: f32, height: u32, fov: f32) -> f32 {
    let (sin, cos) = sin_cos(0.5 * fov);
    cell * height.max(1) as f32 * cos / max(2.0 * sin * LIVE_PIXELS, 1.0e-6)
}

/// The meters across the live cells over settled cells `settled` meters
/// across, on a terrain whose finest level's samples stand `finest` apart:
/// LIVE_ACROSS of them across the narrowest water its streams fill, the
/// narrowest channel its bake cuts ([`NARROWEST`] of those samples, the
/// fewest any feature of its ground spans), each settled cell split along
/// each side into the fewest that does, a power of two no more than a
/// block's side, so the live cells tile the settled cells and each block
/// holds whole settled cells; the settled cells themselves where they are
/// as fine. Rocks standing in the water, solids, turn it by the share of
/// the live cells they cover. For the cook's meter cells over a meter's
/// samples, half a meter: four across a two-meter channel, and two or more
/// across a boulder a meter wide; a quarter of a meter would put four
/// across it, but its steps, twice as many, would leave the budget a reach
/// of a few tens of meters.
fn live_cell(settled: f32, finest: f32) -> f32 {
    let narrowest = NARROWEST * finest;
    let split = max((LIVE_ACROSS * settled / narrowest).ceil(), 1.0) as u32;
    settled / split.next_power_of_two().min(BLOCK_SIDE) as f32
}

/// The most steps the live water takes a frame over cells `cell` meters
/// across: as many as a frame [`LONGEST_FRAME`] long holds, each the
/// longest in which its fastest water and waves cross a share of a cell
/// (`step_seconds`), so the steps a frame dispatches stay a fixed count:
/// eight of half-meter cells.
pub fn live_steps(cell: f32) -> u32 {
    (LONGEST_FRAME / f64::from(step_seconds(cell))).ceil() as u32
}

/// Places a window around the detail eye (the camera's eye unless frozen,
/// [`TerrainDebug`]) wherever water draws, over the first baked terrain no
/// other surrounds, in live cells over its settled water's ([`live_cell`]),
/// reaching [`live_reach`] on the frame's viewport, as far as the water's
/// budget of cell steps holds ([`follow`]); drops every other window, and
/// every window waiting ahead or being gathered, as it does where water
/// does not draw.
pub(crate) fn place_water_windows(
    mut terrains: ResMut<Terrains>,
    camera: Res<Camera>,
    viewport: Option<Res<Viewport>>,
    debug: Option<Res<TerrainDebug>>,
    water: Option<Res<Water>>,
) {
    // Windows are read, not watched: placing them changes nothing
    // terrains' observers wait on.
    let terrains = terrains.bypass_change_detection();
    let water = water.map(|water| *water);
    let drawn = water.is_some_and(|water| water.enabled);
    let budget = water.unwrap_or_default().live_budget;
    let wanted = drawn
        .then(|| {
            terrains
                .entries
                .iter()
                .position(|state| state.around.is_none() && state.baked.is_some())
        })
        .flatten();
    let camera = camera.sanitized();
    let height = viewport.map_or_else(|| Viewport::default().height, |viewport| viewport.height);
    let eye = debug
        .and_then(|debug| debug.freeze)
        .unwrap_or(camera.position);
    for (index, state) in terrains.entries.iter_mut().enumerate() {
        let settled = state.settled.cell();
        if Some(index) != wanted || !(settled.is_finite() && settled > 0.0) {
            state.hydrology.window = None;
            state.hydrology.ahead = None;
            state.hydrology.gathering = None;
            continue;
        }
        let cell = live_cell(settled, state.terrain.pyramid.spacing(0));
        let reach = live_reach(cell, height, camera.vertical_fov_radians);
        follow(state, (reach, cell, budget), eye);
    }
}

/// Follows the detail eye at `eye` with `state`'s window, in cells `cell`
/// meters across, as many along each side as reach `reach` meters from the
/// eye and FOLLOW of a snap past it with whole blocks of them, or as many
/// as `budget` cell steps hold over a frame's most steps ([`live_steps`])
/// where they are fewer, though no fewer than hold FOLLOW of a snap about
/// its middle and a block more on each side, its rim's: once the eye
/// stands nearer another snap than the window's, the window at the snap
/// nearest it is gathered ahead, GATHERED cells a frame, taking over what
/// the window, the window waiting ahead and the gathering before this one
/// hold on its datum; once whole it waits ahead, and takes the window's
/// place once the eye has gone FOLLOW of a snap from the window's middle
/// along x or z, or at once where no window of its size runs. The window
/// running keeps the cells its reach asks for, and the pixels a cell spans
/// as far out as it holds the reach, as they change.
fn follow(state: &mut TerrainState, (reach, cell, budget): (f32, f32, u32), eye: Vec3) {
    let side = BLOCK_SIDE as f32;
    let snap_blocks = max((SNAP / (cell * side)).round(), 1.0);
    let snap = snap_blocks * side * cell;
    let block = cell * side;
    let asked = max((2.0 * (reach + FOLLOW * snap) / block).ceil(), 1.0);
    let afforded = (f64::from(budget) / f64::from(live_steps(cell))).sqrt() as f32 / side;
    let least = (2.0 * FOLLOW * snap / block).ceil() + 2.0;
    let cells = min(asked, max(afforded.floor(), least)) as u32 * BLOCK_SIDE;
    let asked = asked as u32 * BLOCK_SIDE;
    let size = cells as f32 * cell;
    // The pixels a cell spans as far out as the window holds the reach
    // about the eye wherever it stands in it: LIVE_PIXELS where it holds all
    // the reach asks, more where the budget cuts it.
    let pixels = LIVE_PIXELS * max(reach / max(0.5 * size - FOLLOW * snap, cell), 1.0);
    let eye = Vec2::new(eye.x, eye.z);
    let before = state
        .hydrology
        .window
        .clone()
        .filter(|window| window.cell == cell);
    let current = before.as_ref().filter(|window| window.cells == cells);
    // The window's first corner on the terrain's lattice of its cells, at
    // the snap nearest the eye.
    let flat = Vec2::new(state.terrain.origin.x, state.terrain.origin.z);
    let at = ((eye - flat - Vec2::splat(0.5 * size)) / snap).round();
    let step = snap_blocks as i64 * i64::from(BLOCK_SIDE);
    let first = [at.x as i64 * step, at.y as i64 * step];
    if let Some(window) = current.filter(|window| window.first == first) {
        if (window.asked, window.pixels) != (asked, pixels) {
            state.hydrology.window = Some(Arc::new(WaterWindow {
                asked,
                pixels,
                ..WaterWindow::clone(window)
            }));
        }
        state.hydrology.gathering = None;
        return;
    }
    let follows =
        current.is_none_or(|window| (eye - window.middle()).abs().max_element() > FOLLOW * snap);
    let waiting = state
        .hydrology
        .ahead
        .clone()
        .filter(|ahead| ahead.cell == cell && ahead.cells == cells);
    if let Some(ahead) = waiting.as_ref().filter(|ahead| ahead.first == first) {
        state.hydrology.gathering = None;
        if follows {
            state.hydrology.window = Some(ahead.clone());
            state.hydrology.ahead = None;
        }
        return;
    }
    let Some(lowest) = finest(state, first, (cells, cell), (eye - flat, reach)) else {
        return;
    };
    if !settled(state, first, (cells, cell)) {
        return;
    }
    // Its datum: the window before it's while its ground lies near that.
    let lowest = lowest.floor();
    let datum = before.as_ref().map_or(lowest, |window| {
        if (lowest - window.datum).abs() <= DATUM_DRIFT {
            window.datum
        } else {
            lowest
        }
    });
    let before = before.filter(|window| window.datum == datum);
    let waiting = waiting.filter(|ahead| ahead.datum == datum);
    let origin = flat + Vec2::new(first[0] as f32, first[1] as f32) * cell;
    let mut gathering = match state.hydrology.gathering.take() {
        Some(gathering) if gathering.gathers(first, (cells, cell), datum) => gathering,
        earlier => {
            let earlier = earlier.filter(|earlier| earlier.cell == cell && earlier.datum == datum);
            let present: &TerrainState = state;
            Gathering::new(
                (origin, first),
                (cells, cell, snap_blocks as u32),
                datum,
                |at| {
                    before
                        .as_ref()
                        .and_then(|window| window.block_on(at))
                        .or_else(|| waiting.as_ref().and_then(|ahead| ahead.block_on(at)))
                        .or_else(|| earlier.as_ref().and_then(|earlier| earlier.block_on(at)))
                        .filter(|block| block.fine || !refined(present, at, cell))
                        .cloned()
                },
            )
        }
    };
    if gathering.gather(state, GATHERED) {
        let window = Arc::new(gathering.finish(state, before.as_deref(), (asked, pixels)));
        if follows {
            state.hydrology.window = Some(window);
            state.hydrology.ahead = None;
        } else {
            state.hydrology.ahead = Some(window);
        }
    } else {
        state.hydrology.gathering = Some(gathering);
    }
}

/// A window being gathered for the square the eye nears, a few of its
/// blocks a frame while the window keeps running: its first corner (world x
/// and z) and the cell there on the terrain's lattice of its cells, its
/// cells along each side, the meters across each and the blocks it moves
/// by, its datum, and its blocks, row by row, each once gathered or taken
/// over.
pub(super) struct Gathering {
    origin: Vec2,
    first: [i64; 2],
    cells: u32,
    cell: f32,
    snap: u32,
    datum: f32,
    blocks: Vec<Option<Arc<WaterBlock>>>,
}

impl Gathering {
    /// The window from `origin` (world x and z), the cell `first` on the
    /// terrain's lattice of its cells, of `cells` cells `cell` meters across
    /// along each side moving by `snap` blocks, on `datum`: each of its
    /// blocks `carried` holds at its place on the terrain's lattice of
    /// blocks taken over, the rest still to gather.
    fn new(
        (origin, first): (Vec2, [i64; 2]),
        (cells, cell, snap): (u32, f32, u32),
        datum: f32,
        carried: impl Fn([i64; 2]) -> Option<Arc<WaterBlock>>,
    ) -> Self {
        let side = i64::from(cells / BLOCK_SIDE);
        let corner = first.map(|first| first.div_euclid(i64::from(BLOCK_SIDE)));
        let blocks = (0..side * side)
            .map(|at| carried([corner[0] + at % side, corner[1] + at / side]))
            .collect();
        Self {
            origin,
            first,
            cells,
            cell,
            snap,
            datum,
            blocks,
        }
    }

    /// Whether it gathers the window from the terrain's lattice cell
    /// `first` of `cells` cells `cell` meters across along each side, on
    /// `datum`.
    fn gathers(&self, first: [i64; 2], (cells, cell): (u32, f32), datum: f32) -> bool {
        self.first == first && self.cells == cells && self.cell == cell && self.datum == datum
    }

    /// Its block at `at` on the terrain's lattice of blocks, once gathered.
    fn block_on(&self, at: [i64; 2]) -> Option<&Arc<WaterBlock>> {
        let side = i64::from(self.cells / BLOCK_SIDE);
        let x = at[0] - self.first[0].div_euclid(i64::from(BLOCK_SIDE));
        let z = at[1] - self.first[1].div_euclid(i64::from(BLOCK_SIDE));
        if !(0..side).contains(&x) || !(0..side).contains(&z) {
            return None;
        }
        self.blocks[(z * side + x) as usize].as_ref()
    }

    /// Gathers as many of its blocks still to gather as hold `cells` cells
    /// from `state`, in order, a run of them along a row at a time; whether
    /// it holds every block.
    fn gather(&mut self, state: &TerrainState, cells: usize) -> bool {
        let side = (self.cells / BLOCK_SIDE) as usize;
        let corner = self
            .first
            .map(|first| first.div_euclid(i64::from(BLOCK_SIDE)));
        let mut left = cells / BLOCK_CELLS;
        let mut at = 0;
        while at < self.blocks.len() {
            if self.blocks[at].is_some() {
                at += 1;
                continue;
            }
            if left == 0 {
                return false;
            }
            let row = at / side;
            let mut end = at + 1;
            while end < (row + 1) * side && end - at < left && self.blocks[end].is_none() {
                end += 1;
            }
            let (from, to) = ((at - row * side) as i64, (end - row * side) as i64);
            let columns = corner[0] + from..corner[0] + to;
            let run = gather_run(
                state,
                (self.cell, self.datum),
                corner[1] + row as i64,
                columns,
            );
            for (offset, block) in run.into_iter().enumerate() {
                self.blocks[at + offset] = Some(Arc::new(block));
            }
            left -= end - at;
            at = end;
        }
        true
    }

    /// The window, its blocks all gathered, with the cells just past its
    /// edges and its rivers' sources: taken over from `before`, the window
    /// before it on its datum, where that holds them, and read from `state`
    /// where it does not, each cell past its edges as the settled water
    /// stands at it over its own ground, as it would inside a window
    /// (`SteadyCells::matched`); with `asked`, the cells its live reach asks
    /// for along each side, and `pixels`, the pixels a cell spans where its
    /// live water hands back.
    fn finish(
        self,
        state: &TerrainState,
        before: Option<&WaterWindow>,
        (asked, pixels): (u32, f32),
    ) -> WaterWindow {
        let n = self.cells as usize;
        let around: Vec<[i64; 2]> = ring_cells(self.first, self.cells).collect();
        let mut ring: Vec<Option<WaterCell>> = around
            .iter()
            .map(|&at| before.and_then(|window| window.water_on(at)))
            .collect();
        // Along each edge, the stretch of the cells the window before it
        // does not hold is read, over the settled cells their settled water
        // reads.
        let settled = state.settled.cell();
        let split = (settled / self.cell).round() as i64;
        let middle = |[i, j]: [i64; 2]| (Vec2::new(i as f32, j as f32) + 0.5) * self.cell;
        let reach = Vec2::splat((SPLINE_REACH + 1.0) * settled);
        let bed = |i: i64, j: i64| {
            let corner = |x: i64, z: i64| {
                ground_at(state, Vec2::new(x as f32, z as f32) * self.cell, self.datum).0
            };
            0.25 * (corner(i, j) + corner(i + 1, j) + corner(i, j + 1) + corner(i + 1, j + 1))
        };
        for edge in 0..4 {
            let line = edge * n..(edge + 1) * n;
            let fresh: Vec<usize> = line.filter(|&at| ring[at].is_none()).collect();
            let (Some(&low), Some(&high)) = (fresh.first(), fresh.last()) else {
                continue;
            };
            let ends = (middle(around[low]) - reach, middle(around[high]) + reach);
            let steady = SteadyCells::read(state, ends, self.datum);
            // The cells along an edge lie a settled cell's live cells in a
            // row in the same settled cell, read once.
            let mut read: Option<([i64; 2], Vec<WaterCell>)> = None;
            for at in fresh {
                let [i, j] = around[at];
                let holding = [i.div_euclid(split), j.div_euclid(split)];
                if read.as_ref().is_none_or(|(those, _)| *those != holding) {
                    read = Some((holding, steady.matched(holding, (self.cell, split), &bed)));
                }
                let within = j.rem_euclid(split) * split + i.rem_euclid(split);
                ring[at] = read.as_ref().map(|(_, cells)| cells[within as usize]);
            }
        }
        let sources = sources(state, (self.first, self.cells, self.cell), before);
        WaterWindow {
            origin: self.origin.to_array(),
            first: self.first,
            cells: self.cells,
            asked,
            pixels,
            cell: self.cell,
            snap: self.snap,
            datum: self.datum,
            blocks: self.blocks.into_iter().flatten().collect(),
            ring: ring
                .into_iter()
                .map(|water| water.unwrap_or(WaterCell::DRY))
                .collect(),
            sources: sources.into(),
        }
    }
}

/// The terrain's lattice cells just past the edges of the square from its
/// lattice cell `first`, `cells` of them along each side, in the order
/// windows hold them: along its south edge, its north, its west and its
/// east, each from its first corner.
fn ring_cells(first: [i64; 2], cells: u32) -> impl Iterator<Item = [i64; 2]> {
    let n = i64::from(cells);
    let [x, z] = first;
    (0..4).flat_map(move |edge| {
        (0..n).map(move |k| match edge {
            0 => [x + k, z - 1],
            1 => [x + k, z + n],
            2 => [x - 1, z + k],
            _ => [x + n, z + k],
        })
    })
}

/// The lowest ground (world meters) under the square from the terrain's
/// lattice cell `first` of `cells` cells `cell` meters across along each
/// side, probed PROBED meters apart, where `state` holds tiles under every
/// probe, its finest under every probe within the live reach `reach` of the
/// terrain-local eye `eye` (x and z) and a probe's spacing more, so every
/// tile reaching into the reach is found; `None` where it does not.
fn finest(
    state: &TerrainState,
    first: [i64; 2],
    (cells, cell): (u32, f32),
    (eye, reach): (Vec2, f32),
) -> Option<f32> {
    let size = cells as f32 * cell;
    let steps = max((size / PROBED).ceil(), 1.0) as u32;
    let apart = size / steps as f32;
    let low = Vec2::new(first[0] as f32, first[1] as f32) * cell;
    let mut lowest = f32::INFINITY;
    for j in 0..=steps {
        for i in 0..=steps {
            let point = low + Vec2::new(i as f32, j as f32) * apart;
            let (level, sample) = state.sample(point)?;
            if level != 0 && point.distance(eye) <= reach + PROBED {
                return None;
            }
            lowest = min(lowest, sample.height);
        }
    }
    Some(state.terrain.origin.y + lowest)
}

/// Whether `state`'s finest tiles stand under every corner of the block at
/// `at` on the terrain's lattice of blocks of cells `cell` meters across,
/// so a block gathered from coarser ground there is gathered again.
fn refined(state: &TerrainState, at: [i64; 2], cell: f32) -> bool {
    let side = i64::from(BLOCK_SIDE);
    [[0, 0], [1, 0], [0, 1], [1, 1]].iter().all(|[x, z]| {
        let corner = Vec2::new(((at[0] + x) * side) as f32, ((at[1] + z) * side) as f32) * cell;
        state.sample(corner).is_some_and(|(level, _)| level == 0)
    })
}

/// Whether `state` holds the settled water the window from the cell `first`
/// of the terrain's lattice of cells `cell` meters across, over `cells` of
/// them along each side, reads: that of every level-0 tile its bake's water
/// reaches under the window, the cells just past its edges and the settled
/// cells their settled water reads around them (`SteadyCells::matched`).
fn settled(state: &TerrainState, first: [i64; 2], (cells, cell): (u32, f32)) -> bool {
    let pyramid = &state.terrain.pyramid;
    let size = pyramid.tile_size(0);
    let reach = Vec2::splat(cell + (SPLINE_REACH + 1.0) * state.settled.cell());
    let low = Vec2::new(first[0] as f32, first[1] as f32) * cell - reach;
    let high = Vec2::new(
        (first[0] + i64::from(cells)) as f32,
        (first[1] + i64::from(cells)) as f32,
    ) * cell
        + reach;
    let [columns, rows] = pyramid.tiles(0);
    let range = |low: f32, high: f32, count: u32| {
        let first = max((low / size).floor(), 0.0) as u32;
        let last = min((high / size).floor(), count as f32 - 1.0) as u32;
        first..=last
    };
    range(low.y, high.y, rows).all(|z| {
        range(low.x, high.x, columns)
            .all(|x| state.settled.has(TileKey::new(0, x as u16, z as u16)))
    })
}

/// The blocks of the terrain's lattice of blocks along row `row`, over
/// `columns`, in live cells `cell` meters across, as `state`'s resident
/// tiles and settled water stand there, their heights over `datum` (world
/// meters): the ground at their corners from the finest tiles holding them
/// (`ground_at`), their cells' water as the settled water stands at them
/// over their own ground (`SteadyCells::matched`, each block holding whole
/// settled cells), their ground's roughness and what their beds hold from
/// the beds of the finest tiles under their middles and those tiles'
/// settled water there, and what the solids standing in the settled water
/// held whole cover of their cells.
fn gather_run(
    state: &TerrainState,
    (cell, datum): (f32, f32),
    row: i64,
    columns: Range<i64>,
) -> Vec<WaterBlock> {
    let side = i64::from(BLOCK_SIDE);
    let anchor = state.terrain.origin.y;
    let pyramid = &state.terrain.pyramid;
    let settled = state.settled.cell();
    let split = (settled / cell).round() as i64;
    // Its cells, from `low` up to before `high` on the terrain's lattice.
    let low = [columns.start * side, row * side];
    let high = [columns.end * side, (row + 1) * side];
    let middle = |i: i64, j: i64| (Vec2::new(i as f32, j as f32) + 0.5) * cell;
    // The ground at its corners, rows along z, and whether the finest
    // tiles hold it.
    let stride = (high[0] - low[0] + 1) as usize;
    let mut ground = Vec::with_capacity(stride * (SIDE + 1));
    let mut fine = Vec::with_capacity(stride * (SIDE + 1));
    for j in low[1]..=high[1] {
        for i in low[0]..=high[0] {
            let (height, finest) = ground_at(state, Vec2::new(i as f32, j as f32) * cell, datum);
            ground.push(height);
            fine.push(finest);
        }
    }
    // Its cells' water, `stride - 1` cells a row, settled cell by settled
    // cell, from the settled water around them.
    let across = stride - 1;
    let around = Vec2::splat((SPLINE_REACH + 1.0) * settled);
    let ends = (
        middle(low[0], low[1]) - around,
        middle(high[0] - 1, high[1] - 1) + around,
    );
    let steady = SteadyCells::read(state, ends, datum);
    let bed = |i: i64, j: i64| {
        let at = (j - low[1]) as usize * stride + (i - low[0]) as usize;
        0.25 * (ground[at] + ground[at + 1] + ground[at + stride] + ground[at + stride + 1])
    };
    let mut waters = vec![WaterCell::DRY; across * SIDE];
    for z in low[1].div_euclid(split)..high[1].div_euclid(split) {
        for x in low[0].div_euclid(split)..high[0].div_euclid(split) {
            let cells = steady.matched([x, z], (cell, split), &bed);
            for (k, standing) in (0_i64..).zip(cells) {
                let (i, j) = (x * split + k % split, z * split + k / split);
                waters[(j - low[1]) as usize * across + (i - low[0]) as usize] = standing;
            }
        }
    }
    // What the solids standing in the settled water cover of its cells,
    // their tops over the datum, `stride - 1` cells a row.
    let reach = (
        Vec2::new(low[0] as f32, low[1] as f32) * cell,
        Vec2::new(high[0] as f32, high[1] as f32) * cell,
    );
    let solids: Vec<Footprint> = state
        .settled
        .held
        .iter()
        .flat_map(|(_, (water, _))| water.solids.iter().copied())
        .filter(|solid| {
            let (at, half) = (solid.middle(), solid.reach());
            (at + half).cmpge(reach.0).all() && (at - half).cmple(reach.1).all()
        })
        .collect();
    let mut covers = vec![[0.0_f32; 2]; across * SIDE];
    let cells = [across as u32, BLOCK_SIDE];
    for ([i, j], [share, top]) in cover_cells(&solids, (low, cells), cell) {
        covers[j as usize * across + i as usize] = [share, anchor + top - datum];
    }
    let tile = |point: Vec2| {
        let key = pyramid.tile_at(0, point)?;
        Some((state.store.get(&key)?, point - pyramid.tile_origin(key)))
    };
    let start = columns.start;
    columns
        .map(|column| {
            let corner = ((column - start) * side) as usize;
            let at = |k: usize| {
                let (i, j) = (
                    column * side + (k % SIDE) as i64,
                    row * side + (k / SIDE) as i64,
                );
                middle(i, j)
            };
            let beds: [(f32, u32); BLOCK_CELLS] = std::array::from_fn(|k| {
                tile(at(k)).map_or((roughness(None), 0), |(tile, local)| {
                    (roughness(tile.bed(local).as_ref()), tile.bed_word(local))
                })
            });
            let corners = |k: usize| k / (SIDE + 1) * stride + corner + k % (SIDE + 1);
            WaterBlock {
                stamp: STAMPS.fetch_add(1, Ordering::Relaxed) + 1,
                fine: (0..CORNERS).all(|k| fine[corners(k)]),
                corners: std::array::from_fn(|k| ground[corners(k)]),
                water: std::array::from_fn(|k| waters[k / SIDE * across + corner + k % SIDE]),
                roughness: beds.map(|(rough, _)| rough),
                bed_words: beds.map(|(_, word)| word),
                cover: std::array::from_fn(|k| covers[k / SIDE * across + corner + k % SIDE]),
            }
        })
        .collect()
}

/// The ground at terrain-local `point` over `datum` (world meters), from
/// the finest tiles `state` holds there (the terrain's origin where it
/// holds none), and whether those are the finest level's.
fn ground_at(state: &TerrainState, point: Vec2, datum: f32) -> (f32, bool) {
    let anchor = state.terrain.origin.y;
    let sample = state.sample(point);
    let height = sample.map_or(anchor, |(_, sample)| anchor + sample.height);
    (height - datum, sample.is_some_and(|(level, _)| level == 0))
}

/// Where the rivers of `state`'s settled water bring water to the window
/// from the terrain's lattice cell `first` of `cells` cells `cell` meters
/// across along each side, cells in order: over the cells `before` covers,
/// what it brings there; over the rest, what each settled cell over them
/// brings, shared among the window's cells by the area of it each covers,
/// every cell's summed and the way it comes the rate-weighted mean of what
/// it is brought. Each cell's comes from the tiles under it in their keys'
/// order, so a cell is brought the same in every window covering it.
fn sources(
    state: &TerrainState,
    (first, cells, cell): ([i64; 2], u32, f32),
    before: Option<&WaterWindow>,
) -> Vec<WaterSourceCell> {
    let settled = state.settled.cell();
    let pyramid = &state.terrain.pyramid;
    let n = i64::from(cells);
    let area = settled * settled;
    let covered = |x: i64, z: i64| {
        before.is_some_and(|window| {
            let held = i64::from(window.cells);
            let (i, j) = (
                first[0] + x - window.first[0],
                first[1] + z - window.first[1],
            );
            (0..held).contains(&i) && (0..held).contains(&j)
        })
    };
    let mut found: Vec<WaterSourceCell> = before.map_or_else(Vec::new, |window| {
        let held = window.cells;
        window
            .sources
            .iter()
            .filter_map(|source| {
                let x = window.first[0] + i64::from(source.cell % held) - first[0];
                let z = window.first[1] + i64::from(source.cell / held) - first[1];
                ((0..n).contains(&x) && (0..n).contains(&z)).then(|| WaterSourceCell {
                    cell: (z * n + x) as u32,
                    ..*source
                })
            })
            .collect()
    });
    // The window's cells along `axis` that a settled cell from `low` to
    // `low + settled` (terrain-local meters) overlaps, each counted from
    // the window's first and with the meters it overlaps.
    let spans = |axis: usize, low: f32| {
        let high = low + settled;
        let from = ((low / cell).floor() as i64).max(first[axis]);
        let to = ((high / cell).ceil() as i64).min(first[axis] + n);
        (from..to).filter_map(move |at| {
            let overlap = min((at + 1) as f32 * cell, high) - max(at as f32 * cell, low);
            (overlap > 0.0).then_some((at - first[axis], overlap))
        })
    };
    // The level-0 tiles whose settled cells reach over the window, in their
    // keys' order.
    let size = pyramid.tile_size(0);
    let [columns, rows] = pyramid.tiles(0);
    let range = |low: f32, high: f32, count: u32| {
        let first = max((low / size).floor(), 0.0) as u32;
        let last = min((high / size).floor(), count as f32 - 1.0) as u32;
        first..=last
    };
    let margin = Vec2::splat(settled);
    let low = Vec2::new(first[0] as f32, first[1] as f32) * cell - margin;
    let high = Vec2::new((first[0] + n) as f32, (first[1] + n) as f32) * cell + margin;
    let mut keys: Vec<TileKey> = range(low.y, high.y, rows)
        .flat_map(|z| {
            range(low.x, high.x, columns).map(move |x| TileKey::new(0, x as u16, z as u16))
        })
        .collect();
    keys.sort_unstable();
    let mut brought: Vec<(u32, f32, [f32; 2])> = Vec::new();
    for key in keys {
        let Some((water, _)) = state.settled.held.get(&key) else {
            continue;
        };
        let columns = i64::from(water.cells[0]);
        for source in &water.sources {
            let low = Vec2::new(
                (water.first[0] + i64::from(source.cell) % columns) as f32,
                (water.first[1] + i64::from(source.cell) / columns) as f32,
            ) * settled;
            for (z, along_z) in spans(1, low.y) {
                for (x, along_x) in spans(0, low.x) {
                    if covered(x, z) {
                        continue;
                    }
                    let rate = source.rate * along_x * along_z / area;
                    let way = [source.way[0] * rate, source.way[1] * rate];
                    brought.push(((z * n + x) as u32, rate, way));
                }
            }
        }
    }
    brought.sort_by_key(|&(at, _, _)| at);
    let mut merged: Vec<(WaterSourceCell, [f32; 2])> = Vec::with_capacity(brought.len());
    for (at, rate, way) in brought {
        match merged.last_mut() {
            Some((last, weighted)) if last.cell == at => {
                last.rate += rate;
                *weighted = [weighted[0] + way[0], weighted[1] + way[1]];
            }
            _ => merged.push((
                WaterSourceCell {
                    cell: at,
                    rate,
                    way: [0.0; 2],
                },
                way,
            )),
        }
    }
    found.extend(merged.into_iter().map(|(source, weighted)| {
        let share = if source.rate > 0.0 {
            1.0 / source.rate
        } else {
            0.0
        };
        WaterSourceCell {
            way: [weighted[0] * share, weighted[1] * share],
            ..source
        }
    }));
    found.sort_by_key(|source| source.cell);
    found
}

/// Settled water as a window reads it: a rectangle of the terrain's
/// lattice of settled cells (its first cell along x and z, its columns and
/// rows, and the meters across each cell), each cell's water row by row,
/// its heights above the window's datum; cells past it read dry.
struct SteadyCells {
    first: [i64; 2],
    columns: usize,
    rows: usize,
    cell: f32,
    cells: Vec<Steady>,
}

/// A settled cell as a window reads it: dry; held by a lake at its level,
/// with its foam; or flowing, its water standing at its surface and running
/// at its velocity, with its foam and how far its surface and its velocity
/// strayed from their means as it settled (RMS, meters and meters a
/// second).
#[derive(Clone, Copy)]
enum Steady {
    Dry,
    Held {
        level: f32,
        foam: f32,
    },
    Flowing {
        surface: f32,
        velocity: Vec2,
        foam: f32,
        heave: f32,
        rush: f32,
    },
}

impl SteadyCells {
    /// `state`'s settled water under the terrain-local points from `low` to
    /// `high`, its heights above `datum` (world meters).
    fn read(state: &TerrainState, (low, high): (Vec2, Vec2), datum: f32) -> Self {
        let cell = state.settled.cell();
        let pyramid = &state.terrain.pyramid;
        let anchor = state.terrain.origin.y;
        let first = [(low.x / cell).floor() as i64, (low.y / cell).floor() as i64];
        let last = [
            (high.x / cell).floor() as i64,
            (high.y / cell).floor() as i64,
        ];
        let columns = (last[0] - first[0] + 1) as usize;
        let rows = (last[1] - first[1] + 1) as usize;
        let mut cells = Vec::with_capacity(columns * rows);
        for j in first[1]..=last[1] {
            for i in first[0]..=last[0] {
                cells.push(match state.settled.cell_at(pyramid, i, j) {
                    Some(SteadyCell::Held { level, foam }) => Steady::Held {
                        level: level + anchor - datum,
                        foam,
                    },
                    Some(SteadyCell::Flowing {
                        surface,
                        velocity,
                        foam,
                        heave,
                        rush,
                        ..
                    }) => Steady::Flowing {
                        surface: surface + anchor - datum,
                        velocity: Vec2::from(velocity),
                        foam,
                        heave: known(heave),
                        rush: known(rush),
                    },
                    Some(SteadyCell::Dry) | None => Steady::Dry,
                });
            }
        }
        Self {
            first,
            columns,
            rows,
            cell,
            cells,
        }
    }

    /// The settled cell (i, j) of the terrain's lattice; dry past the
    /// rectangle.
    fn get(&self, i: i64, j: i64) -> Steady {
        let (x, z) = (i - self.first[0], j - self.first[1]);
        if x < 0 || z < 0 || x >= self.columns as i64 || z >= self.rows as i64 {
            return Steady::Dry;
        }
        self.cells[z as usize * self.columns + x as usize]
    }

    /// The settled water at terrain-local `point`, the middle of cell `at`
    /// on the terrain's lattice of live cells, through the settled water's
    /// smooth surface: the uniform cubic B-spline whose control points are
    /// the 4 × 4 settled cells around it (`spline`), as water's shading
    /// takes its smooth surface through its samples (`meshes/water.wgsl`'s
    /// `water_smooth`), its surface, velocity, foam and swings each through
    /// the weights of those holding water alone, shared out over them, so it
    /// keeps its level up to where they end, a lake's cells standing at its
    /// level and still; held by a lake where the settled cell holding it is;
    /// dry where none of them holds water. It starts off that steady state
    /// by noise of zero mean hashed at its cell, as far as the settled
    /// water's surface and its speed strayed as it settled there, its
    /// speed's shared by x and z.
    fn smooth(&self, point: Vec2, at: [i64; 2]) -> WaterCell {
        let spot = point / self.cell;
        let holding = spot.floor();
        if let Steady::Held { level, foam } = self.get(holding.x as i64, holding.y as i64) {
            return WaterCell {
                surface: level,
                velocity: [0.0; 2],
                held: level,
                foam,
                unsteady: [0.0; 3],
            };
        }
        // The control points, from the one before the settled middle
        // nearest below the point along x and z, and the point's share of
        // the way between the middle two.
        let spot = spot - 0.5;
        let corner = spot.floor();
        let (along_x, along_z) = (spline(spot.x - corner.x), spline(spot.y - corner.y));
        let first = [corner.x as i64 - 1, corner.y as i64 - 1];
        let (mut total, mut surfaces, mut foams) = (0.0, 0.0, 0.0);
        let (mut velocities, mut swings) = (Vec2::ZERO, Vec2::ZERO);
        for (row, &weight_z) in (0_i64..).zip(&along_z) {
            for (column, &weight_x) in (0_i64..).zip(&along_x) {
                let (level, running, white, swing) =
                    match self.get(first[0] + column, first[1] + row) {
                        Steady::Dry => continue,
                        Steady::Held { level, foam } => (level, Vec2::ZERO, foam, Vec2::ZERO),
                        Steady::Flowing {
                            surface,
                            velocity,
                            foam,
                            heave,
                            rush,
                        } => (surface, velocity, foam, Vec2::new(heave, rush)),
                    };
                let weight = weight_x * weight_z;
                total += weight;
                surfaces += weight * level;
                velocities += weight * running;
                foams += weight * white;
                swings += weight * swing;
            }
        }
        if total <= 0.0 {
            return WaterCell::DRY;
        }
        let share = 1.0 / total;
        let swings = swings * share;
        let across = swings.y * std::f32::consts::FRAC_1_SQRT_2;
        WaterCell {
            surface: surfaces * share,
            velocity: (velocities * share).to_array(),
            held: f32::NAN,
            foam: foams * share,
            unsteady: [
                swings.x * unit_noise(at[0], at[1], 0),
                across * unit_noise(at[0], at[1], 1),
                across * unit_noise(at[0], at[1], 2),
            ],
        }
    }

    /// The settled water of the live cells over settled cell `holding` (on
    /// the terrain's lattice of settled cells), `split` along each side of
    /// it and `cell` meters across, row by row, each over its own ground,
    /// which `bed` gives at its place on the terrain's lattice of live
    /// cells: each through the smooth surface ([`Self::smooth`]); and, where
    /// the settled cell flows, those of them holding water raised or lowered
    /// alike and their velocity turned alike, so their mean surface and
    /// velocity are the settled cell's, or, where none of them would hold
    /// water, each at the settled cell's surface, velocity and foam. Over
    /// each settled cell, the live cells' settled water is that cell's
    /// water, the same water as the settled water around the live window.
    fn matched(
        &self,
        holding: [i64; 2],
        (cell, split): (f32, i64),
        bed: impl Fn(i64, i64) -> f32,
    ) -> Vec<WaterCell> {
        let first = [holding[0] * split, holding[1] * split];
        let count = (split * split) as usize;
        let (mut cells, mut beds) = (Vec::with_capacity(count), Vec::with_capacity(count));
        for j in first[1]..first[1] + split {
            for i in first[0]..first[0] + split {
                let middle = (Vec2::new(i as f32, j as f32) + 0.5) * cell;
                cells.push(self.smooth(middle, [i, j]));
                beds.push(bed(i, j));
            }
        }
        let Steady::Flowing {
            surface,
            velocity,
            foam,
            ..
        } = self.get(holding[0], holding[1])
        else {
            return cells;
        };
        let wet = |water: &WaterCell, ground: f32| water.surface - ground > DRY;
        let (mut wets, mut level, mut running) = (0_u32, 0.0, Vec2::ZERO);
        for (water, &ground) in cells.iter().zip(&beds) {
            if wet(water, ground) {
                wets += 1;
                level += water.surface;
                running += Vec2::from(water.velocity);
            }
        }
        if wets == 0 {
            for water in &mut cells {
                water.surface = surface;
                water.velocity = velocity.to_array();
                water.foam = foam;
            }
            return cells;
        }
        let share = 1.0 / wets as f32;
        let (raise, turn) = (surface - level * share, velocity - running * share);
        for (water, &ground) in cells.iter_mut().zip(&beds) {
            if wet(water, ground) {
                water.surface += raise;
                water.velocity = (Vec2::from(water.velocity) + turn).to_array();
            }
        }
        cells
    }
}

/// The weights the uniform cubic B-spline gives four control points a step
/// apart at a point `t` (0 to 1) of the way from the second to the third,
/// as water's shading takes them (`meshes/water.wgsl`'s `water_spline`):
/// they sum to 1.
fn spline(t: f32) -> [f32; 4] {
    let s = 1.0 - t;
    [
        s * s * s / 6.0,
        ((3.0 * t - 6.0) * t * t + 4.0) / 6.0,
        (((3.0 - 3.0 * t) * t + 3.0) * t + 1.0) / 6.0,
        t * t * t / 6.0,
    ]
}

/// `value`, or none where a record holds none of it (NaN).
fn known(value: f32) -> f32 {
    if value.is_nan() { 0.0 } else { value }
}

/// A value of zero mean and unit variance hashed from the terrain's lattice
/// cell (i, j) and `channel`: uniform over NOISE_REACH either side of zero.
fn unit_noise(i: i64, j: i64, channel: u32) -> f32 {
    let mut h = (i as u32).wrapping_mul(0x9e37_79b9)
        ^ (j as u32).wrapping_mul(0x85eb_ca6b)
        ^ channel.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    (((h >> 8) as f32 + 0.5) / 16_777_216.0 * 2.0 - 1.0) * NOISE_REACH
}
