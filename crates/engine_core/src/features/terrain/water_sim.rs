//! Water's simulation windows: the square of ground around the camera over
//! which the renderer runs rivers by the shallow-water equations (its
//! `water_sim`), placed where [`Water::simulated`] asks for one, in cells
//! [`Water::live_cell`] across, a whole number of the scheme's blocks of
//! them along each side: its ground and the roughness of its rivers' beds,
//! the solids standing in its water, the water to start from, where its
//! rivers bring theirs, the water standing just past its edges, and what
//! the beds hold under its water (their reach and span), which it draws
//! with as the settled water does. All of it but the ground and the beds'
//! roughness comes from the terrain's settled water
//! ([`super::settled`]): its solids are those standing in the water of the
//! records held whole, covering its cells as they cover the settled cells
//! ([`cover_cells`]); and the settled water's cells may be coarser or finer
//! than the window's: a window's cell stands on the planes of the surfaces
//! of the settled cells around it, blended by their nearness, and runs at
//! their velocity, its depth its own over its own ground; a lake holds it
//! where it holds the settled cell under its
//! middle; it starts off that steady state as unsteadily as the water
//! settled there; and its rivers bring what the settled cells it covers
//! bring, shared by the area of each it covers. The window starts with
//! water as it runs at its steady state, flowing and foaming, the cells
//! past its edges give and take water as the steady state does, and its
//! rivers gain what they drain inside it, so water left alone stays as it
//! settled. Its heights stand above a datum, its lowest ground as probed in
//! whole meters, which keeps them small for the GPU's single precision.
//!
//! A window holds its cells by the scheme's blocks of them, each made from
//! its place on the terrain's lattice of blocks, the window's cells' size
//! and datum and what the terrain holds there alone, so whichever way the
//! camera came a window holds what one gathered at once would. It follows
//! the camera a snap of whole blocks at a time, keeping its datum while its
//! ground lies near it: the next window takes over the blocks the window
//! before it holds and gathers the strips the move uncovered, GATHERED
//! cells a frame while the window before it keeps running, then the cells
//! just past its edges and its rivers' sources, taken over too where the
//! window before it held them. It waits for its ground's finest tiles and
//! the settled water of the tiles it and the cells past its edges read.
//! Each frame the solids standing in its square reach the renderer with it,
//! nearest the eye first ([`WaterWindow::touching`]), to push its water.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy_ecs::prelude::{DetectChangesMut, Res, ResMut};
use glam::{Vec2, Vec3};

use super::residency::TerrainState;
use super::{Terrains, Water};
use crate::math::{clamp, max, min};
use crate::terrain_data::shallow_water::{DRY, roughness};
use crate::terrain_data::{BLOCK_CELLS, BLOCK_SIDE, Footprint, SteadyCell, TileKey, cover_cells};
use crate::{Camera, WaterObstacle};

/// Meters a window's first corner snaps to, as the whole number of the
/// scheme's blocks of its cells nearest them, and the share of a snap the
/// camera goes from its middle, along x or z, before it follows.
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
/// Cells a frame gathers at most for the window the camera asks for while
/// the window before it keeps running: a millisecond or two of the main
/// thread's. A move of a snap along both axes of a window 512 meters across
/// in meter cells uncovers some 32,000 cells, gathered over 8 frames as the
/// camera crosses a few meters.
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

/// The square over which water is simulated: its first corner (world x and
/// z) and the terrain's lattice cell there, its cells along each side, a
/// whole number of the scheme's blocks of them, and the meters across each,
/// and the height its heights stand above (world meters, whole: its lowest
/// ground as probed, or the datum of the window before it while that lies
/// near it); its blocks of cells, row by row, each as any window holding it
/// gathers it (`WaterBlock`); the cells just past its edges, each as the
/// settled water around it stands there (along its south edge, its north,
/// its west and its east, each from its first corner); and where its rivers
/// bring water.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterWindow {
    pub origin: [f32; 2],
    pub first: [i64; 2],
    pub cells: u32,
    pub cell: f32,
    pub datum: f32,
    blocks: Arc<[Arc<WaterBlock>]>,
    ring: Arc<[WaterCell]>,
    pub sources: Arc<[WaterSourceCell]>,
}

/// A block of a window's cells, [`BLOCK_SIDE`] along each side, as any
/// window holding it gathers it: the stamp it took, which no other block
/// holds; the ground at its cells' corners over the datum of the windows
/// holding it (`BLOCK_SIDE + 1` a row, rows along z); and, cells row by row,
/// each one's water as the settled water around it stands there, the
/// roughness of its ground as the scheme counts it (the D84 of the river bed
/// under it in meters, negated where steps or boulders are carved into the
/// bed, whose friction then counts its grains alone, and a hillside's
/// stones and turf outside rivers' channels), what the bed holds under its
/// middle as the settled water's sample nearest it holds it (its reach's
/// code beside its span's, [`crate::terrain_data::Tile::bed_word`]) and the
/// share of it covered by the solids standing in the settled water, with the
/// height their tops stand at over the datum ([`cover_cells`]; 0 each where
/// none covers it).
#[derive(Clone, Debug, PartialEq)]
pub struct WaterBlock {
    stamp: u64,
    corners: [f32; CORNERS],
    water: [WaterCell; BLOCK_CELLS],
    roughness: [f32; BLOCK_CELLS],
    bed_words: [u32; BLOCK_CELLS],
    cover: [[f32; 2]; BLOCK_CELLS],
}

/// A cell of a window, or one just past its edges, as the settled water
/// around it stands there: the surface it stands at (meters above the
/// window's datum; NaN where dry), its velocity along x and z (meters a
/// second), the level a lake holds it at (NaN where none does) and its
/// foam; and how far it starts off that steady state, as unsteadily as the
/// water settled there: its surface (meters) and its velocity along x and z
/// (meters a second), each of zero mean.
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

    /// The level a lake holds its cell (x, z) at, where it stands over the
    /// cell's ground; NaN where none does.
    pub fn held(&self, x: u32, z: u32) -> f32 {
        let level = self.water(x, z).held;
        if level > self.bed(x, z) {
            level
        } else {
            f32::NAN
        }
    }

    /// What its cell (x, z) starts from: its surface, its flow along x and
    /// z (square meters a second) and its foam. A cell a lake holds stands
    /// still at its level; one holding flowing water starts off its steady
    /// state by its unsteadiness, its surface by at most half its depth so
    /// it stays wet, its flow its depth over its ground at its velocity; a
    /// dry one stands at its ground.
    pub fn start(&self, x: u32, z: u32) -> [f32; 4] {
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
        let [heave, along_x, along_z] = water.unsteady;
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
    /// its sources there, on its datum.
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
            cell: self.cell,
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
    /// The squares around the camera over which terrains' water runs live:
    /// each one's least and greatest corners (world x and z).
    pub fn water_windows(&self) -> impl Iterator<Item = (Vec2, Vec2)> + '_ {
        self.entries
            .iter()
            .filter_map(|state| state.hydrology.window.as_deref())
            .map(WaterWindow::square)
    }
}

/// Places a window around the camera over the terrain holding it where
/// water asks for one ([`follow`]); drops every window, and every one being
/// gathered, where water asks for none.
pub(crate) fn place_water_windows(
    mut terrains: ResMut<Terrains>,
    camera: Res<Camera>,
    water: Option<Res<Water>>,
) {
    // Windows are read, not watched: placing them changes nothing
    // terrains' observers wait on.
    let terrains = terrains.bypass_change_detection();
    let asked = water.filter(|water| water.enabled && water.simulated > 0.0);
    let wanted = asked.as_ref().and_then(|_| {
        terrains
            .entries
            .iter()
            .position(|state| state.around.is_none() && state.baked.is_some())
    });
    for (index, state) in terrains.entries.iter_mut().enumerate() {
        match &asked {
            Some(water) if Some(index) == wanted => follow(state, water, camera.position),
            _ => {
                state.hydrology.window = None;
                state.hydrology.gathering = None;
            }
        }
    }
}

/// Follows the camera at `eye` with `state`'s window, in cells
/// [`Water::live_cell`] across, as many along each side as fill
/// [`Water::simulated`] with whole blocks of them: once the camera has gone
/// FOLLOW of a snap from the window's middle along x or z, the window at
/// the snap nearest the camera is gathered, GATHERED cells a frame, taking
/// over what the window before it and the gathering before this one hold on
/// its datum, and takes the window's place once whole.
fn follow(state: &mut TerrainState, water: &Water, eye: Vec3) {
    let cell = water.live_cell;
    let side = BLOCK_SIDE as f32;
    let cells = max((water.simulated / (cell * side)).round(), 1.0) as u32 * BLOCK_SIDE;
    let snap_blocks = max((SNAP / (cell * side)).round(), 1.0);
    let snap = snap_blocks * side * cell;
    let size = cells as f32 * cell;
    let eye = Vec2::new(eye.x, eye.z);
    let before = state
        .hydrology
        .window
        .clone()
        .filter(|window| window.cell == cell);
    let current = before.as_ref().filter(|window| window.cells == cells);
    let off = current.map(|window| (eye - window.middle()).abs());
    if off.is_some_and(|off| off.max_element() <= FOLLOW * snap) {
        state.hydrology.gathering = None;
        return;
    }
    // The window's first corner on the terrain's lattice of its cells, at
    // the snap nearest the camera.
    let flat = Vec2::new(state.terrain.origin.x, state.terrain.origin.z);
    let at = ((eye - flat - Vec2::splat(0.5 * size)) / snap).round();
    let step = snap_blocks as i64 * i64::from(BLOCK_SIDE);
    let first = [at.x as i64 * step, at.y as i64 * step];
    if current.is_some_and(|window| window.first == first) {
        return;
    }
    let Some(lowest) = finest(state, first, (cells, cell)) else {
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
    let origin = flat + Vec2::new(first[0] as f32, first[1] as f32) * cell;
    let mut gathering = match state.hydrology.gathering.take() {
        Some(gathering) if gathering.gathers(first, (cells, cell), datum) => gathering,
        earlier => {
            let earlier = earlier.filter(|earlier| earlier.cell == cell && earlier.datum == datum);
            Gathering::new((origin, first), (cells, cell), datum, |at| {
                before
                    .as_ref()
                    .and_then(|window| window.block_on(at))
                    .or_else(|| earlier.as_ref().and_then(|earlier| earlier.block_on(at)))
                    .cloned()
            })
        }
    };
    if gathering.gather(state, GATHERED) {
        let window = gathering.finish(state, before.as_deref());
        state.hydrology.window = Some(Arc::new(window));
    } else {
        state.hydrology.gathering = Some(gathering);
    }
}

/// A window being gathered for the square the camera asks for, a few of its
/// blocks a frame while the window before it keeps running: its first
/// corner (world x and z) and the terrain's lattice cell there, its cells
/// along each side and the meters across each, its datum, and its blocks,
/// row by row, each once gathered or taken over.
pub(super) struct Gathering {
    origin: Vec2,
    first: [i64; 2],
    cells: u32,
    cell: f32,
    datum: f32,
    blocks: Vec<Option<Arc<WaterBlock>>>,
}

impl Gathering {
    /// The window from `origin` (world x and z), the terrain's lattice cell
    /// `first`, of `cells` cells `cell` meters across along each side, on
    /// `datum`: each of its blocks `carried` holds at its place on the
    /// terrain's lattice of blocks taken over, the rest still to gather.
    fn new(
        (origin, first): (Vec2, [i64; 2]),
        (cells, cell): (u32, f32),
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
    /// where it does not.
    fn finish(self, state: &TerrainState, before: Option<&WaterWindow>) -> WaterWindow {
        let n = self.cells as usize;
        let around: Vec<[i64; 2]> = ring_cells(self.first, self.cells).collect();
        let mut ring: Vec<Option<WaterCell>> = around
            .iter()
            .map(|&at| before.and_then(|window| window.water_on(at)))
            .collect();
        // Along each edge, the stretch of the cells the window before it
        // does not hold is read.
        let middle = |[i, j]: [i64; 2]| (Vec2::new(i as f32, j as f32) + 0.5) * self.cell;
        for edge in 0..4 {
            let line = edge * n..(edge + 1) * n;
            let fresh: Vec<usize> = line.filter(|&at| ring[at].is_none()).collect();
            let (Some(&low), Some(&high)) = (fresh.first(), fresh.last()) else {
                continue;
            };
            let ends = (middle(around[low]), middle(around[high]));
            let steady = SteadyCells::read(state, ends, self.datum);
            for at in fresh {
                ring[at] = Some(steady.at(middle(around[at])));
            }
        }
        let sources = sources(state, (self.first, self.cells, self.cell), before);
        WaterWindow {
            origin: self.origin.to_array(),
            first: self.first,
            cells: self.cells,
            cell: self.cell,
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
/// side, probed PROBED meters apart, where `state`'s finest tiles stand
/// under every probe; `None` where they do not.
fn finest(state: &TerrainState, first: [i64; 2], (cells, cell): (u32, f32)) -> Option<f32> {
    let size = cells as f32 * cell;
    let steps = max((size / PROBED).ceil(), 1.0) as u32;
    let apart = size / steps as f32;
    let low = Vec2::new(first[0] as f32, first[1] as f32) * cell;
    let mut lowest = f32::INFINITY;
    for j in 0..=steps {
        for i in 0..=steps {
            let (level, sample) = state.sample(low + Vec2::new(i as f32, j as f32) * apart)?;
            if level != 0 {
                return None;
            }
            lowest = min(lowest, sample.height);
        }
    }
    Some(state.terrain.origin.y + lowest)
}

/// Whether `state` holds the settled water the window from the terrain's
/// lattice cell `first` of cells `cell` meters across, over `cells` of them
/// along each side, reads: that of every level-0 tile its bake's water
/// reaches within two settled cells of the window and the cells just past
/// its edges, which their cells blend from and their slopes read.
fn settled(state: &TerrainState, first: [i64; 2], (cells, cell): (u32, f32)) -> bool {
    let pyramid = &state.terrain.pyramid;
    let size = pyramid.tile_size(0);
    let reach = Vec2::splat(cell + 2.0 * state.settled.cell());
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
/// `columns`, in cells `cell` meters across, as `state`'s resident tiles
/// and settled water stand there, their heights over `datum` (world
/// meters): the ground at their corners from the finest tiles holding them,
/// their cells' water from the settled water around them, their ground's
/// roughness and what their beds hold from the beds of the finest tiles
/// under their middles and those tiles' settled water there, and what the
/// solids standing in the settled water held whole cover of their cells.
fn gather_run(
    state: &TerrainState,
    (cell, datum): (f32, f32),
    row: i64,
    columns: Range<i64>,
) -> Vec<WaterBlock> {
    let side = i64::from(BLOCK_SIDE);
    let anchor = state.terrain.origin.y;
    let pyramid = &state.terrain.pyramid;
    // Its cells, from `low` up to before `high` on the terrain's lattice.
    let low = [columns.start * side, row * side];
    let high = [columns.end * side, (row + 1) * side];
    let middle = |i: i64, j: i64| (Vec2::new(i as f32, j as f32) + 0.5) * cell;
    // The ground at its corners, rows along z.
    let stride = (high[0] - low[0] + 1) as usize;
    let mut ground = Vec::with_capacity(stride * (SIDE + 1));
    for j in low[1]..=high[1] {
        for i in low[0]..=high[0] {
            let point = Vec2::new(i as f32, j as f32) * cell;
            let height = state
                .sample(point)
                .map_or(anchor, |(_, sample)| anchor + sample.height);
            ground.push(height - datum);
        }
    }
    let ends = (middle(low[0], low[1]), middle(high[0] - 1, high[1] - 1));
    let steady = SteadyCells::read(state, ends, datum);
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
    let across = stride - 1;
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
            WaterBlock {
                stamp: STAMPS.fetch_add(1, Ordering::Relaxed) + 1,
                corners: std::array::from_fn(|k| {
                    ground[k / (SIDE + 1) * stride + corner + k % (SIDE + 1)]
                }),
                water: std::array::from_fn(|k| steady.at(at(k))),
                roughness: beds.map(|(rough, _)| rough),
                bed_words: beds.map(|(_, word)| word),
                cover: std::array::from_fn(|k| covers[k / SIDE * across + corner + k % SIDE]),
            }
        })
        .collect()
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
/// with its foam; or flowing, its water standing on the plane through its
/// surface at its middle with its surface's slope along x and z, running at
/// its velocity, with its foam and how far its surface and its velocity
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
        slope: Vec2,
        velocity: Vec2,
        foam: f32,
        heave: f32,
        rush: f32,
    },
}

impl SteadyCells {
    /// `state`'s settled water around the terrain-local points from `low`
    /// to `high`: the settled cells they blend from and those beside them,
    /// which their slopes read; its heights above `datum` (world meters).
    /// What a point within reads of it is what a read of any rectangle
    /// holding the point reads.
    fn read(state: &TerrainState, (low, high): (Vec2, Vec2), datum: f32) -> Self {
        let cell = state.settled.cell();
        let pyramid = &state.terrain.pyramid;
        let anchor = state.terrain.origin.y;
        let first = [
            (low.x / cell - 0.5).floor() as i64 - 1,
            (low.y / cell - 0.5).floor() as i64 - 1,
        ];
        let last = [
            (high.x / cell - 0.5).floor() as i64 + 2,
            (high.y / cell - 0.5).floor() as i64 + 2,
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
                        slope: Vec2::ZERO,
                        velocity: Vec2::from(velocity),
                        foam,
                        heave: known(heave),
                        rush: known(rush),
                    },
                    Some(SteadyCell::Dry) | None => Steady::Dry,
                });
            }
        }
        // Each flowing cell's surface's slope from the levels its wet
        // neighbors stand at.
        let level = |x: i64, z: i64| {
            let inside = x >= 0 && z >= 0 && (x as usize) < columns && (z as usize) < rows;
            match inside.then(|| cells[z as usize * columns + x as usize]) {
                Some(Steady::Held { level, .. }) => Some(level),
                Some(Steady::Flowing { surface, .. }) => Some(surface),
                Some(Steady::Dry) | None => None,
            }
        };
        let slopes: Vec<Vec2> = (0..columns * rows)
            .map(|at| {
                let Steady::Flowing { surface, .. } = cells[at] else {
                    return Vec2::ZERO;
                };
                let (x, z) = ((at % columns) as i64, (at / columns) as i64);
                Vec2::new(
                    surface_slope((level(x - 1, z), surface, level(x + 1, z)), cell),
                    surface_slope((level(x, z - 1), surface, level(x, z + 1)), cell),
                )
            })
            .collect();
        for (steady, found) in cells.iter_mut().zip(slopes) {
            if let Steady::Flowing { slope, .. } = steady {
                *slope = found;
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

    /// The water at terrain-local `point` as the four settled cells around
    /// it stand there, each blending in by its nearness: their surfaces'
    /// planes there (a lake's level, flat), blended over those holding
    /// water, dry where none does; the mean velocity of those flowing; and
    /// their foam. A lake holds it where it holds the settled cell under
    /// it. It starts off that by noise of zero mean hashed at the cells'
    /// middles and blended keeping its variance, as far as its surface and
    /// its speed strayed as it settled.
    fn at(&self, point: Vec2) -> WaterCell {
        let cell = self.cell;
        let u = point / cell - 0.5;
        let base = [u.x.floor() as i64, u.y.floor() as i64];
        let (fu, fv) = (u.x - base[0] as f32, u.y - base[1] as f32);
        let (mut weight, mut level, mut foam, mut moving) = (0.0, 0.0, 0.0, 0.0);
        let mut velocity = Vec2::ZERO;
        let mut strayed = [0.0_f32; 2];
        let mut noise = [0.0_f32; 3];
        let mut squares = 0.0;
        for (dx, dz, share) in [
            (0, 0, (1.0 - fu) * (1.0 - fv)),
            (1, 0, fu * (1.0 - fv)),
            (0, 1, (1.0 - fu) * fv),
            (1, 1, fu * fv),
        ] {
            let (i, j) = (base[0] + dx, base[1] + dz);
            squares += share * share;
            for (channel, value) in noise.iter_mut().enumerate() {
                *value += share * unit_noise(i, j, channel as u32);
            }
            match self.get(i, j) {
                Steady::Dry => {}
                Steady::Held {
                    level: held,
                    foam: white,
                } => {
                    weight += share;
                    level += held * share;
                    foam += white * share;
                }
                Steady::Flowing {
                    surface,
                    slope,
                    velocity: running,
                    foam: white,
                    heave,
                    rush,
                } => {
                    let middle = (Vec2::new(i as f32, j as f32) + 0.5) * cell;
                    weight += share;
                    level += (surface + slope.dot(point - middle)) * share;
                    foam += white * share;
                    moving += share;
                    velocity += running * share;
                    strayed = [strayed[0] + heave * share, strayed[1] + rush * share];
                }
            }
        }
        if weight <= 0.0 {
            return WaterCell::DRY;
        }
        let foam = foam / weight;
        let under = self.get(
            (point.x / cell).floor() as i64,
            (point.y / cell).floor() as i64,
        );
        if let Steady::Held { level, .. } = under {
            return WaterCell {
                surface: level,
                velocity: [0.0; 2],
                held: level,
                foam,
                unsteady: [0.0; 3],
            };
        }
        let moving = max(moving, f32::MIN_POSITIVE);
        let [heave, rush] = strayed.map(|value| value / moving);
        // The noise at unit variance, its speed's shared by x and z.
        let spread = 1.0 / squares.sqrt();
        let across = rush * std::f32::consts::FRAC_1_SQRT_2 * spread;
        WaterCell {
            surface: level / weight,
            velocity: (velocity / moving).to_array(),
            held: f32::NAN,
            foam,
            unsteady: [
                heave * spread * noise[0],
                across * noise[1],
                across * noise[2],
            ],
        }
    }
}

/// The slope along an axis of a surface standing at `here`, its neighbors
/// before and after it along the axis standing at `before` and `after`
/// where they hold water, `apart` meters away: their central difference,
/// one-sided where only one holds water, flat where neither does.
fn surface_slope((before, here, after): (Option<f32>, f32, Option<f32>), apart: f32) -> f32 {
    match (before, after) {
        (Some(before), Some(after)) => (after - before) / (2.0 * apart),
        (Some(before), None) => (here - before) / apart,
        (None, Some(after)) => (after - here) / apart,
        (None, None) => 0.0,
    }
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
