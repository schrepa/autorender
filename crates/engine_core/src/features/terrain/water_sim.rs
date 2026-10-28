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
//! of cells along each side. Wherever water draws, a window runs.
//!
//! A window is where it stands, and the little the GPU does not hold: its
//! first corner and cells, the meters across them and the live cells along
//! each side of a settled cell, the snap it moves by and the height its
//! heights stand above (a datum, its lowest ground as probed in whole
//! meters, which keeps them small for the GPU's single precision); where its
//! rivers bring water, what the settled cells over each of its cells bring
//! shared by area, so water left alone stays as it settled; the solids
//! standing in the settled water over it, held whole; and the tiles its
//! water is set up from, the finest resident one holding each level-0 tile
//! it and a settled cell past its edges reach ([`WaterRead`]). The renderer
//! sets every cell up on the GPU from what it holds of those tiles, in the
//! frame the window newly covers the cell or the tile under it changes (its
//! `meshes/water_setup.wgsl`): its ground and what its bed holds, the
//! solids' cover by their share of it, as they cover any cells
//! ([`cover_cells`](crate::terrain_data::cover_cells)), its water as the
//! settled water's smooth surface stands there matched to the settled cell
//! holding it, the lakes it holds, and the cells past its edges. So a window
//! runs in the frame it comes: where a tile's settled water is not resident
//! yet, the coarser level shading reads there stands in until the finest
//! arrives, and the cells over it are set up again then.
//!
//! A window follows the detail eye a snap of whole blocks at a time
//! ([`WaterWindow::snap`]): once the eye goes FOLLOW of a snap from its
//! middle along x or z, to the snap nearest the eye, keeping its datum while
//! its ground lies near it. Each frame the solids standing in its square
//! reach the renderer with it, nearest the eye first
//! ([`WaterWindow::touching`]), to push its water.

use std::sync::Arc;

use bevy_ecs::prelude::{DetectChangesMut, Res, ResMut};
use glam::{Vec2, Vec3};

use super::residency::TerrainState;
use super::{TerrainDebug, Terrains, Viewport, Water};
use crate::math::{max, min, sin_cos};
use crate::terrain_data::shallow_water::step_seconds;
use crate::terrain_data::{BLOCK_SIDE, Footprint, NARROWEST, SteadyWater, Tile, TileKey};
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
/// clock; water's particles step no more of it either.
pub const LONGEST_FRAME: f64 = 1.0 / 30.0;
/// Meters a window's first corner snaps to, as the whole number of the
/// scheme's blocks of its cells nearest them, and the share of a snap the
/// eye goes from its middle, along x or z, before it follows: a window
/// reaches that far past the live reach on every side, so the reach never
/// leaves it.
const SNAP: f32 = 32.0;
const FOLLOW: f32 = 0.75;
/// Meters apart a window's ground is probed for its lowest as it is
/// placed: finer than a tile, so none is missed.
const PROBED: f32 = 32.0;
/// Meters a window's lowest ground may lie from the datum of the window
/// before it before it takes its own: heights within a few hundred meters
/// of their datum keep single precision's steps (2^-23 of 512 m is 0.06 mm)
/// under the depth water dries at.
const DATUM_DRIFT: f32 = 256.0;
/// The most solids standing in a window's square that push its water each
/// frame, nearest the eye first: every cell they reach reads them all, so
/// they bound the work of the pass pushing it; those past them push none.
pub const SIM_TOUCHES: usize = 32;

/// The square over which water runs live: its first corner (world x and
/// z) and the cell there on the terrain's lattice of its cells, its cells
/// along each side, a whole number of the scheme's blocks of them, those
/// its live reach asks for, which a frame's budget of their steps cuts it
/// to ([`Water::live_budget`]), the pixels across the view a cell spans, at
/// the view's middle, as far out as it holds the live reach about the eye
/// wherever the eye stands in it, where water's pass hands its water back
/// to the settled water ([`LIVE_PIXELS`] where it holds all the reach asks,
/// more where the budget cuts it), the meters across each cell (a live
/// cell's, `live_cell`'s) and the live cells along each side of a settled
/// cell, the blocks it moves by along x or z as it follows the eye, and the
/// height its heights stand above (world meters, whole: its lowest ground
/// as probed, or the datum of the window before it while that lies near
/// it); where its rivers bring water; the solids standing in the settled
/// water over it (terrain-local, in their records' order), which the
/// renderer covers its cells with; and the level-0 tiles its water is set
/// up from, row by row.
#[derive(Clone, Debug)]
pub struct WaterWindow {
    pub origin: [f32; 2],
    pub first: [i64; 2],
    pub cells: u32,
    pub asked: u32,
    pub pixels: f32,
    pub cell: f32,
    pub split: u32,
    pub snap: u32,
    pub datum: f32,
    pub sources: Arc<[WaterSourceCell]>,
    pub solids: Arc<[Footprint]>,
    pub reads: Arc<[WaterRead]>,
}

/// A level-0 tile a window's water is set up from: its key and its square
/// (terrain-local least and greatest corners), the finest resident tile
/// holding it, whose ground and water the GPU sets the window's cells over
/// it up from (none where no level holds it), and its record of settled
/// water held whole, which the window's sources and solids are found from
/// (none where none is held).
#[derive(Clone, Debug)]
pub struct WaterRead {
    pub key: TileKey,
    pub square: [Vec2; 2],
    pub tile: Option<Arc<Tile>>,
    settled: Option<Arc<SteadyWater>>,
}

impl WaterRead {
    /// Whether the cells over it are set up from the same tile as over
    /// `other`, read by a window before.
    pub fn same_tile(&self, other: &Self) -> bool {
        self.key == other.key && alike(self.tile.as_ref(), other.tile.as_ref())
    }

    /// Whether it reads what `other` reads: the same tile and record.
    fn same(&self, other: &Self) -> bool {
        self.same_tile(other) && alike(self.settled.as_ref(), other.settled.as_ref())
    }
}

/// Whether `a` and `b` are the same one, or both none.
fn alike<T>(a: Option<&Arc<T>>, b: Option<&Arc<T>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
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

    /// The square of its middle `cells` cells along each side, whole blocks
    /// of them (all of it, where it has no more): its sources there, on its
    /// datum, with its solids and set up from its tiles, its cells spanning
    /// as many more pixels where it hands back as it holds less of the
    /// reach.
    #[must_use]
    pub fn within(&self, cells: u32) -> Self {
        let side = self.blocks();
        let kept = (cells / BLOCK_SIDE).clamp(1, side);
        let (shift, n) = ((side - kept) / 2 * BLOCK_SIDE, kept * BLOCK_SIDE);
        // The reach it holds about the eye, the eye FOLLOW of a snap from its
        // middle at most, and the pixels a cell spans there.
        let follows = FOLLOW * self.snap as f32 * BLOCK_SIDE as f32 * self.cell;
        let holds = |cells: u32| max(0.5 * cells as f32 * self.cell - follows, self.cell);
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
            first: [
                self.first[0] + i64::from(shift),
                self.first[1] + i64::from(shift),
            ],
            cells: n,
            pixels: self.pixels * holds(self.cells) / holds(n),
            sources,
            ..self.clone()
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
/// budget of cell steps holds ([`follow`]); drops every other window, as it
/// does where water does not draw.
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
/// its middle and a block more on each side, its rim's: the window stays
/// while the eye stands within FOLLOW of a snap of its middle along x and
/// z, and moves to the snap nearest the eye once it goes farther, or at
/// once where no window of its size runs, on the datum of the window before
/// it while its ground lies near that. A window standing where it stood is
/// made again where the tiles it reads changed (another finest resident
/// tile, or its settled water arrived), its sources and solids found again,
/// and where the cells its reach asks for, or the pixels a cell spans as
/// far out as it holds the reach, changed.
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
    let split = (state.settled.cell() / cell).round() as u32;
    let eye = Vec2::new(eye.x, eye.z);
    let flat = Vec2::new(state.terrain.origin.x, state.terrain.origin.z);
    let before = state
        .hydrology
        .window
        .clone()
        .filter(|window| window.cell == cell && window.split == split);
    let staying = before.as_ref().filter(|window| {
        window.cells == cells && (eye - window.middle()).abs().max_element() <= FOLLOW * snap
    });
    let (first, datum) = if let Some(window) = staying {
        (window.first, window.datum)
    } else {
        // The window's first corner on the terrain's lattice of its cells,
        // at the snap nearest the eye.
        let at = ((eye - flat - Vec2::splat(0.5 * size)) / snap).round();
        let step = snap_blocks as i64 * i64::from(BLOCK_SIDE);
        let first = [at.x as i64 * step, at.y as i64 * step];
        let Some(lowest) = lowest(state, first, (cells, cell)) else {
            return;
        };
        let lowest = lowest.floor();
        let datum = before.as_ref().map_or(lowest, |window| {
            if (lowest - window.datum).abs() <= DATUM_DRIFT {
                window.datum
            } else {
                lowest
            }
        });
        (first, datum)
    };
    let reads = reads(state, first, (cells, cell));
    if let Some(window) = staying {
        let read = window.reads.len() == reads.len()
            && window.reads.iter().zip(&reads).all(|(held, read)| held.same(read));
        if read && (window.asked, window.pixels) == (asked, pixels) {
            return;
        }
    }
    let low = Vec2::new(first[0] as f32, first[1] as f32) * cell;
    state.hydrology.window = Some(Arc::new(WaterWindow {
        origin: (flat + low).to_array(),
        first,
        cells,
        asked,
        pixels,
        cell,
        split,
        snap: snap_blocks as u32,
        datum,
        sources: sources(state, (first, cells, cell)).into(),
        solids: solids(state, (low, low + Vec2::splat(size))).into(),
        reads: reads.into(),
    }));
}

/// The lowest ground (world meters) under the square from the terrain's
/// lattice cell `first` of `cells` cells `cell` meters across along each
/// side, probed PROBED meters apart on the finest tiles `state` holds under
/// each probe; `None` where it holds none under one.
fn lowest(state: &TerrainState, first: [i64; 2], (cells, cell): (u32, f32)) -> Option<f32> {
    let size = cells as f32 * cell;
    let steps = max((size / PROBED).ceil(), 1.0) as u32;
    let apart = size / steps as f32;
    let low = Vec2::new(first[0] as f32, first[1] as f32) * cell;
    let mut lowest = f32::INFINITY;
    for j in 0..=steps {
        for i in 0..=steps {
            let (_, sample) = state.sample(low + Vec2::new(i as f32, j as f32) * apart)?;
            lowest = min(lowest, sample.height);
        }
    }
    Some(state.terrain.origin.y + lowest)
}

/// The level-0 tiles of `state`'s terrain the window from its lattice cell
/// `first` of `cells` cells `cell` meters across along each side is set up
/// from, its square and a settled cell past it on every side, where the
/// cells just past its edges and their settled cells lie, row by row: each
/// with the finest tile `state` holds over it, and its record of settled
/// water held whole.
fn reads(state: &TerrainState, first: [i64; 2], (cells, cell): (u32, f32)) -> Vec<WaterRead> {
    let pyramid = &state.terrain.pyramid;
    let size = pyramid.tile_size(0);
    let [columns, rows] = pyramid.tiles(0);
    let range = |low: f32, high: f32, count: u32| {
        let first = max((low / size).floor(), 0.0) as u32;
        let last = min((high / size).floor(), count as f32 - 1.0) as u32;
        first..=last
    };
    let margin = Vec2::splat(state.settled.cell());
    let low = Vec2::new(first[0] as f32, first[1] as f32) * cell - margin;
    let high = Vec2::new(
        (first[0] + i64::from(cells)) as f32,
        (first[1] + i64::from(cells)) as f32,
    ) * cell
        + margin;
    let mut reads = Vec::new();
    for z in range(low.y, high.y, rows) {
        for x in range(low.x, high.x, columns) {
            let key = TileKey::new(0, x as u16, z as u16);
            let corner = pyramid.tile_origin(key);
            let middle = corner + Vec2::splat(0.5 * size);
            let tile = (0..pyramid.levels()).find_map(|level| {
                let held = pyramid.tile_at(level, middle)?;
                state.store.get(&held).cloned()
            });
            reads.push(WaterRead {
                key,
                square: [corner, corner + Vec2::splat(size)],
                tile,
                settled: state
                    .settled
                    .held
                    .get(&key)
                    .map(|(water, _)| Arc::clone(water)),
            });
        }
    }
    reads
}

/// The solids standing in `state`'s settled water held whole whose
/// footprints reach over the square from `low` to `high` (terrain-local),
/// in their records' order.
fn solids(state: &TerrainState, (low, high): (Vec2, Vec2)) -> Vec<Footprint> {
    state
        .settled
        .held
        .iter()
        .flat_map(|(_, (water, _))| water.solids.iter().copied())
        .filter(|solid| {
            let (at, half) = (solid.middle(), solid.reach());
            (at + half).cmpge(low).all() && (at - half).cmple(high).all()
        })
        .collect()
}

/// Where the rivers of `state`'s settled water bring water to the window
/// from the terrain's lattice cell `first` of `cells` cells `cell` meters
/// across along each side, cells in order: what each settled cell over them
/// brings, shared among the window's cells by the area of it each covers,
/// every cell's summed and the way it comes the rate-weighted mean of what
/// it is brought. Each cell's comes from the tiles under it in their keys'
/// order, so a cell is brought the same in every window covering it.
fn sources(
    state: &TerrainState,
    (first, cells, cell): ([i64; 2], u32, f32),
) -> Vec<WaterSourceCell> {
    let settled = state.settled.cell();
    let pyramid = &state.terrain.pyramid;
    let n = i64::from(cells);
    let area = settled * settled;
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
    merged
        .into_iter()
        .map(|(source, weighted)| {
            let share = if source.rate > 0.0 {
                1.0 / source.rate
            } else {
                0.0
            };
            WaterSourceCell {
                way: [weighted[0] * share, weighted[1] * share],
                ..source
            }
        })
        .collect()
}
