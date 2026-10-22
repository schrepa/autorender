//! Water's simulation windows: the square of ground around the camera over
//! which the renderer runs rivers by the shallow-water equations (its
//! `water_sim`), placed where [`Water::simulated`] asks for one, in cells
//! [`Water::live_cell`] across: its ground and the roughness of its rivers'
//! beds, the water to start from, where its rivers bring theirs, the water
//! standing just past its edges, and what the beds hold under its water
//! (their reach and nearest obstacle), which it draws with as the settled
//! water does. All of it but the ground and the beds' roughness comes from
//! the terrain's settled water ([`super::settled`]), whose
//! cells may be coarser or finer than the window's: a window's cell stands
//! on the planes of the surfaces of the settled cells around it, blended by
//! their nearness, and runs at their velocity, its depth its own over its
//! own ground; a lake holds it where it holds the settled cell under its
//! middle; it starts off that steady state as unsteadily as the water
//! settled there; and its rivers bring what the settled cells it covers
//! bring, shared by the area of each it covers. The window starts with
//! water as it runs at its steady state, flowing and foaming, the cells
//! past its edges give and take water as the steady state does, and its
//! rivers gain what they drain inside it, so water left alone stays as it
//! settled. Its heights stand above a datum, its lowest ground in whole
//! meters, which keeps them small for the GPU's single precision. A window
//! follows the camera, carrying over what it already covers, and its datum
//! while its ground stays near it, and reading the rest anew, and waits for
//! its ground's finest tiles and the settled water of the tiles it and the
//! cells past its edges read.

use std::sync::Arc;

use bevy_ecs::prelude::{Res, ResMut};
use glam::Vec2;

use super::residency::TerrainState;
use super::{Terrains, Water};
use crate::Camera;
use crate::math::{clamp, max, min};
use crate::terrain_data::shallow_water::DRY;
use crate::terrain_data::{Bed, SteadyCell, TileKey};

/// Meters a window's first corner snaps to, and the share of them the
/// camera goes from its middle, along x or z, before it follows.
const SNAP: f32 = 32.0;
const FOLLOW: f32 = 0.75;
/// Meters apart a window's ground is checked for its finest tiles before it
/// is gathered: finer than a tile, so none is missed.
const PROBED: f32 = 32.0;
/// What a cell dry past a window's edge stands at: below any ground.
pub const GHOST_DRY: f32 = -1.0e30;
/// Meters a window's lowest ground may lie from the datum of the window it
/// carries over from before it takes its own: heights within a few hundred
/// meters of their datum keep single precision's steps (2^-23 of 512 m is
/// 0.06 mm) under the depth water dries at.
const DATUM_DRIFT: f32 = 256.0;
/// The roughness ground outside rivers' channels holds, as the settled
/// water's own ground holds it: a D84 of 5 centimeters, a hillside's
/// stones and turf.
const OPEN_ROUGHNESS: f32 = 0.05;
/// Uniform noise over this either side of zero has unit variance: the
/// square root of 3.
const NOISE_REACH: f32 = 1.732_050_8;

/// The square over which water is simulated: its first corner (world x and
/// z), its cells along each side and the meters across each, and the height
/// its heights stand above (world meters, whole: its lowest ground, or the
/// datum of the window it carried over from while its own lies near that);
/// the ground at its cells' corners (`cells + 1` a row, rows along z); its
/// cells' water and that of the cells just past its edges, each as the
/// settled water around it stands there (`cells + 2` a row from the cell
/// past its first corner, rows along z); the roughness of each of its
/// cells' ground, row by row: the D84 of the river bed under it in meters,
/// negated where steps or boulders are carved into the bed (whose friction
/// then counts its grains alone), and 5 centimeters outside rivers'
/// channels; what the bed holds under each cell's middle, row by row, as the
/// settled water's sample nearest it holds it: its reach's code and its
/// nearest obstacle, offset from the cell's middle
/// ([`crate::terrain_data::Tile::bed_words`]); and where its rivers bring
/// water.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterWindow {
    pub origin: [f32; 2],
    pub cells: u32,
    pub cell: f32,
    pub datum: f32,
    pub corners: Arc<[f32]>,
    water: Arc<[WaterCell]>,
    pub roughness: Arc<[f32]>,
    pub bed_words: Arc<[[u32; 2]]>,
    pub sources: Arc<[WaterSourceCell]>,
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

    /// It, its heights `lift` meters higher.
    fn lifted(self, lift: f32) -> Self {
        Self {
            surface: self.surface + lift,
            held: self.held + lift,
            ..self
        }
    }
}

impl WaterWindow {
    /// Its middle (world x and z).
    pub fn middle(&self) -> Vec2 {
        Vec2::from(self.origin) + Vec2::splat(0.5 * self.cells as f32 * self.cell)
    }

    /// The water of its cell (i, j), or of one just past its edges for -1
    /// or `cells`.
    fn water_at(&self, i: i64, j: i64) -> WaterCell {
        let side = i64::from(self.cells) + 2;
        self.water[((j + 1) * side + i + 1) as usize]
    }

    /// The ground at its cell (i, j)'s middle: its corners' mean.
    fn bed(&self, i: u32, j: u32) -> f32 {
        let row = self.cells as usize + 1;
        let at = j as usize * row + i as usize;
        let corners = &self.corners;
        0.25 * (corners[at] + corners[at + 1] + corners[at + row] + corners[at + row + 1])
    }

    /// The level a lake holds its cell (i, j) at, where it stands over the
    /// cell's ground; NaN where none does.
    pub fn held(&self, i: u32, j: u32) -> f32 {
        let level = self.water_at(i64::from(i), i64::from(j)).held;
        if level > self.bed(i, j) {
            level
        } else {
            f32::NAN
        }
    }

    /// What its cell (i, j) starts from: its surface, its flow along x and
    /// z (square meters a second) and its foam. A cell a lake holds stands
    /// still at its level; one holding flowing water starts off its steady
    /// state by its unsteadiness, its surface by at most half its depth so
    /// it stays wet, its flow its depth over its ground at its velocity; a
    /// dry one stands at its ground.
    pub fn start(&self, i: u32, j: u32) -> [f32; 4] {
        let water = self.water_at(i64::from(i), i64::from(j));
        let level = self.held(i, j);
        if !level.is_nan() {
            return [level, 0.0, 0.0, water.foam];
        }
        let bed = self.bed(i, j);
        let depth = water.surface - bed;
        if depth.is_nan() || depth <= DRY {
            return [bed, 0.0, 0.0, 0.0];
        }
        let [heave, along_x, along_z] = water.unsteady;
        let depth = depth + clamp(heave, -0.5 * depth, 0.5 * depth);
        let velocity = Vec2::from(water.velocity) + Vec2::new(along_x, along_z);
        [bed + depth, depth * velocity.x, depth * velocity.y, water.foam]
    }

    /// The cells just past its edges as their water stands
    /// ([`WaterGhost`]): south, then north, of each column, then west, then
    /// east, of each row. A flowing one's flow is its depth over the ground
    /// at the face it borders times its velocity.
    pub fn ghosts(&self) -> Vec<WaterGhost> {
        let n = self.cells as usize;
        let corner = |i: usize, j: usize| self.corners[j * (n + 1) + i];
        let ghost = |(i, j): (i64, i64), ground: f32, inward: Vec2| {
            let water = self.water_at(i, j);
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
        let last = n as i64;
        let mut ghosts = Vec::with_capacity(4 * n);
        for k in 0..n {
            let column = k as i64;
            ghosts.push(ghost(
                (column, -1),
                0.5 * (corner(k, 0) + corner(k + 1, 0)),
                Vec2::Y,
            ));
            ghosts.push(ghost(
                (column, last),
                0.5 * (corner(k, n) + corner(k + 1, n)),
                Vec2::NEG_Y,
            ));
        }
        for k in 0..n {
            let row = k as i64;
            ghosts.push(ghost(
                (-1, row),
                0.5 * (corner(0, k) + corner(0, k + 1)),
                Vec2::X,
            ));
            ghosts.push(ghost(
                (last, row),
                0.5 * (corner(n, k) + corner(n, k + 1)),
                Vec2::NEG_X,
            ));
        }
        ghosts
    }

    /// The square of its middle `cells` cells along each side (all of it,
    /// where it has no more): its ground, water, roughness, beds' words and
    /// sources there, the cells just past that square's edges standing as it
    /// holds them, on its datum.
    pub fn within(&self, cells: u32) -> Self {
        let n = self.cells as usize;
        let m = cells.min(self.cells) as usize;
        let off = (n - m) / 2;
        let cut = |count: usize, stride: usize| {
            (0..count).flat_map(move |j| (0..count).map(move |i| (j + off) * stride + i + off))
        };
        let sources = self
            .sources
            .iter()
            .filter_map(|source| {
                let (i, j) = (source.cell as usize % n, source.cell as usize / n);
                let inside = i >= off && j >= off && i < off + m && j < off + m;
                inside.then(|| WaterSourceCell {
                    cell: ((j - off) * m + i - off) as u32,
                    ..*source
                })
            })
            .collect();
        Self {
            origin: (Vec2::from(self.origin) + Vec2::splat(off as f32 * self.cell)).to_array(),
            cells: m as u32,
            cell: self.cell,
            datum: self.datum,
            corners: cut(m + 1, n + 1).map(|at| self.corners[at]).collect(),
            water: cut(m + 2, n + 2).map(|at| self.water[at]).collect(),
            roughness: cut(m, n).map(|at| self.roughness[at]).collect(),
            bed_words: cut(m, n).map(|at| self.bed_words[at]).collect(),
            sources,
        }
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

/// Places a window around the camera over the terrain holding it where
/// water asks for one, in cells [`Water::live_cell`] across, following the
/// camera SNAP meters at a time once it has gone FOLLOW of that from the
/// window's middle along x or z; drops every window where water asks for
/// none.
pub(crate) fn place_water_windows(
    mut terrains: ResMut<Terrains>,
    camera: Res<Camera>,
    water: Option<Res<Water>>,
) {
    let asked = water.filter(|water| water.enabled && water.simulated > 0.0);
    let Some(water) = asked else {
        for state in &mut terrains.entries {
            state.hydrology.window = None;
        }
        return;
    };
    let cell = water.live_cell;
    let cells = max((water.simulated / cell).round(), 8.0) as u32;
    let size = cells as f32 * cell;
    // The window's corner on the terrain's lattice of its cells, near a
    // multiple of SNAP.
    let snap = max((SNAP / cell).round(), 1.0) * cell;
    let eye = Vec2::new(camera.position.x, camera.position.z);
    let wanted = terrains
        .entries
        .iter()
        .position(|state| state.around.is_none() && state.baked.is_some());
    for index in 0..terrains.entries.len() {
        if Some(index) != wanted {
            terrains.entries[index].hydrology.window = None;
            continue;
        }
        let state = &terrains.entries[index];
        let current = state
            .hydrology
            .window
            .clone()
            .filter(|window| window.cells == cells && window.cell == cell);
        let off = current.as_ref().map(|window| (eye - window.middle()).abs());
        if off.is_some_and(|off| off.max_element() <= FOLLOW * snap) {
            continue;
        }
        let flat = Vec2::new(state.terrain.origin.x, state.terrain.origin.z);
        let first = ((eye - flat - Vec2::splat(0.5 * size)) / snap).round() * snap / cell;
        let first = [first.x.round() as i64, first.y.round() as i64];
        let origin = flat + Vec2::new(first[0] as f32, first[1] as f32) * cell;
        if !finest(state, origin, size) || !settled(state, first, (cells, cell)) {
            continue;
        }
        let window = gather(state, (origin, first), (cells, cell), current.as_deref());
        terrains.entries[index].hydrology.window = Some(Arc::new(window));
    }
}

/// Whether `state`'s finest tiles stand under the square from `origin`
/// (world x and z) `size` meters across, checked PROBED meters apart.
fn finest(state: &TerrainState, origin: Vec2, size: f32) -> bool {
    let anchor = Vec2::new(state.terrain.origin.x, state.terrain.origin.z);
    let steps = max((size / PROBED).ceil(), 1.0) as u32;
    let apart = size / steps as f32;
    (0..=steps).all(|j| {
        (0..=steps).all(|i| {
            let point = origin + Vec2::new(i as f32, j as f32) * apart - anchor;
            state.sample(point).is_some_and(|(level, _)| level == 0)
        })
    })
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

/// The window from `origin` (world x and z), the terrain's lattice cell
/// `first` of cells `cell` meters across, over `cells` of them along each
/// side, as `state`'s resident tiles and settled water stand there: what
/// `previous` covers carried over from it, its heights onto this window's
/// datum, and the rest read anew.
fn gather(
    state: &TerrainState,
    (origin, first): (Vec2, [i64; 2]),
    (cells, cell): (u32, f32),
    previous: Option<&WaterWindow>,
) -> WaterWindow {
    let anchor = state.terrain.origin;
    let flat = Vec2::new(anchor.x, anchor.z);
    let pyramid = &state.terrain.pyramid;
    let n = cells as usize;
    // Where `previous` holds the same points: its cells counted from this
    // window's, a whole number of them apart.
    let shift = previous.and_then(|previous| {
        let offset = (origin - Vec2::from(previous.origin)) / cell;
        let whole = offset.round();
        ((offset - whole).abs().max_element() < 1.0e-3).then(|| whole.as_ivec2())
    });
    let previous = previous.filter(|_| shift.is_some());
    // Where `previous` holds point (i, j) of a grid of `count` along each
    // side laid as its own is.
    let carried = |i: usize, j: usize, count: usize| {
        let from = glam::IVec2::new(i as i32, j as i32) + shift?;
        let inside = from.cmpge(glam::IVec2::ZERO).all()
            && from.cmplt(glam::IVec2::splat(count as i32)).all();
        inside.then(|| from.y as usize * count + from.x as usize)
    };
    // The ground at its corners (world meters), and the datum its heights
    // stand above: its lowest ground in whole meters, or `previous`'s while
    // that lies near it.
    let mut ground = Vec::with_capacity((n + 1) * (n + 1));
    for j in 0..=n {
        for i in 0..=n {
            ground.push(match (previous, carried(i, j, n + 1)) {
                (Some(previous), Some(at)) => previous.corners[at] + previous.datum,
                _ => {
                    let point = origin + Vec2::new(i as f32, j as f32) * cell - flat;
                    state
                        .sample(point)
                        .map_or(anchor.y, |(_, sample)| anchor.y + sample.height)
                }
            });
        }
    }
    let lowest = ground
        .iter()
        .fold(f32::INFINITY, |low, &height| min(low, height))
        .floor();
    let datum = previous.map_or(lowest, |previous| {
        if (lowest - previous.datum).abs() <= DATUM_DRIFT {
            previous.datum
        } else {
            lowest
        }
    });
    let lift = previous.map_or(0.0, |previous| previous.datum - datum);
    let corners: Arc<[f32]> = ground.iter().map(|&height| height - datum).collect();
    // The water of its cells and of those just past its edges: carried
    // over, or read anew from the settled cells around the rectangle
    // holding every cell `previous` does not.
    let side = n + 2;
    let middle = |gi: usize, gj: usize| {
        let at = Vec2::new(
            (first[0] + gi as i64 - 1) as f32,
            (first[1] + gj as i64 - 1) as f32,
        );
        (at + 0.5) * cell
    };
    let mut fresh: Option<([usize; 2], [usize; 2])> = None;
    for gj in 0..side {
        for gi in 0..side {
            if carried(gi, gj, side).is_none() {
                fresh = Some(fresh.map_or(([gi, gj], [gi, gj]), |(low, high)| {
                    (
                        [low[0].min(gi), low[1].min(gj)],
                        [high[0].max(gi), high[1].max(gj)],
                    )
                }));
            }
        }
    }
    let read = fresh.map(|(low, high)| {
        let rectangle = (middle(low[0], low[1]), middle(high[0], high[1]));
        SteadyCells::read(state, rectangle, datum)
    });
    let mut water = Vec::with_capacity(side * side);
    for gj in 0..side {
        for gi in 0..side {
            water.push(match (previous, carried(gi, gj, side), &read) {
                (Some(previous), Some(at), _) => previous.water[at].lifted(lift),
                (_, _, Some(read)) => read.at(middle(gi, gj)),
                (_, _, None) => WaterCell::DRY,
            });
        }
    }
    // The roughness of its cells' ground, from the beds of the finest tiles
    // under their middles, and what those beds hold under the tiles'
    // settled water there.
    let tile = |point: Vec2| {
        let key = pyramid.tile_at(0, point)?;
        Some((state.store.get(&key)?, point - pyramid.tile_origin(key)))
    };
    let mut roughness = Vec::with_capacity(n * n);
    let mut bed_words = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let (cell_roughness, cell_words) = match (previous, carried(i, j, n)) {
                (Some(previous), Some(at)) => (previous.roughness[at], previous.bed_words[at]),
                _ => tile(middle(i + 1, j + 1)).map_or((rough(None), [0; 2]), |(tile, local)| {
                    (rough(tile.bed(local)), tile.bed_words(local))
                }),
            };
            roughness.push(cell_roughness);
            bed_words.push(cell_words);
        }
    }
    WaterWindow {
        origin: origin.to_array(),
        cells,
        cell,
        datum,
        corners,
        water: water.into(),
        roughness: roughness.into(),
        bed_words: bed_words.into(),
        sources: sources(state, first, (cells, cell)).into(),
    }
}

/// The roughness of ground whose river bed is `bed` (none outside rivers'
/// channels), as the window's solver counts it: the bed's D84 in meters,
/// negated where steps or boulders are carved into it, whose friction then
/// counts its grains alone; OPEN_ROUGHNESS outside channels.
fn rough(bed: Option<Bed>) -> f32 {
    match bed {
        Some(bed) if bed.carved => -bed.d84,
        Some(bed) => bed.d84,
        None => OPEN_ROUGHNESS,
    }
}

/// Where the rivers of `state`'s settled water bring water to the window
/// from the terrain's lattice cell `first` of cells `cell` meters across,
/// over `cells` of them along each side, cells in order: what each settled
/// cell brings, shared among the window's cells by the area of it each
/// covers, every cell's summed and the way it comes the rate-weighted mean
/// of what it is brought.
fn sources(
    state: &TerrainState,
    first: [i64; 2],
    (cells, cell): (u32, f32),
) -> Vec<WaterSourceCell> {
    let settled = state.settled.cell();
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
    let mut brought: Vec<(u32, f32, [f32; 2])> = Vec::new();
    for (_, (water, _)) in state.settled.held.iter() {
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
    let mut found: Vec<(WaterSourceCell, [f32; 2])> = Vec::with_capacity(brought.len());
    for (at, rate, way) in brought {
        match found.last_mut() {
            Some((last, weighted)) if last.cell == at => {
                last.rate += rate;
                *weighted = [weighted[0] + way[0], weighted[1] + way[1]];
            }
            _ => found.push((
                WaterSourceCell {
                    cell: at,
                    rate,
                    way: [0.0; 2],
                },
                way,
            )),
        }
    }
    found
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
