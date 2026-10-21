//! Water's simulation windows: the square of ground around the camera over
//! which the renderer runs rivers by the shallow-water equations (its
//! `water_sim`), placed where [`Water::simulated`] asks for one: its
//! ground, the water to start from, where its rivers bring theirs, and the
//! water standing just past its edges. All of it but the ground comes from
//! the terrain's settled water ([`super::settled`]): the window starts with
//! water as it runs at its steady state, flowing and foaming, the cells
//! past its edges give and take water as the steady state does, and its
//! rivers gain what they drain inside it, so water left alone stays as it
//! settled. A window follows the camera, carrying over what it already
//! covers and reading the rest anew, and waits for its ground's finest
//! tiles and the settled water of the tiles it and the cells past its edges
//! cover.

use std::sync::Arc;

use bevy_ecs::prelude::{Res, ResMut};
use glam::Vec2;

use super::residency::TerrainState;
use super::{Terrains, Water};
use crate::Camera;
use crate::math::max;
use crate::terrain_data::{SteadyCell, TileKey};

/// Meters a window's first corner snaps to, and the share of them the
/// camera goes from its middle, along x or z, before it follows.
const SNAP: f32 = 32.0;
const FOLLOW: f32 = 0.75;
/// Meters apart a window's ground is checked for its finest tiles before it
/// is gathered: finer than a tile, so none is missed.
const PROBED: f32 = 32.0;
/// What a cell dry past a window's edge stands at: below any ground.
pub const GHOST_DRY: f32 = -1.0e30;

/// The square over which water is simulated: its first corner (world x and
/// z), its cells along each side and the meters across each; the ground at
/// its cells' corners (world heights, `cells + 1` a row, rows along z); each
/// cell's water to start from as it settles (its surface, or its ground
/// where dry, and its flow along x and z in square meters a second), the
/// level a lake holds it at (NaN where none does) and its foam, row by row;
/// where its rivers bring water; and the cells just past its edges
/// ([`WaterGhost`]): south, then north, of each column, then west, then
/// east, of each row.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterWindow {
    pub origin: [f32; 2],
    pub cells: u32,
    pub cell: f32,
    pub corners: Arc<[f32]>,
    pub seed: Arc<[[f32; 3]]>,
    pub held: Arc<[f32]>,
    pub foam: Arc<[f32]>,
    pub sources: Arc<[WaterSourceCell]>,
    pub ring: Arc<[WaterGhost]>,
}

impl WaterWindow {
    /// Its middle (world x and z).
    pub fn middle(&self) -> Vec2 {
        Vec2::from(self.origin) + Vec2::splat(0.5 * self.cells as f32 * self.cell)
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
/// stands at (world meters; [`GHOST_DRY`] where dry), its flow along x and
/// z (square meters a second), the level a lake holds it at (NaN where none
/// does) and its foam.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterGhost {
    pub surface: f32,
    pub flow: [f32; 2],
    pub held: f32,
    pub foam: f32,
}

/// Places a window around the camera over the terrain holding it where
/// water asks for one, following the camera SNAP meters at a time once it
/// has gone FOLLOW of that from the window's middle along x or z; drops
/// every window where water asks for none.
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
    let cell = water.simulation_cell;
    let cells = max((water.simulated / cell).round(), 8.0) as u32;
    let size = cells as f32 * cell;
    // The window's corner on the terrain's lattice of cells, near a
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
        if state.settled.cell() != cell
            || !finest(state, origin, size)
            || !settled(state, first, cells)
        {
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

/// Whether `state` holds the settled water of every lattice cell of the
/// window from cell `first` over `cells` cells along each side and of the
/// cells just past its edges: of every level-0 tile they lie in that its
/// bake's water reaches.
fn settled(state: &TerrainState, first: [i64; 2], cells: u32) -> bool {
    let pyramid = &state.terrain.pyramid;
    let cell = state.settled.cell();
    let size = pyramid.tile_size(0);
    let low = Vec2::new((first[0] - 1) as f32, (first[1] - 1) as f32) * cell;
    let high = Vec2::new(
        (first[0] + i64::from(cells) + 1) as f32,
        (first[1] + i64::from(cells) + 1) as f32,
    ) * cell;
    let [columns, rows] = pyramid.tiles(0);
    let range = |low: f32, high: f32, count: u32| {
        let first = max((low / size).floor(), 0.0) as u32;
        let last = crate::math::min((high / size).floor(), count as f32 - 1.0) as u32;
        first..=last
    };
    range(low.y, high.y, rows).all(|z| {
        range(low.x, high.x, columns)
            .all(|x| state.settled.has(TileKey::new(0, x as u16, z as u16)))
    })
}

/// The window from `origin` (world x and z), the terrain's lattice cell
/// `first`, over `cells` cells `cell` meters across, as `state`'s resident
/// tiles and settled water stand there: what `previous` covers carried over
/// from it, the rest read anew.
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
    let carried = |i: usize, j: usize, count: usize| {
        let from = glam::IVec2::new(i as i32, j as i32) + shift?;
        let inside = from.cmpge(glam::IVec2::ZERO).all()
            && from.cmplt(glam::IVec2::splat(count as i32)).all();
        inside.then(|| from.y as usize * count + from.x as usize)
    };
    let mut corners = Vec::with_capacity((n + 1) * (n + 1));
    for j in 0..=n {
        for i in 0..=n {
            corners.push(match (previous, carried(i, j, n + 1)) {
                (Some(previous), Some(at)) => previous.corners[at],
                _ => {
                    let point = origin + Vec2::new(i as f32, j as f32) * cell - flat;
                    state
                        .sample(point)
                        .map_or(anchor.y, |(_, sample)| anchor.y + sample.height)
                }
            });
        }
    }
    let bed = |at: usize| {
        let (i, j) = (at % n, at / n);
        0.25 * (corners[j * (n + 1) + i]
            + corners[j * (n + 1) + i + 1]
            + corners[(j + 1) * (n + 1) + i]
            + corners[(j + 1) * (n + 1) + i + 1])
    };
    let lattice = |i: i64, j: i64| state.settled.cell_at(pyramid, i, j);
    let mut seed = Vec::with_capacity(n * n);
    let mut held = Vec::with_capacity(n * n);
    let mut foam = Vec::with_capacity(n * n);
    for at in 0..n * n {
        let (i, j) = (at % n, at / n);
        if let (Some(previous), Some(from)) = (previous, carried(i, j, n)) {
            seed.push(previous.seed[from]);
            held.push(previous.held[from]);
            foam.push(previous.foam[from]);
            continue;
        }
        let ground = bed(at);
        let (water, lake, white) = match lattice(first[0] + i as i64, first[1] + j as i64) {
            Some(SteadyCell::Held { level, foam }) => {
                let level = level + anchor.y;
                ([level, 0.0, 0.0], level, foam)
            }
            Some(SteadyCell::Flowing {
                surface,
                velocity,
                foam,
            }) => {
                let surface = surface + anchor.y;
                let depth = max(surface - ground, 0.0);
                (
                    [surface, depth * velocity[0], depth * velocity[1]],
                    f32::NAN,
                    foam,
                )
            }
            Some(SteadyCell::Dry) | None => ([ground, 0.0, 0.0], f32::NAN, 0.0),
        };
        seed.push(water);
        held.push(lake);
        foam.push(white);
    }
    let sources = sources(state, first, cells);
    let ring = ring(state, first, cells, &corners);
    WaterWindow {
        origin: origin.to_array(),
        cells,
        cell,
        corners: corners.into(),
        seed: seed.into(),
        held: held.into(),
        foam: foam.into(),
        sources: sources.into(),
        ring: ring.into(),
    }
}

/// Where the rivers of `state`'s settled water bring water to the window
/// from lattice cell `first` over `cells` cells along each side, cells in
/// order.
fn sources(state: &TerrainState, first: [i64; 2], cells: u32) -> Vec<WaterSourceCell> {
    let n = i64::from(cells);
    let mut found = Vec::new();
    for (_, (water, _)) in state.settled.held.iter() {
        let columns = i64::from(water.cells[0]);
        for source in &water.sources {
            let (x, z) = (
                water.first[0] + i64::from(source.cell) % columns - first[0],
                water.first[1] + i64::from(source.cell) / columns - first[1],
            );
            if x >= 0 && z >= 0 && x < n && z < n {
                found.push(WaterSourceCell {
                    cell: (z * n + x) as u32,
                    rate: source.rate,
                    way: source.way,
                });
            }
        }
    }
    found.sort_by_key(|source| source.cell);
    found
}

/// The cells just past the edges of the window from lattice cell `first`
/// over `cells` cells along each side, as `state`'s settled water stands
/// them (world heights): south, then north, of each column, then west,
/// then east, of each row. A flowing cell's depth is over the ground at the
/// window's face it borders, its `corners` giving it.
fn ring(state: &TerrainState, first: [i64; 2], cells: u32, corners: &[f32]) -> Vec<WaterGhost> {
    let anchor = state.terrain.origin.y;
    let pyramid = &state.terrain.pyramid;
    let n = cells as usize;
    let corner = |i: usize, j: usize| corners[j * (n + 1) + i];
    let ghost = |i: i64, j: i64, ground: f32| match state.settled.cell_at(pyramid, i, j) {
        Some(SteadyCell::Held { level, foam }) => WaterGhost {
            surface: level + anchor,
            flow: [0.0; 2],
            held: level + anchor,
            foam,
        },
        Some(SteadyCell::Flowing {
            surface,
            velocity,
            foam,
        }) => {
            let surface = surface + anchor;
            let depth = max(surface - ground, 0.0);
            WaterGhost {
                surface,
                flow: [depth * velocity[0], depth * velocity[1]],
                held: f32::NAN,
                foam,
            }
        }
        Some(SteadyCell::Dry) | None => WaterGhost {
            surface: GHOST_DRY,
            flow: [0.0; 2],
            held: f32::NAN,
            foam: 0.0,
        },
    };
    let mut ring = Vec::with_capacity(4 * n);
    for k in 0..n {
        let i = first[0] + k as i64;
        ring.push(ghost(
            i,
            first[1] - 1,
            0.5 * (corner(k, 0) + corner(k + 1, 0)),
        ));
        ring.push(ghost(
            i,
            first[1] + n as i64,
            0.5 * (corner(k, n) + corner(k + 1, n)),
        ));
    }
    for k in 0..n {
        let j = first[1] + k as i64;
        ring.push(ghost(
            first[0] - 1,
            j,
            0.5 * (corner(0, k) + corner(0, k + 1)),
        ));
        ring.push(ghost(
            first[0] + n as i64,
            j,
            0.5 * (corner(n, k) + corner(n, k + 1)),
        ));
    }
    ring
}
