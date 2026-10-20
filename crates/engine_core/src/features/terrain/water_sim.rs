//! Water's simulation windows: the square of ground around the camera over
//! which the renderer runs rivers by the shallow-water equations (its
//! `water_sim`), placed where [`Water::simulated`] asks for one: its
//! ground, the water the hydrology bake stands there to start from, the
//! lakes that hold their level, and the rivers flowing in. A window
//! follows the camera, carrying over what it already covers and reading
//! only the ground and water it newly covers from the tiles, and waits for
//! its ground's finest tiles: coarser ones lack the channels the rivers run
//! in.

use std::sync::Arc;

use bevy_ecs::prelude::{Res, ResMut};
use glam::Vec2;

use super::residency::TerrainState;
use super::{Terrains, Water};
use crate::Camera;
use crate::math::max;

/// Meters a window's middle snaps to, and the share of them the camera goes
/// from its middle, along x or z, before it follows.
const SNAP: f32 = 32.0;
const FOLLOW: f32 = 0.75;
/// The least a river flowing in brings (cubic meters a second).
const TRICKLE: f32 = 0.05;
/// Cells inside a window's edge within which rivers count as flowing in
/// (closer, they leave again at once).
const MARGIN: f32 = 2.0;
/// Meters apart a window's ground is checked for its finest tiles before it
/// is gathered: finer than a tile, so none is missed.
const PROBED: f32 = 32.0;

/// The square over which water is simulated: its first corner (world x and
/// z), its cells along each side and the meters across each; the ground at
/// its cells' corners (world heights, `cells + 1` a row, rows along z); each
/// cell's water to start from, as the bake stands and runs it (its surface,
/// or its ground where dry, and its flow along x and z in square meters a
/// second) and the level a lake holds it at (NaN where none does), row by
/// row; and the rivers flowing in.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterWindow {
    pub origin: [f32; 2],
    pub cells: u32,
    pub cell: f32,
    pub corners: Arc<[f32]>,
    pub seed: Arc<[[f32; 3]]>,
    pub held: Arc<[f32]>,
    pub inflows: Arc<[WaterInflow]>,
}

impl WaterWindow {
    /// Its middle (world x and z).
    pub fn middle(&self) -> Vec2 {
        Vec2::from(self.origin) + Vec2::splat(0.5 * self.cells as f32 * self.cell)
    }
}

/// A river flowing into a [`WaterWindow`]: the cells it flows in at (their
/// indices, row by row), the cubic meters a second it brings, and the way
/// it runs (a unit vector along x and z).
#[derive(Clone, Debug, PartialEq)]
pub struct WaterInflow {
    pub cells: Vec<u32>,
    pub rate: f32,
    pub way: [f32; 2],
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
    let cells = max((water.simulated / water.simulation_cell).round(), 8.0) as u32;
    let cell = water.simulation_cell;
    let size = cells as f32 * cell;
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
        if off.is_some_and(|off| off.max_element() <= FOLLOW * SNAP) {
            continue;
        }
        let origin = (eye / SNAP).round() * SNAP - Vec2::splat(0.5 * size);
        if !finest(state, origin, size) {
            continue;
        }
        let window = gather(&terrains, index, origin, (cells, cell), current.as_deref());
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

/// The window from `origin` (world x and z) over `cells` cells `cell`
/// meters across over terrain `index`, as its resident tiles and baked
/// rivers stand there: what `previous` covers carried over from it, the
/// rest read from the tiles.
fn gather(
    terrains: &Terrains,
    index: usize,
    origin: Vec2,
    (cells, cell): (u32, f32),
    previous: Option<&WaterWindow>,
) -> WaterWindow {
    let state = &terrains.entries[index];
    let anchor = state.terrain.origin;
    let flat = Vec2::new(anchor.x, anchor.z);
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
    let mut seed = Vec::with_capacity(n * n);
    let mut held = Vec::with_capacity(n * n);
    for at in 0..n * n {
        let (i, j) = (at % n, at / n);
        if let (Some(previous), Some(from)) = (previous, carried(i, j, n)) {
            seed.push(previous.seed[from]);
            held.push(previous.held[from]);
            continue;
        }
        let ground = bed(at);
        let middle = origin + (Vec2::new(i as f32, j as f32) + 0.5) * cell - flat;
        let water = state
            .water(middle)
            .filter(|water| water.surface + anchor.y > ground);
        let (start, lake) = match water {
            Some(water) if water.lake => {
                let surface = water.surface + anchor.y;
                ([surface, 0.0, 0.0], surface)
            }
            Some(water) => {
                let surface = water.surface + anchor.y;
                let depth = surface - ground;
                (
                    [surface, depth * water.flow.x, depth * water.flow.y],
                    f32::NAN,
                )
            }
            None => ([ground, 0.0, 0.0], f32::NAN),
        };
        seed.push(start);
        held.push(lake);
    }
    let wet: Vec<bool> = (0..n * n)
        .map(|at| held[at].is_nan() && seed[at][0] > bed(at))
        .collect();
    let inflows = inflows(terrains, origin, (cells, cell), (&held, &wet));
    WaterWindow {
        origin: origin.to_array(),
        cells,
        cell,
        corners: corners.into(),
        seed: seed.into(),
        held: held.into(),
        inflows: inflows.into(),
    }
}

/// The baked rivers flowing into the window from `origin` over `cells`
/// cells `cell` meters across: each where it crosses into the window, and
/// each starting within it but at a confluence (where the rivers it starts
/// from bring its water), lakes' outlets among them (lakes give none), over
/// the cells within half its width no lake holds (`held`) that its baked
/// water covers (`wet`), or all of those where it covers none.
fn inflows(
    terrains: &Terrains,
    origin: Vec2,
    (cells, cell): (u32, f32),
    (held, wet): (&[f32], &[bool]),
) -> Vec<WaterInflow> {
    let n = cells as i32;
    let size = cells as f32 * cell;
    let inside = |point: Vec2| {
        let margin = MARGIN * cell;
        point.x > origin.x + margin
            && point.y > origin.y + margin
            && point.x < origin.x + size - margin
            && point.y < origin.y + size - margin
    };
    let mut found = Vec::new();
    for (_, terrain, layers) in terrains.waters() {
        let Some(hydrology) = layers.hydrology() else {
            continue;
        };
        let courses: Vec<_> = hydrology.rivers().courses().collect();
        let key = |point: [f32; 2]| [point[0].to_bits(), point[1].to_bits()];
        let ends: crate::collections::HashSet<[u32; 2]> = courses
            .iter()
            .filter_map(|course| course.points().last().copied())
            .map(key)
            .collect();
        let world = |point: [f32; 2]| Vec2::new(point[0] + terrain.x, point[1] + terrain.z);
        for course in &courses {
            let points = course.points();
            for k in 0..points.len().saturating_sub(1) {
                let at = world(points[k]);
                let entering = inside(at)
                    && if k == 0 {
                        !ends.contains(&key(points[0]))
                    } else {
                        !inside(world(points[k - 1]))
                    };
                let rate = course.discharge(k);
                let flowing = rate >= TRICKLE;
                if !entering || !flowing {
                    continue;
                }
                let reach = max(course.half(k), cell);
                let low = ((at - origin) / cell - reach / cell)
                    .floor()
                    .as_ivec2()
                    .max(glam::IVec2::ZERO);
                let high = ((at - origin) / cell + reach / cell)
                    .ceil()
                    .as_ivec2()
                    .min(glam::IVec2::splat(n - 1));
                let mut covered = Vec::new();
                for j in low.y..=high.y {
                    for i in low.x..=high.x {
                        let middle = origin + (Vec2::new(i as f32, j as f32) + 0.5) * cell;
                        let index = (j * n + i) as usize;
                        if middle.distance(at) <= reach && held[index].is_nan() {
                            covered.push(index as u32);
                        }
                    }
                }
                if covered.iter().any(|&index| wet[index as usize]) {
                    covered.retain(|&index| wet[index as usize]);
                }
                if !covered.is_empty() {
                    let way = (world(points[k + 1]) - at).normalize_or(Vec2::X);
                    found.push(WaterInflow {
                        cells: covered,
                        rate,
                        way: way.to_array(),
                    });
                }
            }
        }
    }
    found
}
