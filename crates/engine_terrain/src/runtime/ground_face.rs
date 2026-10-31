use std::sync::Arc;

use bevy_ecs::prelude::{Entity, Resource, World};
use engine_core::math::max;
use engine_core::{Error, ErrorCode, Expected, Pyramid, RecordKey, TileKey, section_in};
use engine_places::{
    Ground, GroundBridge, GroundFace, GroundGrid, GroundObstacle, GroundPoint, GroundRead,
    GroundRecord, GroundRegions, GroundSource, GroundWorker,
};
use glam::Vec2;

use super::residency::{TerrainState, reaches};
use super::{PlacedShape, Terrains, Water};
use crate::codec::{self, record};
use crate::jobs::{TerrainWorker, TileJob, WalkedTile, stamped, tile_output};
use crate::layer::{Below, Row, maps};
use crate::tile::RiverInputs;
use crate::{
    Generator, HydrologySample, Layers, PathSample, SampleShape, Stamp, TILE_QUADS, TILE_SAMPLES,
    TILE_STRIDE, Tile, TileWater,
};

pub(crate) struct TerrainGround;

impl GroundFace for TerrainGround {
    fn point(&self, world: &World, x: f32, z: f32) -> Option<GroundPoint> {
        world
            .get_resource::<Terrains>()?
            .entries
            .iter()
            .find_map(|state| {
                let origin = state.terrain.origin;
                let local = Vec2::new(x - origin.x, z - origin.z);
                let (level, sample) = state.sample(local)?;
                let material = nearest(state, level, local)
                    .and_then(|(tile, [column, row])| tile.control_at(column, row))
                    .map(|control| control.a);
                Some(GroundPoint {
                    height: origin.y + sample.height,
                    normal: sample.normal,
                    level,
                    hole: sample.hole,
                    material,
                })
            })
    }

    fn map(&self, world: &World, name: &str, x: f32, z: f32) -> Option<f32> {
        world
            .get_resource::<Terrains>()?
            .entries
            .iter()
            .find_map(|state| {
                let origin = state.terrain.origin;
                let local = Vec2::new(x - origin.x, z - origin.z);
                let (level, _) = state.sample(local)?;
                Some(map_at(state, name, level, local))
            })
            .flatten()
    }

    fn revision(&self, world: &World) -> u64 {
        world
            .get_resource::<Terrains>()
            .map_or(0, Terrains::revision)
    }

    fn obstacles(&self, world: &World, ground: Entity) -> Vec<GroundObstacle> {
        world
            .get_resource::<Terrains>()
            .and_then(|terrains| {
                terrains
                    .entries
                    .iter()
                    .find(|state| state.entity == ground && state.around.is_none())
            })
            .map_or_else(Vec::new, obstacles)
    }

    fn source(&self, world: &World, from: Vec2, to: Vec2) -> GroundRead<GroundSource> {
        let Some(terrains) = world.get_resource::<Terrains>() else {
            return GroundRead::Off;
        };
        let Some(state) = walked(terrains, from, to) else {
            return GroundRead::Off;
        };
        let (Some(recipe), Some(layers)) = (state.recipe.as_ref(), terrains.layers(state.entity))
        else {
            return GroundRead::Waiting;
        };
        let Some(stack) = terrains
            .stacks
            .iter()
            .find(|(stack, _)| Arc::ptr_eq(stack, &recipe.stack))
            .map(|(_, key)| key.clone())
        else {
            return GroundRead::Waiting;
        };
        let pyramid = state.terrain.pyramid;
        let edits: Vec<Stamp> = state.terrain.edits.iter().map(|edit| edit.stamp).collect();
        let identity = codec::encode(&GroundIdentity {
            stack,
            pyramid,
            edits: edits.clone(),
        });
        let bridges = ground_bridges(&layers);
        GroundRead::Ready(GroundSource {
            ground: state.entity,
            origin: state.terrain.origin,
            pyramid,
            identity: Arc::from(identity),
            regions: Arc::new(TerrainRegions {
                recipe: recipe.key.clone(),
                pyramid,
                edits,
                layers,
                bridges,
            }),
        })
    }

    fn heightfield(&self) -> bool {
        true
    }
}

#[derive(Resource)]
pub(crate) struct LiveWaterChecked;

pub(crate) fn check_live_water(world: &mut World) {
    if world.contains_resource::<LiveWaterChecked>() {
        return;
    }
    world.insert_resource(LiveWaterChecked);
    let live = section_in::<Water>(world).enabled;
    let heightfield = world
        .get_resource::<Ground>()
        .is_some_and(|ground| ground.face().heightfield());
    if live && !heightfield {
        let error = Error::new(ErrorCode::MissingPlugin, "$.environment.water")
            .expected(Expected::Present)
            .got("heightfield");
        eprintln!(
            "{}",
            serde_json::json!({"event": "live_water_refused", "error": error})
        );
    }
}

struct GroundIdentity {
    stack: String,
    pyramid: Pyramid,
    edits: Vec<Stamp>,
}

record!(GroundIdentity {
    stack,
    pyramid,
    edits,
});

struct TerrainRegions {
    recipe: String,
    pyramid: Pyramid,
    edits: Vec<Stamp>,
    layers: Arc<Layers>,
    bridges: Arc<[GroundBridge]>,
}

impl GroundRegions for TerrainRegions {
    fn record(&self, key: TileKey) -> GroundRecord {
        let job = TileJob {
            recipe: self.recipe.clone(),
            key,
            stamps: reaching(&self.edits, &self.pyramid, key),
        };
        let input = job.identity();
        let record = RecordKey::new(&crate::SOURCE, TileJob::KIND.name(), &input);
        GroundRecord {
            kind: TileJob::KIND.name(),
            input,
            key: record,
        }
    }

    fn grid(&self, key: TileKey, output: &[u8]) -> Result<GroundGrid, Error> {
        let tile = keyed_tile(key, output)?;
        Ok(ground_grid(
            &tile,
            &self.layers,
            &self.pyramid,
            self.bridges.clone(),
        ))
    }
}

impl GroundWorker for TerrainWorker {
    fn grid(&self, identity: &[u8], key: TileKey) -> Result<GroundGrid, Error> {
        let (tile, layers, pyramid) = walked_tile(self, identity, key)?;
        let grid = ground_grid(&tile, &layers, &pyramid, ground_bridges(&layers));
        *self.last_walk() = Some(WalkedTile {
            identity: identity.into(),
            key,
            tile,
        });
        Ok(grid)
    }
}

pub(super) fn keyed_tile(key: TileKey, output: &[u8]) -> Result<Tile, Error> {
    let tile = tile_output(output)?;
    if tile.key() != key {
        return Err(codec::invalid("$.key").got([
            u16::from(tile.key().level),
            tile.key().x,
            tile.key().z,
        ]));
    }
    Ok(tile)
}

pub(super) fn walked_tile(
    worker: &TerrainWorker,
    identity: &[u8],
    key: TileKey,
) -> Result<(Tile, Arc<Layers>, Pyramid), Error> {
    let GroundIdentity {
        stack,
        pyramid,
        edits,
    } = codec::decode(identity)?;
    if !pyramid.contains(key) {
        return Err(Error::new(ErrorCode::OutOfRange, "$.key").got([
            u16::from(key.level),
            key.x,
            key.z,
        ]));
    }
    let layers = worker
        .stack(&stack)
        .ok_or_else(|| Error::new(ErrorCode::UnknownReference, "$.stack").got(&stack))?;
    let layers = stamped(layers, &reaching(&edits, &pyramid, key))?;
    let mut whole = u64::MAX;
    let tile = Generator::new(layers.clone(), &pyramid, key)
        .advance(&mut whole)
        .ok_or_else(|| {
            Error::new(ErrorCode::InvalidTerrain, "$.heights").expected(Expected::Finite)
        })?;
    let tile: Tile = codec::decode(&codec::encode(&tile))?;
    Ok((tile, layers, pyramid))
}

pub(super) fn tile_surfaces(tile: &Tile) -> Option<Box<[f32]>> {
    let side = TILE_SAMPLES as usize;
    let stride = TILE_STRIDE as usize;
    let stored = |x: usize, z: usize| (z + 1) * stride + x + 1;
    let surfaces = tile.water().map(TileWater::surfaces)?;
    let mut water = vec![f32::NAN; side * side].into_boxed_slice();
    for z in 0..side {
        for x in 0..side {
            water[z * side + x] = surfaces[stored(x, z)];
        }
    }
    Some(water)
}

fn reaching(edits: &[Stamp], pyramid: &Pyramid, key: TileKey) -> Vec<Stamp> {
    edits
        .iter()
        .copied()
        .filter(|stamp| reaches(stamp, pyramid, key))
        .collect()
}

fn ground_bridges(layers: &Layers) -> Arc<[GroundBridge]> {
    layers
        .bridges()
        .into_iter()
        .map(|bridge| GroundBridge {
            deck: bridge.deck,
            width: bridge.width,
            tunnel: bridge.tunnel,
        })
        .collect()
}

fn ground_grid(
    tile: &Tile,
    layers: &Layers,
    pyramid: &Pyramid,
    bridges: Arc<[GroundBridge]>,
) -> GroundGrid {
    let side = TILE_SAMPLES as usize;
    let key = tile.key();
    let spacing = pyramid.spacing(u32::from(key.level));
    let stride = TILE_STRIDE as usize;
    let stored = |x: usize, z: usize| (z + 1) * stride + x + 1;
    let mut heights = vec![0.0; side * side].into_boxed_slice();
    for z in 0..side {
        for x in 0..side {
            heights[z * side + x] = tile.heights()[stored(x, z)];
        }
    }
    let first = pyramid.first_sample(key);
    GroundGrid {
        key,
        spacing,
        heights,
        holes: holes(layers, first, spacing),
        bridges,
    }
}

fn position(sample: [i64; 2], spacing: f32) -> Vec2 {
    Vec2::new(sample[0] as f32, sample[1] as f32) * spacing
}

fn holes(layers: &Layers, first: [i64; 2], spacing: f32) -> Option<Box<[u64]>> {
    let side = TILE_SAMPLES as usize;
    let quads = TILE_QUADS as i64;
    let last = [first[0] + quads, first[1] + quads];
    if !layers.holes_reach(position(first, spacing), position(last, spacing)) {
        return None;
    }
    let mut bits = vec![0_u64; (side * side).div_ceil(64)];
    let mut any = false;
    for z in 0..side {
        for x in 0..side {
            let point = position([first[0] + x as i64, first[1] + z as i64], spacing);
            if layers.hole_at(point.x, point.y) {
                let index = z * side + x;
                bits[index / 64] |= 1 << (index % 64);
                any = true;
            }
        }
    }
    any.then(|| bits.into_boxed_slice())
}

fn nearest(state: &TerrainState, level: u32, local: Vec2) -> Option<(&Arc<Tile>, [i32; 2])> {
    let pyramid = &state.terrain.pyramid;
    let key = pyramid.tile_at(level, local)?;
    let tile = state.store.get(&key)?;
    let at = local - pyramid.tile_origin(key);
    let spacing = tile.spacing();
    Some((
        tile,
        [
            (at.x / spacing).round() as i32,
            (at.y / spacing).round() as i32,
        ],
    ))
}

fn map_at(state: &TerrainState, name: &str, level: u32, local: Vec2) -> Option<f32> {
    let (tile, [column, row]) = nearest(state, level, local)?;
    let layers: &Layers = state.baked.as_ref().unwrap_or(&state.terrain.layers);
    let pyramid = &state.terrain.pyramid;
    let stride = TILE_STRIDE as usize;
    let stored = |at: i32| (at.clamp(-1, stride as i32 - 2) + 1) as usize;
    let (column, row) = (stored(column), stored(row));
    let (up, down) = (row.saturating_sub(1), (row + 1).min(stride - 1));
    let (left, right) = (column.saturating_sub(1), (column + 1).min(stride - 1));
    let heights = tile.heights();
    let at = |r: usize, c: usize| heights[r * stride + c];
    let height = at(row, column);
    let spacing = tile.spacing();
    let key = tile.key();
    let first = pyramid.first_sample(key);
    let sample = [first[0] + column as i64 - 1, first[1] + row as i64 - 1];
    let offset = layers.offset_samples(spacing);
    let x = (sample[0] + offset[0]) as f32 * spacing;
    let z = (sample[1] + offset[1]) as f32 * spacing;
    let shape = || {
        SampleShape::of(
            [up, row, down].map(|r| [left, column, right].map(|c| at(r, c))),
            spacing,
        )
    };
    let shown = layers.shown(x, z);
    let water = || {
        layers
            .hydrology()
            .filter(|_| shown > 0.0)
            .map_or_else(HydrologySample::default, |water| {
                RiverInputs::at(&water.rivers, [x, z], (height, spacing)).with_maps(
                    water,
                    [x, z],
                    spacing,
                )
            })
    };
    let change = || {
        let mut sum = [0.0_f32];
        for map in layers.changes() {
            map.change_row([x], z, spacing, |index, value| {
                sum[index] += value;
            });
        }
        sum[0] * shown
    };
    match name {
        maps::HEIGHT => Some(height),
        maps::SLOPE => {
            let [gx, gz] = shape().gradient;
            Some((gx * gx + gz * gz).sqrt())
        }
        maps::CURVATURE => Some(shape().curvature),
        maps::FLOW => Some(water().flow * shown),
        maps::WETNESS => Some(water().wetness * shown),
        maps::LAKE => Some(water().lake * shown),
        maps::RIVER => Some(water().river),
        maps::GRAIN => Some(water().grain),
        maps::POWER => Some(water().power),
        maps::FLOOD => Some(water().flood),
        maps::WORN => Some(max(-change(), 0.0)),
        maps::DEPOSIT => Some(max(change(), 0.0)),
        maps::PATH => {
            let path = if shown > 0.0 {
                layers.path_at(x, z)
            } else {
                PathSample::FAR
            };
            Some(path.distance.abs() - path.half)
        }
        _ => user_map(
            (layers, pyramid),
            name,
            (u32::from(key.level), sample),
            (x, z, spacing),
        ),
    }
}

fn user_map(
    (layers, pyramid): (&Layers, &Pyramid),
    name: &str,
    (level, sample): (u32, [i64; 2]),
    (x, z, spacing): (f32, f32, f32),
) -> Option<f32> {
    let (index, layer) = layers
        .layers()
        .iter()
        .enumerate()
        .rev()
        .find(|(_, layer)| layer.outputs().iter().any(|output| output == name))?;
    let height = layers.below(index).sample_height(pyramid, level, sample);
    let point = Vec2::new(x, z);
    let row = Row {
        xs: std::slice::from_ref(&x),
        z,
        spacing,
        low: point,
        high: point,
    };
    let mut out = [0.0_f32];
    layer
        .map_row(name, Below::new(layers, index), &row, &[height], &mut out)
        .then_some(out[0])
}

fn walked(terrains: &Terrains, from: Vec2, to: Vec2) -> Option<&TerrainState> {
    terrains.entries.iter().find(|state| {
        if state.around.is_some() {
            return false;
        }
        let pyramid = &state.terrain.pyramid;
        let [x, z] = pyramid.tiles(0);
        let size = Vec2::new(x as f32, z as f32) * TILE_QUADS as f32 * pyramid.spacing(0);
        let corner = Vec2::new(state.terrain.origin.x, state.terrain.origin.z);
        [from, to].iter().all(|&point| {
            let local = point - corner;
            local.cmpge(Vec2::ZERO).all() && local.cmple(size).all()
        })
    })
}

fn obstacles(state: &TerrainState) -> Vec<GroundObstacle> {
    let assets = &state.terrain.surface.assets;
    state
        .store
        .tiles()
        .flat_map(|tile| tile.placed().iter())
        .filter_map(|placed| {
            let collider = assets.get(usize::from(placed.asset))?.collider?;
            let reach = match collider.shape {
                PlacedShape::Sphere { radius } => radius,
                PlacedShape::Cuboid { half_extents } => max(half_extents[0], half_extents[2]),
            };
            Some(GroundObstacle {
                center: Vec2::new(placed.position[0], placed.position[2]),
                radius: reach * placed.scale,
            })
        })
        .collect()
}
