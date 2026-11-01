use super::*;
use crate::codec;
use crate::jobs::{GROUND_GRID, GroundIdentity};
use engine_core::JobRef;
use engine_places::ground::{
    Ground, GroundObstacle, GroundPoint, GroundQuery, GroundRead, GroundSource,
};

struct GroundTerrain {
    entity: Entity,
    origin: Vec3,
    pyramid: Pyramid,
    cutout: Option<(Vec2, Vec2)>,
    around: bool,
    tiles: Vec<(TileKey, Arc<Tile>)>,
    colliders: Vec<Option<PlacedCollider>>,
    walked: Option<Arc<[u8]>>,
}

impl GroundTerrain {
    fn of(state: &TerrainState, stacks: &[(Arc<Layers>, engine_core::RecordKey)]) -> Self {
        Self {
            entity: state.entity,
            origin: state.terrain.origin,
            pyramid: state.terrain.pyramid,
            cutout: state.around.as_ref().map(|around| around.cutout),
            around: state.around.is_some(),
            tiles: state
                .store
                .tiles()
                .map(|tile| (tile.key(), Arc::clone(tile)))
                .collect(),
            colliders: state
                .terrain
                .surface
                .assets
                .iter()
                .map(|asset| asset.collider)
                .collect(),
            walked: walked_identity(state, stacks),
        }
    }

    fn tile(&self, key: TileKey) -> Option<&Arc<Tile>> {
        self.tiles
            .binary_search_by_key(&key, |(held, _)| *held)
            .ok()
            .map(|at| &self.tiles[at].1)
    }

    fn cut_out(&self, local: Vec2) -> bool {
        self.cutout.is_some_and(|(low, high)| {
            local.x > low.x && local.y > low.y && local.x < high.x && local.y < high.y
        })
    }

    fn sample(&self, local: Vec2) -> Option<(u32, TileSample)> {
        if self.cut_out(local) {
            return None;
        }
        let pyramid = &self.pyramid;
        (0..pyramid.levels()).find_map(|level| {
            let key = pyramid.tile_at(level, local)?;
            let tile = self.tile(key)?;
            let sample = tile.sample(local - pyramid.tile_origin(key))?;
            Some((level, sample))
        })
    }

    fn nearest(&self, level: u32, local: Vec2) -> Option<(&Arc<Tile>, [i32; 2])> {
        let key = self.pyramid.tile_at(level, local)?;
        let tile = self.tile(key)?;
        let at = local - self.pyramid.tile_origin(key);
        let spacing = tile.spacing();
        Some((
            tile,
            [
                (at.x / spacing).round() as i32,
                (at.y / spacing).round() as i32,
            ],
        ))
    }

    fn holds(&self, from: Vec2, to: Vec2) -> bool {
        let pyramid = &self.pyramid;
        let [x, z] = pyramid.tiles(0);
        let size = Vec2::new(x as f32, z as f32) * TILE_QUADS as f32 * pyramid.spacing(0);
        let corner = Vec2::new(self.origin.x, self.origin.z);
        [from, to].iter().all(|&point| {
            let local = point - corner;
            local.cmpge(Vec2::ZERO).all() && local.cmple(size).all()
        })
    }
}

fn walked_identity(
    state: &TerrainState,
    stacks: &[(Arc<Layers>, engine_core::RecordKey)],
) -> Option<Arc<[u8]>> {
    let recipe = state.recipe.as_ref()?;
    let stack = stacks
        .iter()
        .find(|(stack, _)| Arc::ptr_eq(stack, &recipe.stack))
        .map(|(_, key)| key.hash())?;
    let edits: Vec<Stamp> = state.terrain.edits.iter().map(|edit| edit.stamp).collect();
    Some(Arc::from(codec::encode(&GroundIdentity {
        stack,
        pyramid: state.terrain.pyramid,
        edits,
    })))
}

pub(crate) struct TerrainGround {
    terrains: Vec<GroundTerrain>,
    revision: u64,
}

impl TerrainGround {
    pub(crate) fn of(terrains: &Terrains) -> Self {
        Self {
            terrains: terrains
                .entries
                .iter()
                .map(|state| GroundTerrain::of(state, &terrains.stacks))
                .collect(),
            revision: terrains.revision(),
        }
    }
}

impl GroundQuery for TerrainGround {
    fn point(&self, at: Vec2) -> Option<GroundPoint> {
        self.terrains.iter().find_map(|terrain| {
            let origin = terrain.origin;
            let local = Vec2::new(at.x - origin.x, at.y - origin.z);
            let (level, sample) = terrain.sample(local)?;
            let material = terrain
                .nearest(level, local)
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

    fn obstacles(&self, ground: Entity) -> Vec<GroundObstacle> {
        let Some(terrain) = self
            .terrains
            .iter()
            .find(|terrain| terrain.entity == ground && !terrain.around)
        else {
            return Vec::new();
        };
        terrain
            .tiles
            .iter()
            .flat_map(|(_, tile)| tile.placed().iter())
            .filter_map(|placed| {
                let collider = terrain
                    .colliders
                    .get(usize::from(placed.asset))
                    .copied()
                    .flatten()?;
                let reach = match collider.shape {
                    PlacedShape::Sphere { radius } => radius,
                    PlacedShape::Cuboid { half_extents } => {
                        engine_math::max(half_extents[0], half_extents[2])
                    }
                };
                Some(GroundObstacle {
                    center: Vec2::new(placed.position[0], placed.position[2]),
                    radius: reach * placed.scale,
                })
            })
            .collect()
    }

    fn source(&self, from: Vec2, to: Vec2) -> GroundRead<GroundSource> {
        let Some(terrain) = self
            .terrains
            .iter()
            .find(|terrain| !terrain.around && terrain.holds(from, to))
        else {
            return GroundRead::Off;
        };
        let Some(identity) = terrain.walked.clone() else {
            return GroundRead::Waiting;
        };
        GroundRead::Ready(GroundSource {
            ground: terrain.entity,
            origin: terrain.origin,
            pyramid: terrain.pyramid,
            identity,
        })
    }

    fn grid(&self) -> Option<JobRef> {
        self.terrains
            .iter()
            .any(|terrain| !terrain.around)
            .then(|| JobRef::from(&GROUND_GRID))
    }

    fn revision(&self) -> u64 {
        self.revision
    }
}

type GroundSeen = (u64, u64, Vec<Option<engine_core::RecordKey>>);

pub(crate) fn publish_ground(
    terrains: Res<Terrains>,
    mut ground: ResMut<Ground>,
    mut published: Local<Option<GroundSeen>>,
) {
    let recipes = terrains
        .entries
        .iter()
        .map(|state| state.recipe.as_ref().map(|recipe| recipe.key))
        .collect();
    let seen = (terrains.epoch, terrains.revision(), recipes);
    if published.as_ref() == Some(&seen) && !ground.is_changed() {
        return;
    }
    *ground = Ground(Arc::new(TerrainGround::of(&terrains)));
    *published = Some(seen);
}
