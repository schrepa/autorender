use super::*;
use crate::jobs::GROUND_GRID;
use engine_core::JobRef;
use engine_places::ground::{Ground, GroundObstacle, GroundPoint, GroundQuery};

struct GroundTerrain {
    origin: Vec3,
    pyramid: Pyramid,
    cutout: Option<(Vec2, Vec2)>,
    around: bool,
    tiles: HashMap<TileKey, Arc<Tile>>,
    colliders: Vec<Option<PlacedCollider>>,
}

impl GroundTerrain {
    fn of(state: &TerrainState) -> Self {
        Self {
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
        }
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
            let tile = self.tiles.get(&key)?;
            let sample = tile.sample(local - pyramid.tile_origin(key))?;
            Some((level, sample))
        })
    }

    fn nearest(&self, level: u32, local: Vec2) -> Option<(&Arc<Tile>, [i32; 2])> {
        let key = self.pyramid.tile_at(level, local)?;
        let tile = self.tiles.get(&key)?;
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
}

pub(crate) struct TerrainGround {
    terrains: Vec<GroundTerrain>,
}

impl TerrainGround {
    pub(crate) fn of(terrains: &Terrains) -> Self {
        Self {
            terrains: terrains.entries.iter().map(GroundTerrain::of).collect(),
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

    fn obstacles(&self, low: Vec2, high: Vec2, found: &mut Vec<GroundObstacle>) {
        for terrain in self.terrains.iter().filter(|terrain| !terrain.around) {
            let origin = Vec2::new(terrain.origin.x, terrain.origin.z);
            let mut keys: Vec<&TileKey> = terrain.tiles.keys().collect();
            keys.sort_unstable();
            for key in keys {
                for placed in terrain.tiles[key].placed() {
                    let Some(collider) = terrain
                        .colliders
                        .get(usize::from(placed.asset))
                        .copied()
                        .flatten()
                    else {
                        continue;
                    };
                    let reach = match collider.shape {
                        PlacedShape::Sphere { radius } => radius,
                        PlacedShape::Cuboid { half_extents } => {
                            engine_math::max(half_extents[0], half_extents[2])
                        }
                    };
                    let center = origin + Vec2::new(placed.position[0], placed.position[2]);
                    let radius = reach * placed.scale;
                    if center.x + radius < low.x
                        || center.y + radius < low.y
                        || center.x - radius > high.x
                        || center.y - radius > high.y
                    {
                        continue;
                    }
                    found.push(GroundObstacle { center, radius });
                }
            }
        }
    }

    fn grid(&self) -> Option<JobRef> {
        self.terrains
            .iter()
            .any(|terrain| !terrain.around)
            .then(|| JobRef::from(&GROUND_GRID))
    }
}

pub(crate) fn publish_ground(
    terrains: Res<Terrains>,
    mut ground: ResMut<Ground>,
    mut published: Local<Option<(u64, u64)>>,
) {
    let seen = (terrains.epoch, terrains.revision());
    if *published == Some(seen) && !ground.is_changed() {
        return;
    }
    *ground = Ground(Arc::new(TerrainGround::of(&terrains)));
    *published = Some(seen);
}
