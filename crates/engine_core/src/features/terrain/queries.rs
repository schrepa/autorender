use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainSample {
    pub terrain: Entity,
    pub height: f32,
    pub normal: Vec3,
    pub hole: bool,
    pub level: u32,
}

impl Terrains {
    pub fn sample(&self, x: f32, z: f32) -> Option<TerrainSample> {
        self.entries.iter().find_map(|state| {
            let origin = state.terrain.origin;
            let (level, sample) = state.sample(Vec2::new(x - origin.x, z - origin.z))?;
            Some(TerrainSample {
                terrain: state.entity,
                height: origin.y + sample.height,
                normal: sample.normal,
                hole: sample.hole,
                level,
            })
        })
    }

    pub fn height_at(&self, x: f32, z: f32) -> Option<f32> {
        self.sample(x, z)
            .filter(|sample| !sample.hole)
            .map(|sample| sample.height)
    }

    pub fn is_hole(&self, x: f32, z: f32) -> Option<bool> {
        self.sample(x, z).map(|sample| sample.hole)
    }

    pub fn region(&self, min: Vec2, max: Vec2, count: [u32; 2], out: &mut Vec<Option<f32>>) {
        out.clear();
        let steps = Vec2::new((count[0].max(2) - 1) as f32, (count[1].max(2) - 1) as f32);
        for row in 0..count[1] {
            for column in 0..count[0] {
                let point = min + (max - min) * Vec2::new(column as f32, row as f32) / steps;
                out.push(self.height_at(point.x, point.y));
            }
        }
    }

    pub fn stats(&self) -> TerrainStats {
        self.stats
    }

    pub fn job_failures(&self) -> (u64, Option<&str>) {
        let failures = self.entries.iter().map(|state| &state.failures);
        let count = failures.clone().map(|failures| failures.count).sum();
        let last = failures
            .filter_map(|failures| failures.last.as_ref())
            .max_by_key(|&&(frame, _)| frame)
            .map(|(_, error)| error.as_str());
        (count, last)
    }

    pub fn terrain(&self, entity: Entity) -> Option<&Terrain> {
        self.entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())
            .map(|state| &state.terrain)
    }

    pub fn bake_record(&self, entity: Entity) -> Option<RecordKey> {
        self.entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())
            .filter(|state| {
                state
                    .baked
                    .as_ref()
                    .is_some_and(|baked| baked.unbaked().is_none())
            })
            .and_then(|state| state.bake.key)
    }

    pub fn layers(&self, entity: Entity) -> Option<Arc<Layers>> {
        let state = self
            .entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())?;
        let base = state
            .baked
            .clone()
            .unwrap_or_else(|| state.terrain.layers.clone());
        if state.terrain.edits.is_empty() {
            return Some(base);
        }
        if let Some((over, edited)) = &state.edited
            && Arc::ptr_eq(over, &base)
        {
            return Some(edited.clone());
        }
        let stamps: Vec<Stamp> = state.terrain.edits.iter().map(|edit| edit.stamp).collect();
        Some(base.with_stamps(&stamps).map_or(base, Arc::new))
    }

    pub fn walked_terrain(&self, from: Vec2, to: Vec2) -> Walked {
        let Some(state) = self.walked(from, to) else {
            return Walked::Off;
        };
        let (Some(recipe), Some(layers)) = (state.recipe.as_ref(), self.layers(state.entity))
        else {
            return Walked::Baking;
        };
        let Some(stack) = self
            .stacks
            .iter()
            .find(|(stack, _)| Arc::ptr_eq(stack, &recipe.stack))
            .map(|(_, key)| key.clone())
        else {
            return Walked::Baking;
        };
        Walked::Ready(WalkedTerrain {
            entity: state.entity,
            origin: state.terrain.origin,
            pyramid: state.terrain.pyramid,
            layers,
            stack,
            recipe: recipe.key.clone(),
            edits: state.terrain.edits.iter().map(|edit| edit.stamp).collect(),
        })
    }

    pub fn walk_obstacles(&self, entity: Entity) -> Vec<Obstacle> {
        self.entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())
            .map_or_else(Vec::new, Self::obstacles)
    }

    fn walked(&self, from: Vec2, to: Vec2) -> Option<&TerrainState> {
        self.entries.iter().find(|state| {
            if state.around.is_some() {
                return false;
            }
            let pyramid = &state.terrain.pyramid;
            let [x, z] = pyramid.tiles(0);
            let size = Vec2::new(x as f32, z as f32)
                * engine_terrain::TILE_QUADS as f32
                * pyramid.spacing(0);
            let corner = Vec2::new(state.terrain.origin.x, state.terrain.origin.z);
            [from, to].iter().all(|&point| {
                let local = point - corner;
                local.cmpge(Vec2::ZERO).all() && local.cmple(size).all()
            })
        })
    }

    fn obstacles(state: &TerrainState) -> Vec<Obstacle> {
        let assets = &state.terrain.surface.assets;
        state
            .store
            .tiles()
            .flat_map(|tile| tile.placed().iter())
            .filter_map(|placed| {
                let collider = assets.get(usize::from(placed.asset))?.collider?;
                let reach = match collider.shape {
                    PlacedShape::Sphere { radius } => radius,
                    PlacedShape::Cuboid { half_extents } => {
                        crate::math::max(half_extents[0], half_extents[2])
                    }
                };
                Some(Obstacle {
                    center: Vec2::new(placed.position[0], placed.position[2]),
                    radius: reach * placed.scale,
                })
            })
            .collect()
    }

    pub fn route(
        &self,
        entity: Entity,
        from: Vec2,
        to: Vec2,
        options: &RouteOptions,
    ) -> Option<Route> {
        let state = self
            .entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())?;
        let layers = state.baked.as_ref().unwrap_or(&state.terrain.layers);
        if layers.unbaked().is_some() {
            return None;
        }
        let origin = Vec2::new(state.terrain.origin.x, state.terrain.origin.z);
        engine_terrain::route(
            layers,
            &state.terrain.pyramid,
            from - origin,
            to - origin,
            options,
        )
    }

    pub fn bakes(&self, entity: Entity) -> TerrainBakes {
        self.entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())
            .map_or_else(TerrainBakes::default, |state| TerrainBakes {
                baking: state.bake.job.is_some(),
                hydrology: state.hydrology.stats,
                erosion: state.erosion,
            })
    }

    pub fn level_detail(&self, entity: Entity) -> Option<(&[f32], &[f32])> {
        self.entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())
            .map(|state| (state.store.errors(), state.ranges.as_slice()))
    }

    pub fn tiles(&self, entity: Entity) -> impl Iterator<Item = &Arc<Tile>> {
        self.entries
            .iter()
            .filter(move |state| state.entity == entity && state.around.is_none())
            .flat_map(|state| state.store.tiles())
    }

    pub fn collision_tiles(&self) -> impl Iterator<Item = (Entity, Vec3, &Arc<Tile>)> {
        self.entries
            .iter()
            .filter(|state| state.terrain.collision)
            .flat_map(|state| {
                state.collision.iter().filter_map(move |key| {
                    let tile = state.store.get(key)?;
                    Some((state.entity, state.terrain.origin, tile))
                })
            })
    }

    pub fn bridges(&self, entity: Entity) -> &[Bridge] {
        self.entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())
            .map_or(&[], |state| &state.bridges)
    }

    pub(crate) fn bridge_sets(&self) -> impl Iterator<Item = bridges::BridgeSet> + '_ {
        self.entries.iter().map(|state| {
            (
                state.entity,
                state.bridges.clone(),
                state.terrain.origin,
                state.terrain.surface.bridge,
            )
        })
    }

    pub fn falls(&self, entity: Entity) -> &[Fall] {
        self.entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())
            .map_or(&[], |state| &state.falls)
    }

    pub(crate) fn fall_sets(&self) -> impl Iterator<Item = (Entity, Arc<[Fall]>, Vec3)> + '_ {
        self.entries
            .iter()
            .map(|state| (state.entity, state.falls.clone(), state.terrain.origin))
    }

    pub fn colliding_bridges(&self) -> impl Iterator<Item = (Entity, Vec3, &[Bridge])> {
        self.entries
            .iter()
            .filter(|state| state.terrain.collision && !state.bridges.is_empty())
            .map(|state| (state.entity, state.terrain.origin, &*state.bridges))
    }

    pub fn revision(&self) -> u64 {
        self.entries
            .iter()
            .map(TerrainState::revision)
            .fold(self.entries.len() as u64, u64::wrapping_add)
    }
}

pub struct WalkedTerrain {
    pub entity: Entity,
    pub origin: Vec3,
    pub pyramid: Pyramid,
    pub layers: Arc<Layers>,
    pub stack: String,
    pub recipe: String,
    pub edits: Vec<Stamp>,
}

pub enum Walked {
    Ready(WalkedTerrain),
    Baking,
    Off,
}

impl WalkedTerrain {
    pub fn reaching(&self, key: TileKey) -> Vec<Stamp> {
        self.edits
            .iter()
            .copied()
            .filter(|stamp| reaches(stamp, &self.pyramid, key))
            .collect()
    }

    pub fn tile_record(&self, key: TileKey) -> (Vec<u8>, RecordKey) {
        let job = TileJob {
            recipe: self.recipe.clone(),
            key,
            stamps: self.reaching(key),
        };
        let input = job.identity();
        let record = RecordKey::new(
            &engine_terrain::sources::TILES,
            TileJob::KIND.name(),
            &input,
        );
        (input, record)
    }
}
