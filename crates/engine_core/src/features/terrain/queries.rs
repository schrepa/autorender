//! Queries of the resident tiles: heights, samples, layers, what walks read
//! and routes, bridges and falls.

use super::*;

/// The surface of a terrain at a point: its terrain, world height, the
/// normal of its smooth surface there, whether its cell is a hole, and the
/// level of the tile it came from (0 is the finest).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainSample {
    pub terrain: Entity,
    pub height: f32,
    pub normal: Vec3,
    pub hole: bool,
    pub level: u32,
}

impl Terrains {
    /// The surface under world point `(x, z)` of the first terrain with a
    /// resident tile there, from the finest such tile.
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

    /// The world height of the terrain under `(x, z)`; `None` where no tile
    /// is resident or the cell is a hole.
    pub fn height_at(&self, x: f32, z: f32) -> Option<f32> {
        self.sample(x, z)
            .filter(|sample| !sample.hole)
            .map(|sample| sample.height)
    }

    /// The meters tracks sink the ground under `(x, z)` at
    /// most, from its surface materials' [`SurfaceMaterial::deform`]; `None`
    /// where no tile with a surface is resident.
    pub fn deform_at(&self, x: f32, z: f32) -> Option<f32> {
        self.entries.iter().find_map(|state| {
            let origin = state.terrain.origin;
            state.deform(Vec2::new(x - origin.x, z - origin.z))
        })
    }

    /// Whether the cell under `(x, z)` is a hole; `None` where no tile is
    /// resident.
    pub fn is_hole(&self, x: f32, z: f32) -> Option<bool> {
        self.sample(x, z).map(|sample| sample.hole)
    }

    /// Heights of a grid of `count` points from `min` to `max` (world x and
    /// z), row by row along z, into `out`: `None` where no tile is resident
    /// or the cell is a hole.
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

    /// Counters of the last streaming update.
    pub fn stats(&self) -> TerrainStats {
        self.stats
    }

    /// The jobs of every terrain that failed on workers since it arrived,
    /// and the latest one's error.
    pub fn job_failures(&self) -> (u64, Option<&str>) {
        let failures = self.entries.iter().map(|state| &state.failures);
        let count = failures.clone().map(|failures| failures.count).sum();
        let last = failures
            .filter_map(|failures| failures.last.as_ref())
            .max_by_key(|&&(frame, _)| frame)
            .map(|(_, error)| error.as_str());
        (count, last)
    }

    /// The terrain of `entity`, while it streams.
    pub fn terrain(&self, entity: Entity) -> Option<&Terrain> {
        self.entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())
            .map(|state| &state.terrain)
    }

    /// The record `entity`'s bakes came from, once they are done: what the
    /// layers [`Self::layers`] gives depend on beyond those authored, which
    /// records made from them name in their identities.
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

    /// The layers `entity`'s tiles generate from: its own, with their
    /// hydrology baked once it is and its edits over them.
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

    /// What walks between world `from` and `to` (x and z) read of the
    /// terrain both lie on ([`Walked`]): never surroundings, which no one
    /// walks.
    pub(crate) fn walked_terrain(&self, from: Vec2, to: Vec2) -> Walked {
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

    /// The footprints of the placed objects that collide on `entity`'s
    /// resident tiles, terrain-local: what walks over it keep clear of.
    pub(crate) fn walk_obstacles(&self, entity: Entity) -> Vec<Obstacle> {
        self.entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())
            .map_or_else(Vec::new, Self::obstacles)
    }

    /// The terrain both world `from` and `to` (x and z) lie on, which walks
    /// between them cross: never surroundings, which no one
    /// walks.
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

    /// The footprints of the placed objects that collide on `state`'s
    /// resident tiles.
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

    /// The cheapest route over `entity`'s generated terrain from world
    /// `from` to `to` (x and z), as [`engine_terrain::route`] finds it over
    /// its layers with their bakes: its points are
    /// terrain-local, as path layers name them. `None` for terrains read
    /// from files, before their bakes are done, and where no route keeps
    /// within the steepest grade. Its work grows with the square of the
    /// ends' distance: for authoring and loading, not every frame.
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

    /// `entity`'s bakes: whether one is in
    /// progress, and what its hydrology and erosion bakes found.
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

    /// Each level's largest measured height difference from the next
    /// coarser one (meters; zero until measured) and the range its tiles
    /// streamed by at the last update (meters from the eye), finest first,
    /// of `entity`'s terrain.
    pub fn level_detail(&self, entity: Entity) -> Option<(&[f32], &[f32])> {
        self.entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())
            .map(|state| (state.store.errors(), state.ranges.as_slice()))
    }

    /// Resident tiles of `entity`'s terrain.
    pub fn tiles(&self, entity: Entity) -> impl Iterator<Item = &Arc<Tile>> {
        self.entries
            .iter()
            .filter(move |state| state.entity == entity && state.around.is_none())
            .flat_map(|state| state.store.tiles())
    }

    /// Level-0 tiles resident around the camera and focus entities, of
    /// terrains that collide: the terrain, its origin and the tile.
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

    /// The bridges of `entity`'s roads, terrain-local, once
    /// its bakes are done.
    pub fn bridges(&self, entity: Entity) -> &[Bridge] {
        self.entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())
            .map_or(&[], |state| &state.bridges)
    }

    /// Every terrain's bridges, its origin and its bridge material.
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

    /// The waterfalls of `entity`'s rivers, terrain-local,
    /// once its bakes are done.
    pub fn falls(&self, entity: Entity) -> &[Fall] {
        self.entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())
            .map_or(&[], |state| &state.falls)
    }

    /// Every terrain's waterfalls and its origin.
    pub(crate) fn fall_sets(&self) -> impl Iterator<Item = (Entity, Arc<[Fall]>, Vec3)> + '_ {
        self.entries
            .iter()
            .map(|state| (state.entity, state.falls.clone(), state.terrain.origin))
    }

    /// Colliding terrains' bridges, with their origins.
    pub fn colliding_bridges(&self) -> impl Iterator<Item = (Entity, Vec3, &[Bridge])> {
        self.entries
            .iter()
            .filter(|state| state.terrain.collision && !state.bridges.is_empty())
            .map(|state| (state.entity, state.terrain.origin, &*state.bridges))
    }

    /// Advances whenever any terrain's resident tiles, or what stands on
    /// them, change.
    pub fn revision(&self) -> u64 {
        self.entries
            .iter()
            .map(TerrainState::revision)
            .fold(self.entries.len() as u64, u64::wrapping_add)
    }
}

/// What walks read of a terrain: the terrain, its origin, its
/// pyramid, its layers with their bakes and edits (which tiles' grounds
/// read), the keys its stack and recipe are shared with workers by, and its
/// edits, as they stood when a walk set out.
pub(crate) struct WalkedTerrain {
    pub(crate) entity: Entity,
    pub(crate) origin: Vec3,
    pub(crate) pyramid: Pyramid,
    pub(crate) layers: Arc<Layers>,
    pub(crate) stack: String,
    pub(crate) recipe: String,
    pub(crate) edits: Vec<Stamp>,
}

/// What walks between two points find of the terrain under them: one ready
/// to walk, one whose bakes or recipe are not done yet, or none under both.
pub(crate) enum Walked {
    Ready(WalkedTerrain),
    Baking,
    Off,
}

impl WalkedTerrain {
    /// The edits reaching tile `key`'s ground, in order.
    fn reaching(&self, key: TileKey) -> Vec<Stamp> {
        self.edits
            .iter()
            .copied()
            .filter(|stamp| reaches(stamp, &self.pyramid, key))
            .collect()
    }

    /// Tile `key`'s portals as `gait` walks it: its job's input, and its
    /// record's key.
    pub(crate) fn portals_record(&self, key: TileKey, gait: &Gait) -> (Vec<u8>, RecordKey) {
        let job = PortalsJob {
            stack: self.stack.clone(),
            pyramid: self.pyramid,
            key,
            stamps: self.reaching(key),
            gait: *gait,
        };
        let input = job.identity();
        let record = RecordKey::new(
            &engine_terrain::sources::PORTALS,
            PortalsJob::KIND.name(),
            &input,
        );
        (input, record)
    }

    /// Tile `key`'s record, which its ground for walks reads, as the terrain
    /// loads it: its job's input, and its record's key.
    pub(crate) fn tile_record(&self, key: TileKey) -> (Vec<u8>, RecordKey) {
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
