//! Queries of the resident tiles: heights, samples, layers, walks and
//! routes, bridges and falls.

use super::*;

/// The surface of a terrain at a point: its terrain, world height, the
/// normal of its triangle, whether its cell is a hole, and the level of the
/// tile it came from (0 is the finest).
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

    /// The layers `entity`'s tiles generate from: its own, with their
    /// hydrology baked once it is and its edits over them;
    /// `None` for terrains read from files.
    /// Stable completed generation layers for optional whole-terrain jobs,
    /// even when no camera requests a tile of this terrain this frame.
    pub(crate) fn completed_layers(&mut self, entity: Entity) -> Option<Arc<Layers>> {
        let state = self
            .entries
            .iter_mut()
            .find(|s| s.entity == entity && s.around.is_none())?;
        let base = match &state.terrain.source {
            TerrainSource::Generated(authored) => {
                state.baked.clone().unwrap_or_else(|| authored.clone())
            }
            TerrainSource::File { .. } => return None,
        };
        if base.unbaked().is_some() {
            return None;
        }
        Some(state.generation(base))
    }

    pub fn layers(&self, entity: Entity) -> Option<Arc<Layers>> {
        let state = self
            .entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())?;
        let base = match &state.terrain.source {
            TerrainSource::Generated(layers) => {
                state.baked.clone().unwrap_or_else(|| layers.clone())
            }
            TerrainSource::File { .. } => return None,
        };
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

    /// The shortest walk over `entity`'s generated terrain from world
    /// `from` to `to` (x and z) as `options` walk: around the
    /// placed objects that collide on its resident tiles, their footprints
    /// widened by the agent's radius, its points world and following the
    /// ground. `None` for terrains read from files and where no walk
    /// reaches. Its work grows with the area around the ends: for planning,
    /// not every frame.
    pub fn walk(
        &self,
        entity: Entity,
        from: Vec2,
        to: Vec2,
        options: &NavOptions,
    ) -> Option<NavPath> {
        let state = self
            .entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())?;
        let layers = self.layers(entity)?;
        let origin = state.terrain.origin;
        let corner = Vec2::new(origin.x, origin.z);
        let mut path = engine_terrain::walk(
            &layers,
            &state.terrain.pyramid,
            from - corner,
            to - corner,
            &Self::obstacles(state),
            options,
        )?;
        for point in &mut path.points {
            *point = [
                point[0] + origin.x,
                point[1] + origin.y,
                point[2] + origin.z,
            ];
        }
        Some(path)
    }

    /// A walk over the generated terrain under world `from` and `to` (x
    /// and z) as `options` walk, to search in budgeted steps:
    /// the terrain, its origin, which the search's terrain-local
    /// points are from, and the search. `None` unless both ends lie on one
    /// generated terrain.
    pub fn walk_search(
        &self,
        from: Vec2,
        to: Vec2,
        options: &NavOptions,
    ) -> Option<(Entity, Vec3, engine_terrain::WalkSearch)> {
        let state = self.walked(from, to)?;
        let layers = self.layers(state.entity)?;
        let origin = state.terrain.origin;
        let corner = Vec2::new(origin.x, origin.z);
        let search = engine_terrain::WalkSearch::new(
            layers,
            &state.terrain.pyramid,
            (from - corner, to - corner),
            Self::obstacles(state),
            options,
        );
        Some((state.entity, origin, search))
    }

    /// The input of a walk job over the generated terrain
    /// under world `from` and `to` (x and z) as `options` walk, once workers
    /// hold its recipe for the generation its tiles are made from now: the
    /// terrain, its origin, which the walk's terrain-local points are from,
    /// and the input. `None` otherwise, as for [`Self::walk_search`].
    pub fn walk_job(
        &self,
        from: Vec2,
        to: Vec2,
        options: &NavOptions,
    ) -> Option<(Entity, Vec3, Vec<u8>)> {
        let state = self.walked(from, to)?;
        let (entity, origin) = (state.entity, state.terrain.origin);
        let recipe = state.recipe.as_ref()?;
        if recipe.generation != state.store.generation() {
            return None;
        }
        let corner = Vec2::new(origin.x, origin.z);
        let job = WalkJob {
            recipe: recipe.key.clone(),
            from: from - corner,
            to: to - corner,
            obstacles: Self::obstacles(state),
            options: *options,
        };
        Some((entity, origin, job.input()))
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
        let TerrainSource::Generated(authored) = &state.terrain.source else {
            return None;
        };
        let layers = state.baked.as_ref().unwrap_or(authored);
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
                baking: state.bake.is_some(),
                hydrology: state.hydrology,
                erosion: state.erosion,
            })
    }

    /// Each level's largest measured height difference from the next
    /// coarser one (meters; zero until measured) and the range it drew to
    /// at the last selection (meters from the eye), finest first, of
    /// `entity`'s terrain.
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
