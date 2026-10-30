use super::*;

const COOK_VIEW_HEIGHT: f32 = 2160.0;

pub(super) struct CookTiles {
    keys: Vec<TileKey>,
    next: usize,
    flight: TileMap<JobId>,
}

impl CookTiles {
    fn of(keys: Vec<TileKey>) -> Self {
        Self {
            keys,
            next: 0,
            flight: TileMap::default(),
        }
    }

    pub(super) fn asked(&self) -> Vec<JobId> {
        self.flight.iter().map(|(_, &id)| id).collect()
    }

    fn take(&mut self, kind: &str, jobs: &mut Jobs, (failures, frame): (&mut JobFailures, u64)) {
        for (key, id) in self
            .flight
            .iter()
            .map(|(key, &id)| (key, id))
            .collect::<Vec<_>>()
        {
            if let Some(output) = jobs.take(id) {
                self.flight.remove(&key);
                if let Err(error) = job_output(kind, output, |_| Ok::<_, String>(())) {
                    failures.record(frame, error);
                }
            }
        }
    }

    fn left(&self) -> u64 {
        (self.keys.len() - self.next + self.flight.len()) as u64
    }
}

impl Terrains {
    pub(crate) fn cook(
        &mut self,
        settings: &TerrainSettings,
        water: &Water,
        jobs: &mut Jobs,
        cooking: &mut Cooking,
    ) {
        self.frame += 1;
        let frame = self.frame;
        self.sync_surroundings();
        let mut slots = JobBudget::of(jobs);
        let mut released = Vec::new();
        let (mut left, mut water_left) = (0, 0);
        for state in &mut self.entries {
            let streams = state.around.is_none() || settings.surroundings;
            if streams && lit(settings) && state.terrain.validate().is_ok() {
                left += u64::from(state.whole.lighting.left());
            }
            if state.around.is_none() && bounces(settings) && state.terrain.validate().is_ok() {
                left += u64::from(state.whole.bounce.left());
            }
            if streams && settings.enabled && state.terrain.validate().is_ok() {
                left += u64::from(state.whole.colors.left());
            }
            let authored = state.terrain.layers.clone();
            if state.terrain.validate().is_err() {
                continue;
            }
            state.update_materials(settings.gpu_materials);
            if bake(state, &authored, &mut slots, frame) {
                left += 1;
                continue;
            }
            share_recipe(state, slots.jobs, &mut self.stacks, &mut released);
            let Some(recipe) = state.recipe.as_ref().map(|recipe| recipe.key.clone()) else {
                continue;
            };
            let mut cook = state
                .cook
                .take()
                .unwrap_or_else(|| CookTiles::of(cooked_tiles(state, settings)));
            cook.take(
                TileJob::KIND.name(),
                slots.jobs,
                (&mut state.failures, frame),
            );
            while let Some(&key) = cook.keys.get(cook.next) {
                let priority = job_priority(VIEW_JOBS, cook.next as u32);
                if !slots.take(priority) {
                    break;
                }
                let (job, record) = state.tile_record(&recipe, key);
                let id = slots.jobs.submit_record(
                    TileJob::KIND.name(),
                    job.identity(),
                    priority,
                    record,
                );
                cook.flight.insert(key, id);
                cook.next += 1;
            }
            left += cook.left();
            state.cook = Some(cook);
            if state.around.is_none() {
                state.shading.cooking = true;
                left += state.shading.left();
            }
            let watering = water.enabled && state.around.is_none();
            if watering {
                state.settled.cooking = true;
                water_left += state.settled.left();
            }
        }
        cooking.left("terrain", left);
        cooking.left("water", water_left);
    }
}

fn cooked_tiles(state: &TerrainState, settings: &TerrainSettings) -> Vec<TileKey> {
    let pyramid = state.terrain.pyramid;
    let ranges = LodRanges::new(
        &pyramid,
        settings.pixel_error,
        COOK_VIEW_HEIGHT,
        Camera::default().vertical_fov_radians,
    );
    let mut keys = Vec::new();
    for level in (0..pyramid.levels()).rev() {
        let [columns, rows] = pyramid.tiles(level);
        let size = pyramid.tile_size(level);
        for z in 0..rows {
            for x in 0..columns {
                let first = Vec2::new(x as f32, z as f32) * size;
                let last = first + Vec2::splat(size);
                let wanted = state.around.as_ref().is_none_or(|around| {
                    let (low, high) = around.cutout;
                    let inside = first.x >= low.x
                        && first.y >= low.y
                        && last.x <= high.x
                        && last.y <= high.y;
                    let gap = (low - last).max(first - high).max(Vec2::ZERO).length();
                    !inside && gap <= LOAD_REACH * ranges.range(level)
                });
                if wanted {
                    keys.push(TileKey::new(level as u8, x as u16, z as u16));
                }
            }
        }
    }
    keys
}
