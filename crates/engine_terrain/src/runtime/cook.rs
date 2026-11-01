use super::*;

const COOK_VIEW_HEIGHT: f32 = 2160.0;

pub(super) struct CookTiles {
    keys: Vec<TileKey>,
    next: usize,
    flight: TileMap<Ticket>,
}

impl CookTiles {
    fn of(keys: Vec<TileKey>) -> Self {
        Self {
            keys,
            next: 0,
            flight: TileMap::default(),
        }
    }

    pub(super) fn asked(self) -> Vec<Ticket> {
        self.flight.into_values()
    }

    fn take(
        &mut self,
        kind: &str,
        asking: &mut Asking<'_, '_, '_>,
        (failures, frame): (&mut JobFailures, u64),
    ) {
        for key in self.flight.keys() {
            let Some(output) = self.flight.get(&key).and_then(|ticket| asking.take(ticket)) else {
                continue;
            };
            if let Some(ticket) = self.flight.remove(&key) {
                asking.jobs.cancel(ticket);
            }
            if let Err(error) = job_output(kind, output, |_| Ok::<_, Error>(())) {
                failures.record(frame, error);
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
        asking: &mut Asking<'_, '_, '_>,
    ) -> u32 {
        self.frame += 1;
        let frame = self.frame;
        self.sync_surroundings();
        let mut released = Vec::new();
        let mut left = 0;
        for state in &mut self.entries {
            let authored = state.terrain.layers.clone();
            if state.terrain.validate().is_err() {
                continue;
            }
            state.update_materials(settings.gpu_materials, asking.kinds);
            if bake(state, &authored, asking, frame) {
                left += 1;
                continue;
            }
            share_recipe(state, asking.jobs, &mut self.stacks, &mut released);
            let Some(recipe) = state.recipe.as_ref().map(|recipe| recipe.key) else {
                continue;
            };
            let mut cook = state
                .cook
                .take()
                .unwrap_or_else(|| CookTiles::of(cooked_tiles(state, settings)));
            cook.take(TILE_BUILD.name(), asking, (&mut state.failures, frame));
            while let Some(&key) = cook.keys.get(cook.next) {
                let priority = job_priority(VIEW_JOBS, cook.next as u32);
                let job = state.tile_job(recipe.hash(), key);
                let ticket = asking.jobs.fetch(&TILE_BUILD, &job.identity(), priority);
                cook.flight.insert(key, ticket);
                cook.next += 1;
            }
            left += cook.left();
            state.cook = Some(cook);
        }
        u32::try_from(left).unwrap_or(u32::MAX)
    }
}

fn cooked_tiles(state: &TerrainState, settings: &TerrainSettings) -> Vec<TileKey> {
    let pyramid = state.terrain.pyramid;
    let ranges = LodRanges::new(
        &pyramid,
        settings.pixel_error,
        COOK_VIEW_HEIGHT,
        vertical_fov(&Camera::default()),
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
