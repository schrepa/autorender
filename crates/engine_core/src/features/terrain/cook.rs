//! Cooking terrains, while a session cooks ([`Cooking`]): each generated
//! terrain's bakes, then every tile a camera on it could want, as records
//! asked for coarsest level first and dropped as they arrive, which the
//! host keeps for its pack. A terrain's surroundings cook the tiles within
//! reach of the ground they surround, as a view from it wants them. Every
//! streaming terrain's lighting cooks as streaming asks for it.

use super::*;

/// The tallest view, in pixels, whose wants of surroundings the cook
/// covers: past what hosts draw.
const COOK_VIEW_HEIGHT: f32 = 2160.0;

/// A terrain's tiles cooking: the tiles it makes, in order, the next to ask
/// for, and the records asked for and not yet back.
pub(super) struct CookTiles {
    keys: Vec<TileKey>,
    next: usize,
    flight: TileMap<JobId>,
}

impl CookTiles {
    /// The jobs of its records asked for.
    pub(super) fn asked(&self) -> Vec<JobId> {
        self.flight.iter().map(|(_, &id)| id).collect()
    }
}

impl Terrains {
    /// Cooks every generated terrain within `settings`, asking for records
    /// as `jobs` admits them; says in `cooking` how many are left: a bake
    /// under way counts one, each tile not back yet one, and each terrain
    /// streaming whose lighting is still asked for or not yet made one.
    pub(crate) fn cook(
        &mut self,
        settings: &TerrainSettings,
        jobs: &mut Jobs,
        cooking: &mut Cooking,
    ) {
        self.frame += 1;
        let frame = self.frame;
        self.sync_surroundings();
        let mut slots = JobBudget::of(jobs);
        let mut released = Vec::new();
        let mut left = 0;
        for state in &mut self.entries {
            let streams = state.around.is_none() || settings.surroundings;
            if streams && lit(settings) && state.terrain.validate().is_ok() {
                left += u64::from(state.lighting.left());
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
            if state.cook.is_none() {
                state.cook = Some(CookTiles {
                    keys: cooked_tiles(state, settings),
                    next: 0,
                    flight: TileMap::default(),
                });
            }
            let Some(mut cook) = state.cook.take() else {
                continue;
            };
            for (key, id) in cook
                .flight
                .iter()
                .map(|(key, &id)| (key, id))
                .collect::<Vec<_>>()
            {
                if let Some(output) = slots.jobs.take(id) {
                    cook.flight.remove(&key);
                    if let Err(error) =
                        job_output(TileJob::KIND.name(), output, |_| Ok::<_, String>(()))
                    {
                        state.failures.record(frame, error);
                    }
                }
            }
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
            left += (cook.keys.len() - cook.next + cook.flight.len()) as u64;
            state.cook = Some(cook);
        }
        cooking.left("terrain", left);
    }
}

/// The tiles `state` cooks, coarsest level first, rows along z: every one
/// of a terrain's, and of surroundings those not inside the ground they
/// surround and within loading reach of it at their level.
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
