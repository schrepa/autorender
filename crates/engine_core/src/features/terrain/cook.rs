//! Cooking terrains, while a session cooks ([`Cooking`]): each generated
//! terrain's bakes, then every tile a camera on it could want, as records
//! asked for coarsest level first and dropped as they arrive, which the
//! host keeps for its pack, and where water asks for water, every level-0
//! tile's settled water its bake's water reaches, over the cook's regions,
//! and every tile's water standing as it settled, each a record of its own
//! ([`super::settled`]). A terrain's surroundings
//! cook the tiles within reach of the ground they surround, as a view from
//! it wants them. Every streaming terrain's lighting and macro colors cook
//! as streaming asks for them, the colors in every family of compressed
//! formats, and every terrain's bounce transfer after its lighting.

use super::*;

/// The tallest view, in pixels, whose wants of surroundings the cook
/// covers: past what hosts draw.
const COOK_VIEW_HEIGHT: f32 = 2160.0;

/// A terrain's tiles cooking: the tiles it makes, in order, the next to
/// ask for, and the records asked for and not yet back.
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

    /// The jobs of its records asked for.
    pub(super) fn asked(&self) -> Vec<JobId> {
        self.flight.iter().map(|(_, &id)| id).collect()
    }

    /// Drops the records that arrived, recording in `failures` at `frame`
    /// those of jobs of `kind` that failed.
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

    /// Records not back yet.
    fn left(&self) -> u64 {
        (self.keys.len() - self.next + self.flight.len()) as u64
    }
}

impl Terrains {
    /// Cooks every generated terrain within `settings`, asking for records
    /// as `jobs` admits them, the settled water too where `water` asks for
    /// water; says in `cooking` how many are left: a bake under way counts
    /// one, each tile, tile's settled water or tile's water standing on it
    /// not back yet one, each terrain
    /// streaming whose lighting is still asked for or not yet made one, each
    /// terrain (not surroundings) whose bounce transfer is one, and each
    /// terrain whose macro colors are not shown yet one (the records count
    /// each family's records of them).
    pub(crate) fn cook(
        &mut self,
        settings: &TerrainSettings,
        water: Option<&Water>,
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
            // Settled water: every watered tile's, over the cook's regions,
            // each settling once its tiles were fetched, while tiles cook.
            let watering = water.is_some_and(|water| water.enabled) && state.around.is_none();
            if watering {
                state.settled.cooking = true;
                water_left += state.settled.left();
            }
        }
        cooking.left("terrain", left);
        cooking.left("water", water_left);
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
