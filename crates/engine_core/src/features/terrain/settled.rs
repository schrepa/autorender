//! Terrains' settled water: each level-0 tile's rivers at their steady
//! state ([`SteadyWater`]), records the cook makes for every tile its
//! bake's water reaches and hosts keep, at the cells' size water's
//! simulation takes ([`Water::simulation_cell`]). Frames ask for the
//! records of the resident level-0 tiles water reaches, made where no host
//! holds them, and fetch every other tile's where a host holds it, nearest
//! first; they keep resident level-0 tiles' records whole and every
//! record's summary ([`SteadySummary`]). Resident level-0 tiles take the
//! water their records and their neighbors' stand at their samples
//! ([`Tile::steadied`]), coarser tiles the flow and foam of the summaries
//! under them ([`Tile::flowing`]): drawing, queries and the window
//! simulated around the camera read water as it settles, near and far. A
//! record made under other edits than the ground has now stands until the
//! one made under them arrives. A tile whose rivers something standing in
//! them holds back keeps the water they run with now.

use super::*;
use crate::MISSING_RECORD;
use engine_terrain::jobs::WaterJob;
use engine_terrain::{SUMMARY_SIDE, SteadyCell, SteadySquare, SteadySummary, SteadyWater};

/// A record of settled water asked for: its job, its key, and whether it
/// is made where no host holds it (a resident level-0 tile's) or only
/// fetched (for its summary).
#[derive(Clone, Copy)]
struct Asked {
    id: JobId,
    record: RecordKey,
    made: bool,
}

/// A terrain's settled water: the cells' size its records hold; the
/// level-0 tiles its bake's water reaches, found over its baked layers;
/// the generation of tiles its records' keys were found for, and each
/// watered tile's record's key then; the records of resident level-0 tiles
/// held whole and every record's summary, each with its key; the records
/// asked for, those that failed (at the frame they did) and those no host
/// holds; per tile, the latest of the records its samples read to arrive,
/// arrivals counted; and the resident tiles given their water, as given
/// and when.
#[derive(Default)]
pub(super) struct Settled {
    cell: f32,
    watered: Option<(Arc<Layers>, Arc<[TileKey]>)>,
    checked: Option<Generation>,
    records: TileMap<RecordKey>,
    pub(super) held: TileMap<(Arc<SteadyWater>, RecordKey)>,
    summaries: TileMap<(SteadySummary, RecordKey)>,
    asked: TileMap<Asked>,
    failed: TileMap<u64>,
    missing: TileMap<RecordKey>,
    changed: TileMap<u64>,
    given: TileMap<(Arc<Tile>, u64)>,
    arrived: u64,
}

impl Settled {
    /// The jobs of its records asked for.
    pub(super) fn asked(&self) -> Vec<JobId> {
        self.asked.iter().map(|(_, asked)| asked.id).collect()
    }

    /// Records arrived, whose summaries it holds, and those asked for.
    pub(super) fn counts(&self) -> (u32, u32) {
        (self.summaries.len() as u32, self.asked.len() as u32)
    }

    /// The meters across its records' cells.
    pub(super) fn cell(&self) -> f32 {
        self.cell
    }

    /// Whether the settled water of level-0 tile `key` is known: its record
    /// is held whole, or its bake's water reaches it not, so it settles
    /// dry.
    pub(super) fn has(&self, key: TileKey) -> bool {
        self.held.contains_key(&key)
            || self
                .watered
                .as_ref()
                .is_some_and(|(_, watered)| watered.binary_search(&key).is_err())
    }

    /// The settled water of the terrain's lattice cell `(i, j)`: what the
    /// record of the level-0 tile holding its middle says, if held whole.
    pub(super) fn cell_at(&self, pyramid: &Pyramid, i: i64, j: i64) -> Option<SteadyCell> {
        let middle = Vec2::new((i as f32 + 0.5) * self.cell, (j as f32 + 0.5) * self.cell);
        let key = pyramid.tile_at(0, middle)?;
        self.held.get(&key)?.0.at(i, j)
    }

    /// Square `(i, j)` of the terrain's lattice of summaries' squares,
    /// [`SUMMARY_SIDE`] along each side of a level-0 tile: what the summary
    /// of the tile holding it says, if held.
    fn square_at(&self, i: i64, j: i64) -> Option<SteadySquare> {
        let side = i64::from(SUMMARY_SIDE);
        if i < 0 || j < 0 {
            return None;
        }
        let x = u16::try_from(i / side).ok()?;
        let z = u16::try_from(j / side).ok()?;
        let (summary, _) = self.summaries.get(&TileKey::new(0, x, z))?;
        summary.square((i % side) as u32, (j % side) as u32)
    }

    /// Counts a record of level-0 tile `key` of `pyramid` arrived: the
    /// tiles whose samples read it change, the tile and its neighbors and
    /// every tile over them.
    fn arrive(&mut self, pyramid: &Pyramid, key: TileKey) {
        self.arrived += 1;
        let [columns, rows] = pyramid.tiles(0);
        for dz in -1..=1_i32 {
            for dx in -1..=1_i32 {
                let (x, z) = (i32::from(key.x) + dx, i32::from(key.z) + dz);
                if x < 0 || z < 0 || x >= columns as i32 || z >= rows as i32 {
                    continue;
                }
                let mut over = Some(TileKey::new(0, x as u16, z as u16));
                while let Some(tile) = over {
                    self.changed.insert(tile, self.arrived);
                    over = pyramid.parent(tile);
                }
            }
        }
    }
}

impl TerrainState {
    /// The job of level-0 tile `key`'s settled water over the stack shared
    /// as `stack`, carrying the edits reaching the ground it settles over,
    /// in cells `cell` meters across, and the key of its record.
    pub(super) fn water_record(
        &self,
        stack: &str,
        key: TileKey,
        cell: f32,
    ) -> (WaterJob, RecordKey) {
        let pyramid = self.terrain.pyramid;
        let (low, high) = engine_terrain::steady_water_bounds(&pyramid, key, cell);
        let job = WaterJob {
            stack: stack.to_owned(),
            pyramid,
            key,
            stamps: self
                .terrain
                .edits
                .iter()
                .map(|edit| edit.stamp)
                .filter(|stamp| {
                    let radius = Vec2::splat(stamp.radius + stamp.falloff);
                    let center = Vec2::from(stamp.center);
                    let (from, to) = (center - radius, center + radius);
                    from.cmple(high).all() && to.cmpge(low).all()
                })
                .collect(),
            cell,
        };
        let record = RecordKey::new(
            &engine_terrain::SOURCE,
            WaterJob::KIND.name(),
            &job.identity(),
        );
        (job, record)
    }
}

/// Cancels every record `state`'s settled water asked for, giving `slots`
/// theirs back.
fn cancel_all(state: &TerrainState, slots: &mut JobBudget<'_>) {
    for id in state.settled.asked() {
        slots.jobs.cancel(id);
        slots.release();
    }
}

/// Follows `state`'s settled water for this frame, where `water` asks for
/// water: takes the records that arrived; asks for the records of the
/// resident level-0 tiles its bake's water reaches and fetches the other
/// watered tiles' while `slots` admits them, nearest `eye`
/// (terrain-local) first, those of tiles no longer resident made no more;
/// and gives resident tiles the water their records stand. Without water,
/// or without a bake's water, it holds nothing.
pub(super) fn follow(
    state: &mut TerrainState,
    water: Option<&Water>,
    slots: &mut JobBudget<'_>,
    (eye, frame): (Vec2, u64),
) {
    let Some((stack, watered)) = refresh(state, water, slots) else {
        return;
    };
    let near: Vec<TileKey> = state
        .store
        .keys_where(|key| key.level == 0 && watered.binary_search(key).is_ok());
    take_arrived(state, slots, &near, frame);
    let settled = &mut state.settled;
    settled
        .held
        .retain(|key, _| near.binary_search(key).is_ok());
    let leaving = settled
        .asked
        .keys_where(|key, asked| asked.made && near.binary_search(key).is_err());
    for key in leaving {
        if let Some(asked) = settled.asked.remove(&key) {
            slots.jobs.cancel(asked.id);
            slots.release();
        }
    }
    ask(state, slots, (&stack, &watered, &near), (eye, frame));
    give(state, &near);
}

/// Readies `state`'s settled water for `water`: none without water or a
/// bake's water (what it asked for cancelled, `slots` given theirs back),
/// anew over other baked layers or cells' size, and each watered tile's
/// record's key found again under other edits, the records asked for under
/// others cancelled. The stack its records settle over and the level-0
/// tiles its bake's water reaches, in key order.
fn refresh(
    state: &mut TerrainState,
    water: Option<&Water>,
    slots: &mut JobBudget<'_>,
) -> Option<(String, Arc<[TileKey]>)> {
    let asking = water.filter(|water| water.enabled && state.around.is_none());
    let (Some(water), Some(stack), Some(baked)) = (
        asking,
        state.recipe.as_ref().map(|recipe| recipe.stack_key.clone()),
        state.baked.clone(),
    ) else {
        cancel_all(state, slots);
        state.settled = Settled::default();
        return None;
    };
    let current = state.settled.cell == water.simulation_cell
        && state
            .settled
            .watered
            .as_ref()
            .is_some_and(|(over, _)| Arc::ptr_eq(over, &baked));
    if !current {
        cancel_all(state, slots);
        let watered: Arc<[TileKey]> = match state.settled.watered.take() {
            Some((over, keys)) if Arc::ptr_eq(&over, &baked) => keys,
            _ => engine_terrain::watered_tiles(&baked, &state.terrain.pyramid).into(),
        };
        state.settled = Settled {
            cell: water.simulation_cell,
            watered: Some((baked, watered)),
            ..Settled::default()
        };
    }
    let watered = state
        .settled
        .watered
        .as_ref()
        .map(|(_, keys)| keys.clone())?;
    let generation = state.store.generation();
    if state.settled.checked != Some(generation) {
        let cell = state.settled.cell;
        let mut records = TileMap::default();
        for &key in watered.iter() {
            records.insert(key, state.water_record(&stack, key, cell).1);
        }
        let settled = &mut state.settled;
        settled.records = records;
        settled.checked = Some(generation);
        let records = &settled.records;
        let stale = settled
            .asked
            .keys_where(|key, asked| records.get(key) != Some(&asked.record));
        for key in stale {
            if let Some(asked) = settled.asked.remove(&key) {
                slots.jobs.cancel(asked.id);
                slots.release();
            }
        }
    }
    Some((stack, watered))
}

/// Asks, while `slots` admits them, for the records over the stack shared
/// as `stack` that `state`'s settled water lacks under the edits the ground
/// has now, nearest `eye` (terrain-local) first: those of the resident
/// level-0 tiles `near`, made where no host holds them, then those of the
/// other tiles of `watered`, fetched where a host holds them. A record
/// that failed is asked for again a while after it did, `frame` now.
fn ask(
    state: &mut TerrainState,
    slots: &mut JobBudget<'_>,
    (stack, watered, near): (&str, &[TileKey], &[TileKey]),
    (eye, frame): (Vec2, u64),
) {
    let pyramid = state.terrain.pyramid;
    let size = pyramid.tile_size(0);
    let nearness = |key: TileKey| {
        let middle = pyramid.tile_origin(key) + Vec2::splat(0.5 * size);
        (middle.distance(eye) / size) as u32
    };
    let settled = &state.settled;
    let waiting = |key: TileKey| {
        !settled.asked.contains_key(&key)
            && settled
                .failed
                .get(&key)
                .is_none_or(|&at| frame >= at + RETRY_AFTER)
    };
    let mut making: Vec<(u32, TileKey)> = near
        .iter()
        .filter(|&&key| {
            settled.held.get(&key).map(|(_, record)| record) != settled.records.get(&key)
                && waiting(key)
        })
        .map(|&key| (nearness(key), key))
        .collect();
    making.sort_unstable();
    let mut fetching: Vec<(u32, TileKey)> = watered
        .iter()
        .filter(|&&key| {
            let record = settled.records.get(&key);
            near.binary_search(&key).is_err()
                && settled.summaries.get(&key).map(|(_, held)| held) != record
                && settled.missing.get(&key) != record
                && waiting(key)
        })
        .map(|&key| (nearness(key), key))
        .collect();
    fetching.sort_unstable();
    for (place, key) in making {
        let priority = job_priority(VIEW_JOBS, place);
        if !slots.take(priority) {
            break;
        }
        let (job, record) = state.water_record(stack, key, state.settled.cell);
        let id = slots
            .jobs
            .submit_record(WaterJob::KIND.name(), job.identity(), priority, record);
        let asked = Asked {
            id,
            record,
            made: true,
        };
        state.settled.asked.insert(key, asked);
    }
    for (place, key) in fetching {
        let priority = job_priority(IDLE_JOBS, place);
        let Some(&record) = state.settled.records.get(&key) else {
            continue;
        };
        if !slots.take(priority) {
            break;
        }
        let id = slots.jobs.fetch(record, priority);
        let asked = Asked {
            id,
            record,
            made: false,
        };
        state.settled.asked.insert(key, asked);
    }
}

/// Takes the records of `state`'s settled water that arrived, holding
/// those of the resident level-0 tiles `near` whole and every one's
/// summary; a record fetched that no host holds is not fetched again.
fn take_arrived(state: &mut TerrainState, slots: &mut JobBudget<'_>, near: &[TileKey], frame: u64) {
    let pyramid = state.terrain.pyramid;
    for (key, asked) in state
        .settled
        .asked
        .iter()
        .map(|(key, &asked)| (key, asked))
        .collect::<Vec<_>>()
    {
        let Some(output) = slots.jobs.take(asked.id) else {
            continue;
        };
        let settled = &mut state.settled;
        settled.asked.remove(&key);
        if !asked.made && matches!(&output, Err(error) if error.as_str() == MISSING_RECORD) {
            settled.missing.insert(key, asked.record);
            continue;
        }
        match job_output(
            WaterJob::KIND.name(),
            output,
            engine_terrain::jobs::water_output,
        ) {
            Ok(water) if water.key == key && water.cell == settled.cell => {
                settled.failed.remove(&key);
                settled.missing.remove(&key);
                settled
                    .summaries
                    .insert(key, (water.summary.clone(), asked.record));
                if near.binary_search(&key).is_ok() {
                    settled.held.insert(key, (Arc::new(water), asked.record));
                }
                settled.arrive(&pyramid, key);
            }
            Ok(_) => {
                settled.failed.insert(key, frame);
            }
            Err(error) => {
                settled.failed.insert(key, frame);
                state.failures.record(frame, error);
            }
        }
    }
}

/// Gives `state`'s resident tiles the water their settled water stands:
/// the level-0 tiles of `near` whose records are held whole the water
/// their records and their neighbors' stand at their samples, coarser
/// tiles holding water the flow and foam of the summaries under them;
/// those given none yet, those that arrived anew since, and those whose
/// samples read a record that arrived since. Tiles whose rivers something
/// standing in them holds back keep the water those rivers run with.
fn give(state: &mut TerrainState, near: &[TileKey]) {
    let pyramid = state.terrain.pyramid;
    let resident = state
        .store
        .keys_where(|key| key.level > 0 || near.binary_search(key).is_ok());
    state
        .settled
        .given
        .retain(|key, _| resident.binary_search(key).is_ok());
    for key in resident {
        let latest = state.settled.changed.get(&key).copied().unwrap_or(0);
        if latest == 0
            || (key.level == 0 && !state.settled.held.contains_key(&key))
            || state.store.watered(&key).is_some_and(Option::is_some)
        {
            continue;
        }
        let Some(tile) = state.store.get(&key).cloned() else {
            continue;
        };
        if key.level > 0 && tile.water().is_none() {
            continue;
        }
        let current = state
            .settled
            .given
            .get(&key)
            .is_some_and(|(given, at)| Arc::ptr_eq(given, &tile) && *at >= latest);
        if current {
            continue;
        }
        let settled = &state.settled;
        let given = Arc::new(if key.level == 0 {
            tile.steadied(&pyramid, settled.cell, &|i, j| {
                settled.cell_at(&pyramid, i, j)
            })
        } else {
            tile.flowing(&pyramid, &|i, j| settled.square_at(i, j))
        });
        state.store.rewater(&key, Some(given.clone()), None);
        let arrived = state.settled.arrived;
        state.settled.given.insert(key, (given, arrived));
    }
}
