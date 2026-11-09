use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_ecs::world::World;
use engine_core::collections::{HashMap, HashSet};
use engine_core::{
    CookAsks, Error, ErrorCode, FrameBudget, FrameTime, IDLE_JOBS, Interest, JobFailures, Jobs,
    MISSING_RECORD, RecordKey, Records, VIEW_JOBS, Views, job_priority,
};
use engine_heightfield::Stamp;
use engine_heightfield::tile::shaded_elsewhere;
use engine_places_simulation::ground::TileKey;
use engine_places_world::ground::{TerrainEntry, TerrainTiles, TileFace};
use glam::Vec2;

use crate::jobs::gpu::TILE_STRIDE;
use crate::jobs::shading::TileShading;
use crate::jobs::{SHADING, ShadingJob, shading_output};
use crate::state::{Bakes, TerrainBakes, listed_tile};

const BUDGET: &str = "terrain_shading";

const TILE_ARRAY_BYTES: u64 = (TILE_STRIDE as u64) * (TILE_STRIDE as u64) * 4;

enum Shading {
    Looking {
        record: RecordKey,
        interest: Interest<RecordKey>,
    },
    Making {
        record: RecordKey,
        interest: Interest<RecordKey>,
        reads: Vec<RecordKey>,
    },
    Held {
        record: RecordKey,
        interest: Interest<RecordKey>,
        shading: Arc<TileShading>,
        shown: Option<engine_core::Stamp>,
    },
    Made {
        record: RecordKey,
    },
    Missing {
        record: RecordKey,
    },
    Failed {
        record: RecordKey,
        interest: Option<Interest<RecordKey>>,
    },
}

impl Shading {
    fn record(&self) -> RecordKey {
        match self {
            Self::Looking { record, .. }
            | Self::Making { record, .. }
            | Self::Held { record, .. }
            | Self::Made { record }
            | Self::Missing { record }
            | Self::Failed { record, .. } => *record,
        }
    }

    fn asked(&self) -> bool {
        matches!(self, Self::Looking { .. } | Self::Making { .. })
    }

    fn arrived(&self) -> bool {
        matches!(self, Self::Held { .. } | Self::Made { .. })
    }

    fn held(&self) -> bool {
        matches!(self, Self::Held { .. })
    }

    fn interest(&self) -> Option<&Interest<RecordKey>> {
        match self {
            Self::Looking { interest, .. }
            | Self::Making { interest, .. }
            | Self::Held { interest, .. } => Some(interest),
            Self::Failed { interest, .. } => interest.as_ref(),
            Self::Made { .. } | Self::Missing { .. } => None,
        }
    }
}

#[derive(Clone, Copy)]
struct Want {
    key: TileKey,
    priority: u32,
    held: bool,
}

#[derive(Default)]
pub(crate) struct Shadings {
    basis: Option<([u8; 32], Arc<[Stamp]>)>,
    tiles: BTreeMap<TileKey, Shading>,
    sharing: BTreeSet<RecordKey>,
    attached: Vec<TileKey>,
    attaching: bool,
    shaded: u32,
}

impl Shadings {
    pub(crate) fn shown(&self, tile: &TileFace) -> Option<&Arc<TileShading>> {
        if let Some(Shading::Held { shading, shown, .. }) = self.tiles.get(&tile.key)
            && *shown == Some(tile.stamp)
        {
            Some(shading)
        } else {
            None
        }
    }

    pub(crate) fn attached(&self) -> &[TileKey] {
        &self.attached
    }

    pub(crate) fn counts(&self) -> (u32, u32) {
        let asked = self.tiles.values().filter(|tile| tile.asked()).count();
        (self.shaded, asked as u32)
    }

    pub(crate) fn holds_share(&self, key: &RecordKey) -> bool {
        self.sharing.contains(key)
    }

    pub(crate) fn release(self) -> BTreeSet<RecordKey> {
        self.sharing
    }

    fn made(&self, key: TileKey, record: RecordKey) -> bool {
        let missing = matches!(
            self.tiles.get(&key),
            Some(Shading::Missing { record: at } | Shading::Making { record: at, .. }) if *at == record
        );
        key.level == 1 || missing
    }

    fn rebase(&mut self, entry: &TerrainEntry, recipe: [u8; 32]) {
        let same = self
            .basis
            .as_ref()
            .is_some_and(|(held, stamps)| *held == recipe && stamps[..] == entry.edits[..]);
        self.basis = Some((recipe, Arc::clone(&entry.edits)));
        if same {
            return;
        }
        let record =
            |key: TileKey| ShadingJob::of(recipe, &entry.pyramid, &entry.edits, key).record();
        self.tiles.retain(|&key, tile| tile.record() == record(key));
    }

    fn take_arrived(
        &mut self,
        records: &Records,
        (failures, frame): (&mut JobFailures, u64),
    ) -> bool {
        let kind = SHADING.name();
        let mut took = false;
        for (&key, tile) in &mut self.tiles {
            if !tile.asked() {
                continue;
            }
            let Some(output) = tile.interest().and_then(|interest| records.get(interest)) else {
                continue;
            };
            took = true;
            let record = tile.record();
            let (looking, interest) = match std::mem::replace(tile, Shading::Missing { record }) {
                Shading::Looking { interest, .. } => (true, interest),
                Shading::Making { interest, .. } => (false, interest),
                Shading::Held { .. }
                | Shading::Made { .. }
                | Shading::Missing { .. }
                | Shading::Failed { .. } => continue,
            };
            *tile = match output {
                Err(error) if looking && missing(&error) => Shading::Missing { record },
                Err(error) => {
                    failures.record(frame, format!("{kind}: {error}"));
                    Shading::Failed {
                        record,
                        interest: Some(interest),
                    }
                }
                Ok(bytes) => match shading_output(&bytes) {
                    Ok(made) if made.key() == key => Shading::Held {
                        record,
                        interest,
                        shading: Arc::new(made),
                        shown: None,
                    },
                    Ok(_) => Shading::Failed {
                        record,
                        interest: Some(interest),
                    },
                    Err(error) => {
                        failures.record(frame, error.within(&format!("$.{kind}")).to_string());
                        Shading::Failed {
                            record,
                            interest: Some(interest),
                        }
                    }
                },
            };
        }
        took
    }

    fn wanted(&self, entry: &TerrainEntry, recipe: [u8; 32], eye: Vec2) -> Vec<Want> {
        let pyramid = entry.pyramid;
        let levels = pyramid.levels();
        let size = pyramid.tile_size(0);
        let pinned = entry.pinned_levels;
        let place = |key: TileKey| {
            let span = pyramid.tile_size(u32::from(key.level));
            let middle = pyramid.tile_origin(key) + Vec2::splat(0.5 * span);
            let pinned_level = u32::from(key.level) + pinned >= levels;
            (!pinned_level, (middle.distance(eye) / size) as u32, key)
        };
        let mut resident: Vec<TileKey> = entry
            .tiles
            .iter()
            .map(|tile| tile.key)
            .filter(|key| shaded_elsewhere(key.level, false))
            .collect();
        resident.sort_unstable_by_key(|&key| place(key));
        let mut wants: Vec<Want> = Vec::with_capacity(resident.len());
        let mut listed: HashMap<TileKey, usize> = HashMap::default();
        let mut list =
            |wants: &mut Vec<Want>, key: TileKey, class: u8, held: bool| match listed.get(&key) {
                Some(&at) => wants[at].held |= held,
                None => {
                    listed.insert(key, wants.len());
                    let priority = job_priority(class, wants.len() as u32);
                    wants.push(Want {
                        key,
                        priority,
                        held,
                    });
                }
            };
        for &key in &resident {
            list(&mut wants, key, VIEW_JOBS, true);
        }
        let mut at = 0;
        while let Some(&want) = wants.get(at) {
            at += 1;
            if want.key.level < 2 || self.tiles.get(&want.key).is_some_and(Shading::arrived) {
                continue;
            }
            let record = ShadingJob::of(recipe, &pyramid, &entry.edits, want.key).record();
            if !self.made(want.key, record) {
                continue;
            }
            for read in TileShading::reads(&pyramid, want.key) {
                list(&mut wants, read, IDLE_JOBS, true);
            }
        }
        wants
    }

    fn keep(&mut self, wants: &[Want]) {
        let mut wanted: HashMap<TileKey, bool> = HashMap::default();
        for want in wants {
            wanted.insert(want.key, want.held);
        }
        self.tiles.retain(|key, _| wanted.contains_key(key));
        for (key, tile) in &mut self.tiles {
            let record = tile.record();
            if tile.held() && !wanted.get(key).copied().unwrap_or(false) {
                *tile = Shading::Made { record };
            }
        }
    }

    fn ask(
        &mut self,
        cx: &mut Follow<'_, '_, '_>,
        wants: &[Want],
        (entry, recipe): (&TerrainEntry, [u8; 32]),
    ) {
        for want in wants {
            let key = want.key;
            let has = match self.tiles.get(&key) {
                None | Some(Shading::Missing { .. }) => false,
                Some(Shading::Made { .. }) => !want.held,
                Some(
                    Shading::Looking { .. }
                    | Shading::Making { .. }
                    | Shading::Held { .. }
                    | Shading::Failed { .. },
                ) => true,
            };
            if has {
                continue;
            }
            let input =
                engine_codec::encode(&ShadingJob::of(recipe, &entry.pyramid, &entry.edits, key));
            let record = RecordKey::of(&SHADING, &input);
            let made = self.made(key, record);
            let finer = if key.level > 1 {
                TileShading::reads(&entry.pyramid, key)
            } else {
                Vec::new()
            };
            if made
                && !finer
                    .iter()
                    .all(|at| self.tiles.get(at).is_some_and(Shading::held))
            {
                continue;
            }
            let tile = if made {
                let mut reads = Vec::new();
                for at in &finer {
                    let Some(Shading::Held {
                        record: read,
                        interest,
                        ..
                    }) = self.tiles.get(at)
                    else {
                        continue;
                    };
                    reads.push(*read);
                    if self.sharing.insert(*read) && cx.shared.insert(*read) {
                        cx.jobs.share_record(interest);
                    }
                }
                Shading::Making {
                    record,
                    interest: cx.jobs.start(&SHADING, input, want.priority),
                    reads,
                }
            } else {
                Shading::Looking {
                    record,
                    interest: cx.jobs.look(&SHADING, &input, want.priority),
                }
            };
            self.tiles.insert(key, tile);
        }
    }

    fn release_shares(&mut self, released: &mut Vec<RecordKey>) {
        let reading: HashSet<RecordKey> = self
            .tiles
            .values()
            .filter_map(|tile| match tile {
                Shading::Making { reads, .. } => Some(reads.iter().copied()),
                Shading::Looking { .. }
                | Shading::Held { .. }
                | Shading::Made { .. }
                | Shading::Missing { .. }
                | Shading::Failed { .. } => None,
            })
            .flatten()
            .collect();
        let shares: Vec<RecordKey> = self.sharing.iter().copied().collect();
        for key in shares {
            if !reading.contains(&key) {
                self.sharing.remove(&key);
                released.push(key);
            }
        }
    }

    fn attach(&mut self, entry: &TerrainEntry, budget: &mut FrameBudget) {
        self.attaching = false;
        self.shaded = 0;
        let mut gave = false;
        for (&key, tile) in &mut self.tiles {
            let Shading::Held { shown, .. } = tile else {
                continue;
            };
            let Some(face) = listed_tile(&entry.tiles, key) else {
                continue;
            };
            if *shown == Some(face.stamp) {
                self.shaded += 1;
                continue;
            }
            if !face.current {
                continue;
            }
            if (gave && budget.expired()) || !budget.spend(BUDGET, 2 * TILE_ARRAY_BYTES) {
                self.attaching = true;
                continue;
            }
            gave = true;
            *shown = Some(face.stamp);
            self.attached.push(key);
            self.shaded += 1;
        }
    }
}

pub(crate) fn cook(world: &World, asks: &mut CookAsks<'_>) {
    let Some(tiles) = world.get_resource::<TerrainTiles>() else {
        return;
    };
    for entry in tiles.terrains.iter() {
        let Some(recipe) = entry.recipe.filter(|_| !entry.surroundings) else {
            continue;
        };
        let pyramid = entry.pyramid;
        let record = |key: TileKey| ShadingJob::of(recipe, &pyramid, &entry.edits, key).record();
        for level in 1..pyramid.levels() {
            for key in pyramid.level_tiles(level) {
                let input = engine_codec::encode(&ShadingJob::of(recipe, &pyramid, &entry.edits, key));
                if asks.asked(&RecordKey::of(&SHADING, &input)) {
                    continue;
                }
                let reads = if key.level > 1 {
                    TileShading::reads(&pyramid, key)
                        .into_iter()
                        .map(record)
                        .collect()
                } else {
                    Vec::new()
                };
                asks.ask_reading(&SHADING, input, reads);
            }
        }
    }
}

fn missing(error: &Error) -> bool {
    error.code == ErrorCode::InvalidRecord
        && error
            .detail
            .got
            .as_ref()
            .and_then(serde_json::Value::as_str)
            == Some(MISSING_RECORD)
}

struct Follow<'a, 'w, 's> {
    jobs: &'a mut Jobs<'w, 's>,
    records: &'a Records,
    failures: &'a mut JobFailures,
    shared: &'a mut BTreeSet<RecordKey>,
    released: &'a mut Vec<RecordKey>,
    budget: &'a mut FrameBudget,
}

pub(crate) fn shade(
    tiles: Res<TerrainTiles>,
    records: Res<Records>,
    views: Res<Views>,
    time: Res<FrameTime>,
    mut budget: ResMut<FrameBudget>,
    mut jobs: Jobs,
    mut bakes: ResMut<Bakes>,
) {
    let eye = views.main().map_or(glam::Vec3::ZERO, |main| main.detail_eye);
    let frame = time.frame;
    let moved = tiles.is_changed();
    let bakes = &mut *bakes;
    let mut shared: BTreeSet<RecordKey> = bakes
        .terrains
        .values()
        .flat_map(|held| held.flow.sharing.iter().copied())
        .collect();
    let mut released = Vec::new();
    for entry in tiles.terrains.iter() {
        let Some(held) = bakes.terrains.get_mut(&entry.terrain) else {
            continue;
        };
        let mut cx = Follow {
            jobs: &mut jobs,
            records: &records,
            failures: &mut bakes.failures,
            shared: &mut shared,
            released: &mut released,
            budget: &mut budget,
        };
        let local = Vec2::new(eye.x - entry.origin.x, eye.z - entry.origin.z);
        follow(held, entry, &mut cx, (local, frame, moved));
    }
    for key in released {
        if !bakes
            .terrains
            .values()
            .any(|held| held.flow.holds_share(&key))
        {
            jobs.unshare(key);
        }
    }
}

fn follow(
    held: &mut TerrainBakes,
    entry: &TerrainEntry,
    cx: &mut Follow<'_, '_, '_>,
    (eye, frame, moved): (Vec2, u64, bool),
) {
    let Some(recipe) = entry.recipe.filter(|_| !entry.surroundings) else {
        let left = std::mem::take(&mut held.flow);
        cx.released.extend(left.release());
        return;
    };
    let flow = &mut held.flow;
    flow.attached.clear();
    let moved = moved || flow.basis.is_none();
    if moved {
        flow.rebase(entry, recipe);
    }
    let arrived = flow.take_arrived(cx.records, (&mut *cx.failures, frame));
    let dirty = moved || arrived;
    if dirty {
        let wants = flow.wanted(entry, recipe, eye);
        flow.keep(&wants);
        flow.ask(cx, &wants, (entry, recipe));
        flow.release_shares(cx.released);
    }
    if dirty || flow.attaching {
        flow.attach(entry, cx.budget);
    }
}
