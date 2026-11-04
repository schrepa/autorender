use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::collections::{HashMap, HashSet};
use engine_core::{
    CookMode, Error, ErrorCode, FrameTime, IDLE_JOBS, JobFailures, Jobs, MISSING_RECORD, RecordKey,
    Records, Ticket, VIEW_JOBS, View, WorkClock, job_priority,
};
use engine_heightfield::Stamp;
use engine_heightfield::tile::shaded_elsewhere;
use engine_places::ground::{GroundWork, TerrainEntry, TerrainTiles, TileFace, TileKey};
use glam::Vec2;

use crate::jobs::{SHADING, ShadingJob, shading_output};
use crate::shading::TileShading;
use crate::state::{Bakes, RETRY_AFTER, TerrainBakes, listed_tile};

const TILE_ARRAY_BYTES: u64 =
    (crate::gpu::TILE_STRIDE as u64) * (crate::gpu::TILE_STRIDE as u64) * 4;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Fetching,
    Making,
}

struct Asked {
    phase: Phase,
    record: RecordKey,
    ticket: Ticket,
    reads: Vec<RecordKey>,
}

struct Held {
    shading: Arc<TileShading>,
    record: RecordKey,
    ticket: Ticket,
    shown: Option<engine_core::Stamp>,
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
    arrived: HashMap<TileKey, RecordKey>,
    held: BTreeMap<TileKey, Held>,
    asked: BTreeMap<TileKey, Asked>,
    failed: HashMap<TileKey, u64>,
    missing: HashSet<RecordKey>,
    sharing: BTreeSet<RecordKey>,
    attached: Vec<TileKey>,
    attaching: bool,
    shaded: u32,
    cooking: bool,
    left: u64,
}

impl Shadings {
    pub(crate) fn shown(&self, tile: &TileFace) -> Option<&Arc<TileShading>> {
        self.held
            .get(&tile.key)
            .filter(|held| held.shown == Some(tile.stamp))
            .map(|held| &held.shading)
    }

    pub(crate) fn attached(&self) -> &[TileKey] {
        &self.attached
    }

    pub(crate) fn counts(&self) -> (u32, u32) {
        (self.shaded, self.asked.len() as u32)
    }

    pub(crate) fn left(&self) -> u64 {
        self.left
    }

    pub(crate) fn holds_share(&self, key: &RecordKey) -> bool {
        self.sharing.contains(key)
    }

    pub(crate) fn forget(self, jobs: &mut Jobs<'_, '_>) -> BTreeSet<RecordKey> {
        for (_, asked) in self.asked {
            jobs.cancel(asked.ticket);
        }
        for (_, held) in self.held {
            jobs.cancel(held.ticket);
        }
        self.sharing
    }

    fn made(&self, key: TileKey, record: RecordKey) -> bool {
        key.level == 1
            || self.missing.contains(&record)
            || (self.cooking && !self.arrived.contains_key(&key))
    }

    fn rebase(&mut self, entry: &TerrainEntry, recipe: [u8; 32], jobs: &mut Jobs<'_, '_>) {
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
        let stale: Vec<TileKey> = self
            .asked
            .iter()
            .filter(|&(&key, asked)| asked.record != record(key))
            .map(|(&key, _)| key)
            .collect();
        for key in stale {
            if let Some(asked) = self.asked.remove(&key) {
                jobs.cancel(asked.ticket);
            }
        }
        let dropped: Vec<TileKey> = self
            .held
            .iter()
            .filter(|&(&key, held)| held.record != record(key))
            .map(|(&key, _)| key)
            .collect();
        for key in dropped {
            if let Some(held) = self.held.remove(&key) {
                jobs.cancel(held.ticket);
            }
        }
        self.arrived.retain(|&key, arrived| *arrived == record(key));
    }

    fn take_arrived(
        &mut self,
        records: &Records,
        jobs: &mut Jobs<'_, '_>,
        (failures, frame): (&mut JobFailures, u64),
    ) -> bool {
        let kind = SHADING.name();
        let keys: Vec<TileKey> = self.asked.keys().copied().collect();
        let mut took = false;
        for key in keys {
            let Some(output) = self
                .asked
                .get(&key)
                .and_then(|asked| records.get(&asked.ticket))
            else {
                continue;
            };
            let Some(asked) = self.asked.remove(&key) else {
                continue;
            };
            took = true;
            let bytes = match output {
                Ok(bytes) => bytes,
                Err(error) if asked.phase == Phase::Fetching && missing(&error) => {
                    self.missing.insert(asked.record);
                    jobs.cancel(asked.ticket);
                    continue;
                }
                Err(error) => {
                    self.failed.insert(key, frame);
                    failures.record(frame, format!("{kind}: {error}"));
                    jobs.cancel(asked.ticket);
                    continue;
                }
            };
            match shading_output(&bytes) {
                Ok(made) if made.key() == key => {
                    self.failed.remove(&key);
                    self.missing.remove(&asked.record);
                    self.arrived.insert(key, asked.record);
                    let held = Held {
                        shading: Arc::new(made),
                        record: asked.record,
                        ticket: asked.ticket,
                        shown: None,
                    };
                    if let Some(before) = self.held.insert(key, held) {
                        jobs.cancel(before.ticket);
                    }
                }
                Ok(_) => {
                    self.failed.insert(key, frame);
                    jobs.cancel(asked.ticket);
                }
                Err(error) => {
                    self.failed.insert(key, frame);
                    failures.record(frame, error.within(&format!("$.{kind}")).to_string());
                    jobs.cancel(asked.ticket);
                }
            }
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
        let cooking = self.cooking;
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
        if cooking {
            for level in 1..levels {
                for key in pyramid.level_tiles(level) {
                    list(&mut wants, key, VIEW_JOBS, false);
                }
            }
        }
        let class = if cooking { VIEW_JOBS } else { IDLE_JOBS };
        let mut at = 0;
        while let Some(&want) = wants.get(at) {
            at += 1;
            if want.key.level < 2 || self.arrived.contains_key(&want.key) {
                continue;
            }
            let record = ShadingJob::of(recipe, &pyramid, &entry.edits, want.key).record();
            if !self.made(want.key, record) {
                continue;
            }
            for read in TileShading::reads(&pyramid, want.key) {
                list(&mut wants, read, class, true);
            }
        }
        wants
    }

    fn keep(&mut self, jobs: &mut Jobs<'_, '_>, wants: &[Want]) {
        let mut wanted: HashMap<TileKey, bool> = HashMap::default();
        for want in wants {
            wanted.insert(want.key, want.held);
        }
        let stale: Vec<TileKey> = self
            .asked
            .keys()
            .filter(|key| !wanted.contains_key(key))
            .copied()
            .collect();
        for key in stale {
            if let Some(asked) = self.asked.remove(&key) {
                jobs.cancel(asked.ticket);
            }
        }
        let dropped: Vec<TileKey> = self
            .held
            .keys()
            .filter(|key| !wanted.get(key).copied().unwrap_or(false))
            .copied()
            .collect();
        for key in dropped {
            if let Some(held) = self.held.remove(&key) {
                jobs.cancel(held.ticket);
            }
        }
        self.arrived.retain(|key, _| wanted.contains_key(key));
        self.failed.retain(|key, _| wanted.contains_key(key));
        self.left = if self.cooking {
            wants
                .iter()
                .filter(|want| {
                    !self.arrived.contains_key(&want.key) && !self.failed.contains_key(&want.key)
                })
                .count() as u64
        } else {
            0
        };
    }

    fn ask(
        &mut self,
        cx: &mut Follow<'_, '_, '_>,
        wants: &[Want],
        (entry, recipe, frame): (&TerrainEntry, [u8; 32], u64),
    ) {
        for want in wants {
            let key = want.key;
            let waits = self
                .failed
                .get(&key)
                .is_some_and(|&at| self.cooking || frame < at + RETRY_AFTER);
            let has = if want.held {
                self.held.contains_key(&key)
            } else {
                self.arrived.contains_key(&key)
            };
            if has || self.asked.contains_key(&key) || waits {
                continue;
            }
            let input = engine_codec::encode(&ShadingJob::of(
                recipe,
                &entry.pyramid,
                &entry.edits,
                key,
            ));
            let record = RecordKey::of(&SHADING, &input);
            let made = self.made(key, record);
            let finer = if key.level > 1 {
                TileShading::reads(&entry.pyramid, key)
            } else {
                Vec::new()
            };
            if made && !finer.iter().all(|at| self.held.contains_key(at)) {
                if self.cooking && finer.iter().any(|at| self.failed.contains_key(at)) {
                    self.failed.insert(key, frame);
                }
                continue;
            }
            let mut reads = Vec::new();
            let (phase, ticket) = if made {
                for at in &finer {
                    let Some(held) = self.held.get(at) else {
                        continue;
                    };
                    reads.push(held.record);
                    if self.sharing.insert(held.record) && cx.shared.insert(held.record) {
                        cx.jobs.share_record(&held.ticket);
                    }
                }
                (Phase::Making, cx.jobs.start(&SHADING, input, want.priority))
            } else {
                (
                    Phase::Fetching,
                    cx.jobs.look(&SHADING, &input, want.priority),
                )
            };
            self.asked.insert(
                key,
                Asked {
                    phase,
                    record,
                    ticket,
                    reads,
                },
            );
        }
    }

    fn release_shares(&mut self, released: &mut Vec<RecordKey>) {
        let reading: HashSet<RecordKey> = self
            .asked
            .values()
            .filter(|asked| asked.phase == Phase::Making)
            .flat_map(|asked| asked.reads.iter().copied())
            .collect();
        let shares: Vec<RecordKey> = self.sharing.iter().copied().collect();
        for key in shares {
            if !reading.contains(&key) {
                self.sharing.remove(&key);
                released.push(key);
            }
        }
    }

    fn attach(&mut self, entry: &TerrainEntry, work: &mut Work) {
        let Self {
            held,
            attached,
            attaching,
            shaded,
            ..
        } = self;
        *attaching = false;
        *shaded = 0;
        let mut gave = false;
        for (&key, made) in held.iter_mut() {
            let Some(tile) = listed_tile(&entry.tiles, key) else {
                continue;
            };
            if made.shown == Some(tile.stamp) {
                *shaded += 1;
                continue;
            }
            if !tile.current {
                continue;
            }
            if gave && work.spent() {
                *attaching = true;
                continue;
            }
            gave = true;
            work.upload = work.upload.saturating_sub(2 * TILE_ARRAY_BYTES);
            made.shown = Some(tile.stamp);
            attached.push(key);
            *shaded += 1;
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

struct Work {
    clock: Option<fn() -> f64>,
    end: f64,
    upload: u64,
}

impl Work {
    fn spent(&self) -> bool {
        self.clock.is_some_and(|clock| clock() >= self.end) || self.upload == 0
    }
}

struct Follow<'a, 'w, 's> {
    jobs: &'a mut Jobs<'w, 's>,
    records: &'a Records,
    failures: &'a mut JobFailures,
    shared: &'a mut BTreeSet<RecordKey>,
    released: &'a mut Vec<RecordKey>,
    work: &'a mut Work,
}

pub(crate) fn shade(
    tiles: Res<TerrainTiles>,
    records: Res<Records>,
    view: Res<View>,
    time: Res<FrameTime>,
    (clock, ground, cook): (Res<WorkClock>, Res<GroundWork>, Res<CookMode>),
    mut jobs: Jobs,
    mut bakes: ResMut<Bakes>,
) {
    let eye = view.detail_eye();
    let frame = time.frame;
    let moved = tiles.is_changed();
    let mut work = Work {
        clock: clock.reading,
        end: clock.now().map_or(0.0, |now| now + ground.millis),
        upload: ground.upload,
    };
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
            work: &mut work,
        };
        let local = Vec2::new(eye.x - entry.origin.x, eye.z - entry.origin.z);
        follow(held, entry, &mut cx, (local, frame, moved));
        if cook.cooking && !entry.surroundings && entry.recipe.is_some() {
            held.flow.cooking = true;
        }
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
        let cooking = held.flow.cooking;
        let left = std::mem::replace(
            &mut held.flow,
            Shadings {
                cooking,
                ..Shadings::default()
            },
        );
        cx.released.extend(left.forget(cx.jobs));
        return;
    };
    let flow = &mut held.flow;
    flow.attached.clear();
    let moved = moved || flow.basis.is_none();
    if moved {
        flow.rebase(entry, recipe, cx.jobs);
    }
    let arrived = flow.take_arrived(cx.records, cx.jobs, (&mut *cx.failures, frame));
    let dirty = moved || arrived || flow.cooking || !flow.failed.is_empty();
    if dirty {
        let wants = flow.wanted(entry, recipe, eye);
        flow.keep(cx.jobs, &wants);
        flow.ask(cx, &wants, (entry, recipe, frame));
        flow.release_shares(cx.released);
    }
    if dirty || flow.attaching {
        flow.attach(entry, cx.work);
    }
}
