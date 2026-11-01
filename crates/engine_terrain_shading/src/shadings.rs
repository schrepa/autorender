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

use crate::jobs::{SHADING, ShadingJob, ShadingReads, shading_output};
use crate::shading::TileShading;
use crate::state::{Bakes, RETRY_AFTER, TerrainBakes, tile_stamps};

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
}

struct Held {
    shading: Arc<TileShading>,
    record: RecordKey,
    ticket: Ticket,
}

#[derive(Clone, Copy)]
struct Want {
    key: TileKey,
    priority: u32,
    held: bool,
}

#[derive(Default)]
pub(crate) struct Shadings {
    found: HashMap<TileKey, (ShadingJob, RecordKey)>,
    found_for: Option<([u8; 32], Arc<[Stamp]>)>,
    arrived: HashMap<TileKey, RecordKey>,
    held: BTreeMap<TileKey, Held>,
    asked: BTreeMap<TileKey, Asked>,
    failed: HashMap<TileKey, u64>,
    missing: HashSet<RecordKey>,
    sharing: BTreeSet<RecordKey>,
    shaded: u32,
    cooking: bool,
    left: u64,
}

impl Shadings {
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

    fn has(&self, key: TileKey, record: RecordKey, held: bool) -> bool {
        if held {
            self.held
                .get(&key)
                .is_some_and(|held| held.record == record)
        } else {
            self.arrived.get(&key) == Some(&record)
        }
    }

    fn made(&self, key: TileKey, record: RecordKey) -> bool {
        key.level == 1
            || self.missing.contains(&record)
            || (self.cooking && self.arrived.get(&key) != Some(&record))
    }

    fn basis(&mut self, recipe: [u8; 32], edits: &Arc<[Stamp]>) {
        let same = self.found_for.as_ref().is_some_and(|(held, stamps)| {
            *held == recipe && (Arc::ptr_eq(stamps, edits) || stamps[..] == edits[..])
        });
        if !same {
            self.found = HashMap::default();
        }
        self.found_for = Some((recipe, Arc::clone(edits)));
    }

    fn record(&mut self, entry: &TerrainEntry, recipe: [u8; 32], key: TileKey) -> RecordKey {
        if let Some((_, record)) = self.found.get(&key) {
            return *record;
        }
        let read = TileShading::reads(&entry.pyramid, key);
        let reads = if key.level == 1 {
            ShadingReads::Tiles(
                read.into_iter()
                    .map(|at| (at, tile_stamps(entry, at)))
                    .collect(),
            )
        } else {
            ShadingReads::Shadings(
                read.into_iter()
                    .map(|at| (at, self.record(entry, recipe, at).hash()))
                    .collect(),
            )
        };
        let job = ShadingJob { recipe, key, reads };
        let record = RecordKey::of(&SHADING, &engine_codec::encode(&job));
        self.found.insert(key, (job, record));
        record
    }

    fn take_arrived(
        &mut self,
        records: &Records,
        jobs: &mut Jobs<'_, '_>,
        (failures, frame): (&mut JobFailures, u64),
    ) {
        let kind = SHADING.name();
        let keys: Vec<TileKey> = self.asked.keys().copied().collect();
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
    }

    fn wanted(&mut self, entry: &TerrainEntry, recipe: [u8; 32], eye: Vec2) -> Vec<Want> {
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
            if want.key.level < 2 {
                continue;
            }
            let record = self.record(entry, recipe, want.key);
            if self.has(want.key, record, false) || !self.made(want.key, record) {
                continue;
            }
            let reads: Vec<TileKey> = match self.found.get(&want.key).map(|(job, _)| &job.reads) {
                Some(ShadingReads::Shadings(reads)) => {
                    reads.iter().map(|&(read, _)| read).collect()
                }
                Some(ShadingReads::Tiles(_)) | None => Vec::new(),
            };
            for read in reads {
                list(&mut wants, read, class, true);
            }
        }
        for want in &wants {
            self.record(entry, recipe, want.key);
        }
        wants
    }

    fn keep(&mut self, jobs: &mut Jobs<'_, '_>, wants: &[Want]) {
        let mut wanted: HashMap<TileKey, bool> = HashMap::default();
        for want in wants {
            wanted.insert(want.key, want.held);
        }
        let found = &self.found;
        let current = |key: &TileKey, record: RecordKey| {
            found.get(key).is_some_and(|(_, now)| *now == record)
        };
        let stale: Vec<TileKey> = self
            .asked
            .iter()
            .filter(|&(key, asked)| !wanted.contains_key(key) || !current(key, asked.record))
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
            .filter(|&(key, held)| {
                let holding = wanted.get(key).copied().unwrap_or(false);
                !(holding && current(key, held.record))
            })
            .map(|(&key, _)| key)
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
                    let record = found.get(&want.key).map(|(_, record)| *record);
                    self.arrived.get(&want.key).copied() != record
                        && !self.failed.contains_key(&want.key)
                })
                .count() as u64
        } else {
            0
        };
    }

    fn ask(&mut self, cx: &mut Follow<'_, '_, '_>, wants: &[Want], frame: u64) {
        for want in wants {
            let key = want.key;
            let Some((job, record)) = self
                .found
                .get(&key)
                .map(|(job, record)| (job.clone(), *record))
            else {
                continue;
            };
            let waits = self
                .failed
                .get(&key)
                .is_some_and(|&at| self.cooking || frame < at + RETRY_AFTER);
            if self.has(key, record, want.held) || self.asked.contains_key(&key) || waits {
                continue;
            }
            let made = self.made(key, record);
            let reads: Vec<(TileKey, RecordKey)> = match &job.reads {
                ShadingReads::Shadings(reads) => reads
                    .iter()
                    .map(|&(at, hash)| (at, RecordKey::from_hash(hash)))
                    .collect(),
                ShadingReads::Tiles(_) => Vec::new(),
            };
            if made && !reads.iter().all(|&(at, read)| self.has(at, read, true)) {
                if self.cooking && reads.iter().any(|(at, _)| self.failed.contains_key(at)) {
                    self.failed.insert(key, frame);
                }
                continue;
            }
            let input = engine_codec::encode(&job);
            let (phase, ticket) = if made {
                for (at, read) in &reads {
                    let Some(held) = self.held.get(at) else {
                        continue;
                    };
                    if self.sharing.insert(*read) && cx.shared.insert(*read) {
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
                },
            );
        }
    }

    fn release_shares(&mut self, released: &mut Vec<RecordKey>) {
        let mut reading: HashSet<RecordKey> = HashSet::default();
        for (key, asked) in &self.asked {
            if asked.phase != Phase::Making {
                continue;
            }
            if let Some((job, _)) = self.found.get(key)
                && let ShadingReads::Shadings(reads) = &job.reads
            {
                reading.extend(reads.iter().map(|&(_, read)| RecordKey::from_hash(read)));
            }
        }
        let shares: Vec<RecordKey> = self.sharing.iter().copied().collect();
        for key in shares {
            if !reading.contains(&key) {
                self.sharing.remove(&key);
                released.push(key);
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
        held.prune(entry);
        let mut cx = Follow {
            jobs: &mut jobs,
            records: &records,
            failures: &mut bakes.failures,
            shared: &mut shared,
            released: &mut released,
            work: &mut work,
        };
        let local = Vec2::new(eye.x - entry.origin.x, eye.z - entry.origin.z);
        follow(held, entry, &mut cx, (local, frame));
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
    (eye, frame): (Vec2, u64),
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
    flow.basis(recipe, &entry.edits);
    flow.take_arrived(cx.records, cx.jobs, (&mut *cx.failures, frame));
    let wants = flow.wanted(entry, recipe, eye);
    flow.keep(cx.jobs, &wants);
    flow.ask(cx, &wants, frame);
    flow.release_shares(cx.released);
    attach(held, entry, cx.work);
}

fn attach(held: &mut TerrainBakes, entry: &TerrainEntry, work: &mut Work) {
    let mut resident: Vec<&TileFace> = entry
        .tiles
        .iter()
        .filter(|tile| shaded_elsewhere(tile.key.level, false))
        .collect();
    resident.sort_unstable_by_key(|tile| tile.key);
    let mut gave = false;
    for tile in resident {
        let flow = &held.flow;
        let Some(made) = flow.held.get(&tile.key) else {
            continue;
        };
        let current = flow
            .found
            .get(&tile.key)
            .is_some_and(|(_, now)| *now == made.record);
        let holds = held
            .shading(&tile.key)
            .is_some_and(|attached| Arc::ptr_eq(attached, &made.shading));
        if holds || !current || !tile.current {
            continue;
        }
        if gave && work.spent() {
            break;
        }
        gave = true;
        work.upload = work.upload.saturating_sub(2 * TILE_ARRAY_BYTES);
        let shading = Arc::clone(&made.shading);
        held.attach(tile, shading);
    }
    held.flow.shaded = held.attached();
}
