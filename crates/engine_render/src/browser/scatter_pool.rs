//! Placed objects' slots (decision 057): every resident tile's objects hold
//! a range of slots of their own, written when the tile arrives and marked
//! dead when it leaves, so streaming tiles uploads only theirs. Freed ranges
//! are reused first fit; the pool starts over when its assets change or
//! freed slots outnumber live ones.

use std::collections::HashMap;
use std::ops::Range;

use engine_core::TileKey;
use engine_core::glam::Vec3;

use crate::gpu_data::{ColorData, GroundData, PoseData};
use crate::scatter::{DEAD, NEAR_LEVEL_SHIFT, Record};

/// A tile's objects as the pool takes them: poses, colors, records and
/// grounds (decision 096).
pub(super) type TileObjects = (Vec<PoseData>, Vec<ColorData>, Vec<Record>, Vec<GroundData>);

#[derive(Default)]
pub(super) struct ScatterPool {
    /// Every slot's pose, color, record and ground, dead ones' records
    /// marked.
    pub(super) poses: Vec<PoseData>,
    pub(super) colors: Vec<ColorData>,
    pub(super) records: Vec<Record>,
    pub(super) grounds: Vec<GroundData>,
    /// Live objects of each draw group, and of those standing far away
    /// (decision 061).
    pub(super) counts: Vec<u32>,
    pub(super) far: u32,
    // Each resident tile's identity (its pointer) and slots, by terrain and
    // key.
    tiles: HashMap<(u64, TileKey), (usize, Range<u32>)>,
    // Free slot ranges, in order, none adjacent.
    free: Vec<Range<u32>>,
    // The tiles revision of each terrain last synchronized.
    revisions: HashMap<u64, u64>,
    // Slots written since the last upload, whether counts changed, and the
    // world bounds of each tile's objects that came or went since.
    dirty: Option<Range<u32>>,
    counts_changed: bool,
    moved: Vec<(Vec3, Vec3)>,
}

impl ScatterPool {
    /// Empties the pool for `groups` draw groups.
    pub(super) fn clear(&mut self, groups: usize) {
        *self = Self {
            counts: vec![0; groups],
            counts_changed: true,
            ..Self::default()
        };
    }

    /// Whether terrain `id`'s tiles changed since revision `revision` was
    /// last synchronized, recording it.
    pub(super) fn tiles_changed(&mut self, id: u64, revision: u64) -> bool {
        self.revisions.insert(id, revision) != Some(revision)
    }

    /// Frees the tiles of terrain `id` that `keep` refuses, given each
    /// one's key and identity.
    pub(super) fn retain(&mut self, id: u64, keep: impl Fn(TileKey, usize) -> bool) {
        let mut freed = Vec::new();
        self.tiles.retain(|&(terrain, key), (identity, slots)| {
            if terrain != id || keep(key, *identity) {
                return true;
            }
            freed.push(slots.clone());
            false
        });
        for slots in freed {
            let bounds = bounds(&self.records[slots.start as usize..slots.end as usize]);
            self.moved.extend(bounds);
            self.free_slots(slots);
        }
    }

    /// Whether terrain `id`'s tile `key` holds slots.
    pub(super) fn contains(&self, id: u64, key: TileKey) -> bool {
        self.tiles.contains_key(&(id, key))
    }

    /// Gives terrain `id`'s tile `key`, of identity `identity`, slots for
    /// `objects`.
    pub(super) fn insert(
        &mut self,
        (id, key): (u64, TileKey),
        identity: usize,
        objects: TileObjects,
    ) {
        let (poses, colors, records, grounds) = objects;
        let count = records.len() as u32;
        if count == 0 {
            return;
        }
        let slots = self.allocate(count);
        let range = slots.start as usize..slots.end as usize;
        self.poses[range.clone()].copy_from_slice(&poses);
        self.colors[range.clone()].copy_from_slice(&colors);
        self.grounds[range.clone()].copy_from_slice(&grounds);
        self.records[range].copy_from_slice(&records);
        for record in &records {
            self.counts[record.group as usize] += 1;
            self.far += u32::from(stands_far(record));
        }
        self.moved.extend(bounds(&records));
        self.counts_changed = true;
        self.mark(slots.clone());
        self.tiles.insert((id, key), (identity, slots));
    }

    /// Slots written since the last call, and whether group counts changed.
    pub(super) fn take_changes(&mut self) -> (Option<Range<u32>>, bool) {
        (self.dirty.take(), std::mem::take(&mut self.counts_changed))
    }

    /// The world bounds of each tile's objects that came or went since the
    /// last call.
    pub(super) fn take_moved(&mut self) -> Vec<(Vec3, Vec3)> {
        std::mem::take(&mut self.moved)
    }

    /// Whether freed slots outnumber live ones enough to start over.
    pub(super) fn fragmented(&self) -> bool {
        let freed: u32 = self.free.iter().map(|range| range.end - range.start).sum();
        freed > 4096 && freed as usize * 2 > self.records.len()
    }

    /// `count` slots: the first free range holding them, or new ones.
    fn allocate(&mut self, count: u32) -> Range<u32> {
        if let Some(index) = self
            .free
            .iter()
            .position(|range| range.end - range.start >= count)
        {
            let start = self.free[index].start;
            self.free[index].start += count;
            if self.free[index].is_empty() {
                self.free.remove(index);
            }
            return start..start + count;
        }
        let start = self.records.len() as u32;
        let end = start as usize + count as usize;
        self.poses.resize(end, [[0.0; 4]; 3]);
        self.colors.resize(
            end,
            ColorData {
                color: [0; 2],
                surface: 0,
                flags: 0,
            },
        );
        self.records.resize(end, dead());
        self.grounds.resize(end, [0; 4]);
        start..start + count
    }

    /// Marks `slots` dead and free.
    fn free_slots(&mut self, slots: Range<u32>) {
        for record in &mut self.records[slots.start as usize..slots.end as usize] {
            self.counts[record.group as usize] -= 1;
            self.far -= u32::from(stands_far(record));
            *record = dead();
        }
        self.counts_changed = true;
        self.mark(slots.clone());
        let at = self.free.partition_point(|range| range.start < slots.start);
        self.free.insert(at, slots);
        // Merge with the neighbors it touches.
        if at + 1 < self.free.len() && self.free[at].end == self.free[at + 1].start {
            self.free[at].end = self.free[at + 1].end;
            self.free.remove(at + 1);
        }
        if at > 0 && self.free[at - 1].end == self.free[at].start {
            self.free[at - 1].end = self.free[at].end;
            self.free.remove(at);
        }
    }

    fn mark(&mut self, slots: Range<u32>) {
        self.dirty = Some(match self.dirty.take() {
            Some(dirty) => dirty.start.min(slots.start)..dirty.end.max(slots.end),
            None => slots,
        });
    }
}

/// The world bounds of `records`' live objects, a radius taller: the
/// terrain sets their bases on finer heights than placed them.
fn bounds(records: &[Record]) -> Option<(Vec3, Vec3)> {
    records
        .iter()
        .filter(|record| record.group != DEAD)
        .map(|record| {
            let center = Vec3::from(record.center);
            let reach = Vec3::new(record.radius, 2.0 * record.radius, record.radius);
            (center - reach, center + reach)
        })
        .reduce(|(low, high), (min, max)| (low.min(min), high.max(max)))
}

/// Whether `record`'s object stands far away (decision 061).
fn stands_far(record: &Record) -> bool {
    record.group != DEAD && (record.terrain >> NEAR_LEVEL_SHIFT) & 0xff != 0
}

/// The record of a dead slot, which culling skips.
fn dead() -> Record {
    Record {
        center: [0.0; 3],
        radius: 0.0,
        group: DEAD,
        scale: 0.0,
        terrain: u32::MAX,
        sink: 0.0,
    }
}
