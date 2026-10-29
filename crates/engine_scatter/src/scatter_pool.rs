use engine_core::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use engine_core::glam::Vec3;
use engine_faces::placed::PlacedRecord;
use engine_render::gpu_data::{ColorData, GroundData, PoseData};
use engine_terrain::{Tile, TileKey};

use crate::scatter::{DEAD, NEAR_LEVEL_MASK, NEAR_LEVEL_SHIFT, STATE_ADOPT, STATE_FRESH};

pub(super) type TileObjects = (Vec<PoseData>, Vec<ColorData>, Vec<PlacedRecord>, Vec<GroundData>);

#[derive(Default)]
pub(super) struct ScatterPool {
    pub(super) poses: Vec<PoseData>,
    pub(super) colors: Vec<ColorData>,
    pub(super) records: Vec<PlacedRecord>,
    pub(super) grounds: Vec<GroundData>,
    pub(super) counts: Vec<u32>,
    pub(super) far: u32,
    tiles: HashMap<(u64, TileKey), (Arc<Tile>, Range<u32>)>,
    free: Vec<Range<u32>>,
    revisions: HashMap<u64, u64>,
    dirty: Option<Range<u32>>,
    counts_changed: bool,
    moved: Vec<(Vec3, Vec3)>,
    refilling: bool,
    revision: u64,
}

impl ScatterPool {
    pub(super) fn clear(&mut self, groups: usize) {
        *self = Self {
            counts: vec![0; groups],
            counts_changed: true,
            refilling: true,
            revision: self.revision + 1,
            ..Self::default()
        };
    }

    pub(super) fn tiles(&self) -> Vec<((u64, TileKey), Range<u32>, usize)> {
        let mut tiles: Vec<_> = self
            .tiles
            .iter()
            .map(|(&key, (tile, slots))| (key, slots.clone(), tile.placed().as_ptr() as usize))
            .collect();
        tiles.sort_unstable_by_key(|(key, _, _)| *key);
        tiles
    }

    pub(super) fn revision(&self) -> u64 {
        self.revision
    }

    pub(super) fn tiles_changed(&mut self, id: u64, revision: u64) -> bool {
        self.revisions.insert(id, revision) != Some(revision)
    }

    pub(super) fn retain(&mut self, id: u64, keep: impl Fn(TileKey, &Arc<Tile>) -> bool) {
        let mut freed = Vec::new();
        self.tiles.retain(|&(terrain, key), (tile, slots)| {
            if terrain != id || keep(key, tile) {
                return true;
            }
            freed.push(slots.clone());
            false
        });
        for slots in freed {
            let bounds = bounds(&self.records[slots.start as usize..slots.end as usize]);
            self.moved.extend(bounds);
            self.free_slots(slots);
            self.revision += 1;
        }
    }

    pub(super) fn contains(&self, id: u64, key: TileKey) -> bool {
        self.tiles.contains_key(&(id, key))
    }

    pub(super) fn insert(
        &mut self,
        (id, key): (u64, TileKey),
        tile: Arc<Tile>,
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
        self.records[range.clone()].copy_from_slice(&records);
        let state = if self.refilling {
            STATE_ADOPT
        } else {
            STATE_FRESH
        };
        for record in &mut self.records[range] {
            record.state = state;
        }
        for record in &records {
            self.counts[record.group as usize] += 1;
            self.far += u32::from(stands_far(record));
        }
        self.moved.extend(bounds(&records));
        self.counts_changed = true;
        self.mark(slots.clone());
        self.tiles.insert((id, key), (tile, slots));
        self.revision += 1;
    }

    pub(super) fn take_changes(&mut self) -> (Option<Range<u32>>, bool) {
        (self.dirty.take(), std::mem::take(&mut self.counts_changed))
    }

    pub(super) fn uploaded(&mut self, slots: Range<u32>) {
        for record in &mut self.records[slots.start as usize..slots.end as usize] {
            record.state = STATE_ADOPT;
        }
        self.refilling = false;
    }

    pub(super) fn take_moved(&mut self) -> Vec<(Vec3, Vec3)> {
        std::mem::take(&mut self.moved)
    }

    pub(super) fn fragmented(&self) -> bool {
        let freed: u32 = self.free.iter().map(|range| range.end - range.start).sum();
        freed > 4096 && freed as usize * 2 > self.records.len()
    }

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

fn bounds(records: &[PlacedRecord]) -> Option<(Vec3, Vec3)> {
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

fn stands_far(record: &PlacedRecord) -> bool {
    record.group != DEAD && (record.terrain >> NEAR_LEVEL_SHIFT) & NEAR_LEVEL_MASK != 0
}

fn dead() -> PlacedRecord {
    PlacedRecord {
        center: [0.0; 3],
        radius: 0.0,
        group: DEAD,
        scale: 0.0,
        terrain: u32::MAX,
        sink: 0.0,
        state: STATE_ADOPT,
        pad: [0; 3],
    }
}
