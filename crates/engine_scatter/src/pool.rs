use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::prelude::Entity;
use engine_core::Stamp;
use engine_math::collections::HashMap;
use engine_places::ground::{Placed, TileKey};
use glam::Vec3;

use crate::groups::Groups;
use crate::layouts::{
    DEAD, GroundData, NEAR_LEVEL_MASK, NEAR_LEVEL_SHIFT, PlacedRecord, PoseData, STATE_ADOPT,
    STATE_FRESH, SlotColor,
};

pub(crate) type TileObjects = (
    Vec<PoseData>,
    Vec<SlotColor>,
    Vec<PlacedRecord>,
    Vec<GroundData>,
);

pub(crate) type PoolKey = (Entity, TileKey);

pub(crate) struct Pooled {
    pub(crate) placed: Arc<[Placed]>,
    pub(crate) stamp: Stamp,
    pub(crate) origin: Vec3,
    pub(crate) slots: Range<u32>,
    bounds: Option<(Vec3, Vec3)>,
    groups: Vec<(u32, u32)>,
    far: u32,
}

impl Pooled {
    pub(crate) fn identity(&self) -> u64 {
        self.placed.as_ptr() as usize as u64
    }
}

pub(crate) enum SlotWrite {
    Objects(u32, TileObjects),
    Dead(Range<u32>),
}

#[derive(Default)]
pub(crate) struct ScatterPool {
    pub(crate) counts: Vec<u32>,
    pub(crate) far: u32,
    pub(crate) end: u32,
    pub(crate) writes: Vec<SlotWrite>,
    tiles: HashMap<PoolKey, Pooled>,
    free: Vec<Range<u32>>,
    moved: Vec<(Vec3, Vec3)>,
    refilling: bool,
}

impl ScatterPool {
    pub(crate) fn clear(&mut self, groups: usize) {
        *self = Self {
            counts: vec![0; groups],
            refilling: true,
            ..Self::default()
        };
    }

    pub(crate) fn settle(&mut self) {
        if !self.writes.is_empty() {
            self.writes.clear();
            self.refilling = false;
        }
    }

    pub(crate) fn tiles(&self) -> Vec<(PoolKey, &Pooled)> {
        let mut tiles: Vec<_> = self
            .tiles
            .iter()
            .map(|(&key, pooled)| (key, pooled))
            .collect();
        tiles.sort_unstable_by_key(|(key, _)| *key);
        tiles
    }

    pub(crate) fn retain(&mut self, id: Entity, keep: impl Fn(TileKey, &Pooled) -> bool) {
        let mut freed = Vec::new();
        self.tiles.retain(|&(terrain, key), pooled| {
            if terrain != id || keep(key, pooled) {
                return true;
            }
            freed.push((
                pooled.slots.clone(),
                pooled.bounds,
                std::mem::take(&mut pooled.groups),
                pooled.far,
            ));
            false
        });
        for (slots, bounds, groups, far) in freed {
            self.moved.extend(bounds);
            for (group, count) in groups {
                self.counts[group as usize] -= count;
            }
            self.far -= far;
            self.vacate(slots);
        }
    }

    pub(crate) fn contains(&self, id: Entity, key: TileKey) -> bool {
        self.tiles.contains_key(&(id, key))
    }

    pub(crate) fn insert(
        &mut self,
        (id, key): PoolKey,
        (placed, stamp, origin): (Arc<[Placed]>, Stamp, Vec3),
        mut objects: TileObjects,
    ) {
        let count = objects.2.len() as u32;
        if count == 0 {
            return;
        }
        let slots = self.allocate(count);
        let state = if self.refilling {
            STATE_ADOPT
        } else {
            STATE_FRESH
        };
        let mut groups: Vec<(u32, u32)> = Vec::new();
        let mut far = 0;
        for record in &mut objects.2 {
            record.state = state;
            far += u32::from(stands_far(record));
            match groups.iter_mut().find(|(group, _)| *group == record.group) {
                Some((_, held)) => *held += 1,
                None => groups.push((record.group, 1)),
            }
        }
        for &(group, held) in &groups {
            self.counts[group as usize] += held;
        }
        self.far += far;
        let reach = bounds(&objects.2);
        self.moved.extend(reach);
        self.writes.push(SlotWrite::Objects(slots.start, objects));
        self.tiles.insert(
            (id, key),
            Pooled {
                placed,
                stamp,
                origin,
                slots,
                bounds: reach,
                groups,
                far,
            },
        );
    }

    pub(crate) fn whole<'a>(&'a self, groups: &'a Groups) -> impl Iterator<Item = SlotWrite> + 'a {
        self.tiles()
            .into_iter()
            .map(move |((terrain, _), pooled)| {
                SlotWrite::Objects(
                    pooled.slots.start,
                    adopted(groups.tile_objects(terrain, pooled.origin, &pooled.placed)),
                )
            })
            .chain(self.free.iter().cloned().map(SlotWrite::Dead))
    }

    pub(crate) fn take_moved(&mut self) -> Vec<(Vec3, Vec3)> {
        std::mem::take(&mut self.moved)
    }

    pub(crate) fn compact(&mut self, groups: &Groups) {
        if let Some(last) = self.free.last().cloned()
            && last.end == self.end
        {
            self.free.pop();
            self.end = last.start;
        }
        let freed: u32 = self.free.iter().map(|range| range.end - range.start).sum();
        if freed <= 4096 || u64::from(freed) * 2 <= u64::from(self.end) {
            return;
        }
        let free = &self.free;
        let fits = |slots: &Range<u32>| {
            free.iter().position(|range| {
                range.end <= slots.start && range.end - range.start >= slots.end - slots.start
            })
        };
        let Some((&key, at)) = self
            .tiles
            .iter()
            .filter_map(|(key, pooled)| Some(((key, pooled.slots.start), fits(&pooled.slots)?)))
            .max_by_key(|((_, start), _)| *start)
            .map(|((key, _), at)| (key, at))
        else {
            return;
        };
        let Some(pooled) = self.tiles.get_mut(&key) else {
            return;
        };
        let from = pooled.slots.clone();
        let count = from.end - from.start;
        let start = self.free[at].start;
        pooled.slots = start..start + count;
        self.free[at].start += count;
        if self.free[at].is_empty() {
            self.free.remove(at);
        }
        self.writes.push(SlotWrite::Objects(
            start,
            adopted(groups.tile_objects(key.0, pooled.origin, &pooled.placed)),
        ));
        self.vacate(from);
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
        let start = self.end;
        self.end += count;
        start..self.end
    }

    fn vacate(&mut self, slots: Range<u32>) {
        self.writes.push(SlotWrite::Dead(slots.clone()));
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
}

fn adopted(mut objects: TileObjects) -> TileObjects {
    for record in &mut objects.2 {
        record.state = STATE_ADOPT;
    }
    objects
}

fn bounds(records: &[PlacedRecord]) -> Option<(Vec3, Vec3)> {
    records
        .iter()
        .map(|record| {
            let center = Vec3::from(record.center);
            let reach = Vec3::new(record.radius, 2.0 * record.radius, record.radius);
            (center - reach, center + reach)
        })
        .reduce(|(low, high), (min, max)| (low.min(min), high.max(max)))
}

fn stands_far(record: &PlacedRecord) -> bool {
    (record.terrain >> NEAR_LEVEL_SHIFT) & NEAR_LEVEL_MASK != 0
}

pub(crate) fn dead() -> PlacedRecord {
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
