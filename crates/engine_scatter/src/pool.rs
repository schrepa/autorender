use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::prelude::Entity;
use engine_core::Stamp;
use engine_heightfield::tile::Placed;
use engine_math::collections::HashMap;
use engine_tiles::TileKey;
use glam::{Vec2, Vec3};

use crate::groups::Groups;
use engine_core::GpuBits;

use crate::layouts::{
    DEAD, NO_LAYER, PlacedRecord, PlacedValues, PoseData, ScatterTerrain, SlotColor,
    TERRAIN_INDEX,
};

pub(crate) type TileObjects = (
    Vec<PoseData>,
    Vec<SlotColor>,
    Vec<PlacedRecord>,
    Vec<PlacedValues>,
);

pub(crate) type PoolKey = (Entity, TileKey);

pub(crate) type Reach = (Vec3, Vec3, f32);

pub(crate) struct Pooled {
    pub(crate) placed: Arc<[Placed]>,
    pub(crate) stamp: Stamp,
    pub(crate) origin: Vec3,
    pub(crate) slots: Range<u32>,
    bounds: Option<Reach>,
    stands: (Vec2, Vec2),
    groups: Vec<(u32, u32)>,
    far: u32,
}

impl Pooled {
    pub(crate) fn identity(&self) -> u64 {
        self.placed.as_ptr() as usize as u64
    }

    pub(crate) fn groups(&self) -> impl Iterator<Item = usize> + '_ {
        self.groups.iter().map(|&(group, _)| group as usize)
    }
}

pub(crate) enum SlotWrite {
    Objects(u32, TileObjects),
    Dead(Range<u32>),
}

#[derive(Clone, Copy)]
pub(crate) struct Spread {
    pub(crate) low: f32,
    pub(crate) high: f32,
    pub(crate) radius: f32,
}

impl Spread {
    const NONE: Self = Self {
        low: f32::INFINITY,
        high: 0.0,
        radius: 0.0,
    };

    fn take(&mut self, record: &PlacedRecord) {
        self.low = engine_math::min(self.low, record.scale);
        self.high = engine_math::max(self.high, record.scale);
        self.radius = engine_math::max(self.radius, record.radius);
    }
}

#[derive(Default)]
pub(crate) struct ScatterPool {
    pub(crate) counts: Vec<u32>,
    pub(crate) spreads: Vec<Spread>,
    pub(crate) far: u32,
    pub(crate) end: u32,
    pub(crate) writes: Vec<SlotWrite>,
    tiles: HashMap<PoolKey, Pooled>,
    free: Vec<Range<u32>>,
    moved: Vec<(Entity, Reach)>,
    revision: u64,
}

impl ScatterPool {
    pub(crate) fn clear(&mut self, groups: usize) {
        *self = Self {
            counts: vec![0; groups],
            spreads: vec![Spread::NONE; groups],
            revision: self.revision + 1,
            ..Self::default()
        };
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    pub(crate) fn settle(&mut self) {
        self.writes.clear();
    }

    pub(crate) fn pooled(&self) -> impl ExactSizeIterator<Item = (&PoolKey, &Pooled)> + '_ {
        self.tiles.iter()
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
        self.revision += u64::from(!freed.is_empty());
        for (slots, bounds, groups, far) in freed {
            self.moved.extend(bounds.map(|bounds| (id, bounds)));
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
        objects: TileObjects,
    ) {
        let count = objects.2.len() as u32;
        if count == 0 {
            return;
        }
        let slots = self.allocate(count);
        self.revision += 1;
        let mut groups: Vec<(u32, u32)> = Vec::new();
        let mut far = 0;
        for record in &objects.2 {
            far += u32::from(stands_far(record));
            if let Some(spread) = self.spreads.get_mut(record.group as usize) {
                spread.take(record);
            }
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
        let stands = stands(&objects.0);
        self.moved.extend(reach.map(|reach| (id, reach)));
        self.writes.push(SlotWrite::Objects(slots.start, objects));
        self.tiles.insert(
            (id, key),
            Pooled {
                placed,
                stamp,
                origin,
                slots,
                bounds: reach,
                stands,
                groups,
                far,
            },
        );
    }

    pub(crate) fn standing_in<'a>(
        &'a self,
        rects: &'a [(Entity, Vec2, Vec2)],
    ) -> impl Iterator<Item = (Range<u32>, bool, Option<(Entity, Reach)>)> + 'a {
        self.tiles
            .iter()
            .filter(|((terrain, _), pooled)| {
                let (low, high) = pooled.stands;
                rects.iter().any(|&(held, from, to)| {
                    held == *terrain
                        && low.x <= to.x
                        && high.x >= from.x
                        && low.y <= to.y
                        && high.y >= from.y
                })
            })
            .map(|(&(terrain, _), pooled)| {
                let (low, high) = pooled.stands;
                let finite = low.is_finite() && high.is_finite();
                let bounds = pooled.bounds.map(|bounds| (terrain, bounds));
                (pooled.slots.clone(), finite, bounds)
            })
    }

    pub(crate) fn whole<'a>(&'a self, groups: &'a Groups) -> impl Iterator<Item = SlotWrite> + 'a {
        self.tiles()
            .into_iter()
            .map(move |((terrain, _), pooled)| {
                SlotWrite::Objects(
                    pooled.slots.start,
                    groups.tile_objects(terrain, pooled.origin, &pooled.placed),
                )
            })
            .chain(self.free.iter().cloned().map(SlotWrite::Dead))
    }

    pub(crate) fn nearest(&self, eye: Vec3) -> Vec<f32> {
        let mut nearest = vec![f32::INFINITY; self.spreads.len()];
        for pooled in self.tiles.values() {
            let Some((low, high, _)) = pooled.bounds else {
                continue;
            };
            let outside = (low - eye).max(eye - high).max(Vec3::ZERO);
            let away = (outside.x * outside.x + outside.z * outside.z).sqrt();
            let reach = eye.abs().max(low.abs()).max(high.abs()).max_element() + away;
            let away = engine_math::max(away - reach * 1.0e-5, 0.0);
            for &(group, _) in &pooled.groups {
                if let Some(near) = nearest.get_mut(group as usize) {
                    *near = engine_math::min(*near, away);
                }
            }
        }
        nearest
    }

    pub(crate) fn take_moved(&mut self) -> Vec<(Entity, Reach)> {
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
        self.revision += 1;
        self.free[at].start += count;
        if self.free[at].is_empty() {
            self.free.remove(at);
        }
        self.writes.push(SlotWrite::Objects(
            start,
            groups.tile_objects(key.0, pooled.origin, &pooled.placed),
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

fn bounds(records: &[PlacedRecord]) -> Option<Reach> {
    records
        .iter()
        .map(|record| {
            let center = Vec3::from(record.center);
            let reach = Vec3::new(record.radius, 2.0 * record.radius, record.radius);
            (center - reach, center + reach, record.radius)
        })
        .reduce(|(low, high, widest), (min, max, radius)| {
            (low.min(min), high.max(max), engine_math::max(widest, radius))
        })
}

fn stands(poses: &[PoseData]) -> (Vec2, Vec2) {
    let mut low = Vec2::INFINITY;
    let mut high = Vec2::NEG_INFINITY;
    for pose in poses {
        let at = Vec2::new(pose[0][3], pose[2][3]);
        if !at.is_finite() {
            return (Vec2::NEG_INFINITY, Vec2::INFINITY);
        }
        low = low.min(at);
        high = high.max(at);
    }
    (low, high)
}

fn stands_far(record: &PlacedRecord) -> bool {
    ScatterTerrain::unpack(record.terrain).is_some_and(|terrain| terrain.near != 0)
}

pub(crate) fn dead() -> PlacedRecord {
    PlacedRecord {
        center: [0.0; 3],
        radius: 0.0,
        group: DEAD,
        scale: 0.0,
        terrain: ScatterTerrain {
            index: TERRAIN_INDEX,
            layer: NO_LAYER,
            near: 0,
            rank: 0,
            hidden: true,
        }
        .pack(),
        sink: 0.0,
        pad: [0; 4],
    }
}
