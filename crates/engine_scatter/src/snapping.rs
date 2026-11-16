use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::prelude::Entity;
use engine_core::Stamp;
use engine_math::collections::{HashMap, HashSet};
use engine_open_world_kit::ground::{GroundBaking, GroundTiles, TerrainEntry, TerrainTiles};
use engine_tiles::TileKey;
use glam::{Vec2, Vec3};

use crate::pool::{Reach, ScatterPool};

type TileId = (Entity, TileKey);

pub(crate) struct SnapInputs<'a> {
    pub(crate) tiles: &'a TerrainTiles,
    pub(crate) ground: &'a GroundTiles,
    pub(crate) baking: &'a GroundBaking,
    pub(crate) assets: ((u64, u64), u64),
    pub(crate) changed: bool,
}

struct Seen {
    tiles: TerrainTiles,
    ground: GroundTiles,
    baking: GroundBaking,
    assets: ((u64, u64), u64),
}

enum Moved {
    All,
    Tiles(Vec<TileId>),
}

const MOST_RECTS: usize = 64;

#[derive(Clone)]
pub(crate) struct Span {
    slots: Range<u32>,
    filtered: bool,
}

#[derive(Clone, Default)]
pub(crate) struct Planned {
    spans: Vec<Span>,
    rects: Vec<[f32; 4]>,
    boxes: Vec<Reach>,
    full: bool,
}

impl Planned {
    pub(crate) fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

#[derive(Default)]
pub(crate) struct SnapHeld {
    pub(crate) snapped: Option<u64>,
    pub(crate) snapping: bool,
    pub(crate) dispatched: bool,
    pub(crate) threads: u32,
    written: Vec<Range<u32>>,
    whole: bool,
    again: Planned,
    seen: Option<Seen>,
    running: Option<(bool, Vec<Reach>)>,
}

impl SnapHeld {
    pub(crate) fn wrote(&mut self, slots: Range<u32>) {
        if !slots.is_empty() {
            self.written.push(slots);
        }
    }

    pub(crate) fn wrote_whole(&mut self) {
        self.whole = true;
    }

    pub(crate) fn executed(&mut self) -> Option<(bool, Vec<Reach>)> {
        let ran = self.snapping && std::mem::take(&mut self.dispatched);
        if self.snapping && !ran {
            self.snapped = None;
        }
        self.running.take().filter(|_| ran)
    }

    pub(crate) fn running(&mut self, planned: &Planned) {
        self.running = Some((planned.full, planned.boxes.clone()));
    }

    pub(crate) fn plan(
        &mut self,
        inputs: &SnapInputs<'_>,
        (pool, generation, objects): (&ScatterPool, u64, u32),
    ) -> Planned {
        let mut full = self.whole || self.snapped != Some(generation);
        let mut now = Planned::default();
        now.spans.extend(self.written.drain(..).map(|slots| Span {
            slots,
            filtered: false,
        }));
        let looked = inputs.changed
            || self
                .seen
                .as_ref()
                .is_none_or(|seen| seen.assets != inputs.assets);
        if looked {
            match self.seen.as_ref().map(|seen| seen.moved(inputs)) {
                Some(Moved::Tiles(changed)) => {
                    full |= !region(&mut now, (pool, inputs.tiles), &changed);
                }
                Some(Moved::All) | None => full = true,
            }
            self.seen = Some(Seen {
                tiles: inputs.tiles.clone(),
                ground: inputs.ground.clone(),
                baking: inputs.baking.clone(),
                assets: inputs.assets,
            });
        }
        if full {
            now.spans.clear();
            now.spans.push(Span {
                slots: 0..objects,
                filtered: false,
            });
            now.rects.clear();
            now.boxes.clear();
            now.full = true;
        }
        let now = merged(now, objects);
        let again = std::mem::replace(&mut self.again, now.clone());
        self.whole = false;
        self.snapped = Some(generation);
        let mut run = now;
        run.spans.extend(again.spans);
        run.rects.extend(again.rects);
        merged(run, objects)
    }

    pub(crate) fn words(&mut self, planned: &Planned, terrains: u32) -> Vec<[u32; 4]> {
        let (spans, rects) = (&planned.spans, &planned.rects);
        let mut words = vec![[0, terrains, spans.len() as u32, rects.len() as u32]];
        let mut threads = 0;
        for span in spans {
            words.push([span.slots.start, threads, u32::from(span.filtered), 0]);
            threads += span.slots.end - span.slots.start;
        }
        words.extend(rects.iter().map(|rect| rect.map(f32::to_bits)));
        words[0][0] = threads;
        self.threads = threads;
        words
    }
}

impl Seen {
    fn moved(&self, now: &SnapInputs<'_>) -> Moved {
        let (before, after) = (&self.ground, now.ground);
        let kept = self.assets == now.assets
            && before.layers == after.layers
            && before.tables == after.tables
            && before.palette == after.palette
            && before.whole == after.whole
            && self.baking.terrains == now.baking.terrains
            && same_terrains(&self.tiles, now.tiles);
        if !kept {
            return Moved::All;
        }
        let mut held: HashMap<TileId, (Stamp, u32)> = before
            .held
            .iter()
            .map(|tile| ((tile.terrain, tile.key), (tile.stamp, tile.layer)))
            .collect();
        let mut changed: Vec<TileId> = Vec::new();
        for tile in &after.held {
            let id = (tile.terrain, tile.key);
            if held.remove(&id) != Some((tile.stamp, tile.layer)) {
                changed.push(id);
            }
        }
        changed.extend(held.into_keys());
        let shaded: HashSet<TileId> = self.baking.shaded.iter().copied().collect();
        let shading: HashSet<TileId> = now.baking.shaded.iter().copied().collect();
        changed.extend(shaded.symmetric_difference(&shading).copied());
        let derived: HashSet<_> = self.baking.derived.iter().copied().collect();
        let deriving: HashSet<_> = now.baking.derived.iter().copied().collect();
        let derivations = derived.symmetric_difference(&deriving);
        changed.extend(derivations.map(|&(terrain, key, _)| (terrain, key)));
        Moved::Tiles(changed)
    }
}

fn same_terrains(before: &TerrainTiles, after: &TerrainTiles) -> bool {
    Arc::ptr_eq(&before.terrains, &after.terrains)
        || (before.terrains.len() == after.terrains.len()
            && before
                .terrains
                .iter()
                .zip(after.terrains.iter())
                .all(|(before, after)| same_ground(before, after)))
}

fn same_ground(a: &TerrainEntry, b: &TerrainEntry) -> bool {
    let same_baked = match (&a.baked, &b.baked) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        (Some(_), None) | (None, Some(_)) => false,
    };
    a.terrain == b.terrain
        && a.origin == b.origin
        && a.pyramid == b.pyramid
        && a.cutout == b.cutout
        && a.surroundings == b.surroundings
        && a.steep == b.steep
        && a.recipe == b.recipe
        && a.edits == b.edits
        && a.relief == b.relief
        && a.pinned_levels == b.pinned_levels
        && a.surface_program == b.surface_program
        && a.site_ranges == b.site_ranges
        && same_baked
        && a.macro_map == b.macro_map
        && a.materials == b.materials
        && a.maps == b.maps
        && a.placement == b.placement
        && a.layers == b.layers
        && a.sample_offset == b.sample_offset
}

fn region(
    planned: &mut Planned,
    (pool, tiles): (&ScatterPool, &TerrainTiles),
    changed: &[TileId],
) -> bool {
    let mut rects: Vec<(Entity, Vec2, Vec2)> = Vec::with_capacity(changed.len());
    for &(terrain, key) in changed {
        let Some(entry) = tiles.terrains.iter().find(|entry| entry.terrain == terrain) else {
            return false;
        };
        let level = u32::from(key.level);
        let size = entry.pyramid.tile_size(level);
        let margin = Vec2::splat(entry.pyramid.spacing(level));
        let low = Vec2::new(entry.origin.x, entry.origin.z)
            + Vec2::new(f32::from(key.x), f32::from(key.z)) * size;
        rects.push((terrain, low - margin, low + Vec2::splat(size) + margin));
    }
    for (slots, filtered, bounds) in pool.standing_in(&rects) {
        planned.spans.push(Span { slots, filtered });
        planned.boxes.extend(bounds.map(|bounds| widened(tiles, bounds)));
    }
    planned
        .rects
        .extend(rects.iter().map(|&(_, low, high)| [low.x, low.y, high.x, high.y]));
    true
}

fn merged(mut planned: Planned, objects: u32) -> Planned {
    let spans = &mut planned.spans;
    for span in spans.iter_mut() {
        span.slots.end = span.slots.end.min(objects);
    }
    spans.retain(|span| span.slots.start < span.slots.end);
    spans.sort_unstable_by_key(|span| span.slots.start);
    let wide = planned.rects.len() > MOST_RECTS;
    let mut joined: Vec<Span> = Vec::with_capacity(spans.len());
    for mut span in spans.drain(..) {
        span.filtered &= !wide;
        match joined.last_mut() {
            Some(last)
                if span.slots.start < last.slots.end
                    || (span.slots.start == last.slots.end && span.filtered == last.filtered) =>
            {
                last.slots.end = last.slots.end.max(span.slots.end);
                last.filtered &= span.filtered;
            }
            Some(_) | None => joined.push(span),
        }
    }
    planned.spans = joined;
    if wide || planned.spans.iter().all(|span| !span.filtered) {
        planned.rects.clear();
    }
    planned
}

pub(crate) fn widened(
    tiles: &TerrainTiles,
    (terrain, (low, high, widest)): (Entity, Reach),
) -> Reach {
    let slack: f32 = tiles
        .terrains
        .iter()
        .find(|entry| entry.terrain == terrain)
        .map_or(0.0, |entry| entry.level_errors.iter().sum());
    (low - Vec3::Y * slack, high + Vec3::Y * slack, widest)
}
