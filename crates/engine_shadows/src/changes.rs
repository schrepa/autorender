use bevy_ecs::prelude::*;
use engine_core::glam::{Mat3, Vec3};
use engine_core::{
    Cursor, DrawnArea, DrawnAreas, DrawnFlags, DrawnTable, Events, Feed, GeometryHandle,
    GlobalTransforms, MaterialHandle, Mobility, PlanState, ScenePasses, Since,
};
use engine_places_rendering::shadows::SHADOW_PASS;

use crate::ShadowSettings;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ShadowChange {
    Box { min: Vec3, max: Vec3, spacing: f32 },
    All,
    Casters,
}

#[derive(Resource, Clone, Debug)]
pub(crate) struct ShadowChanges {
    feed: Feed<ShadowChange>,
}

impl Default for ShadowChanges {
    fn default() -> Self {
        Self { feed: Feed::new(1) }
    }
}

impl ShadowChanges {
    fn push(&mut self, change: ShadowChange) {
        self.feed.push(change);
    }

    fn resize(&mut self, size: usize) {
        self.feed.resize(size);
    }

    pub(crate) fn since<'a>(&'a self, cursor: &mut Cursor) -> Since<'a, ShadowChange> {
        self.feed.since(cursor)
    }
}

type Group = (GeometryHandle, Option<MaterialHandle>);

type Bounds = (Vec3, Vec3);

#[derive(Default)]
pub(crate) struct HeldCasters {
    rows: Cursor,
    moves: Cursor,
    areas: Cursor,
    generation: Option<u64>,
    groups: Vec<Group>,
    boxes: Vec<Option<Bounds>>,
}

fn caster_box(
    table: &DrawnTable,
    globals: &GlobalTransforms,
    shadow: u32,
    slot: u32,
) -> Option<Bounds> {
    let record = table.records().get(slot as usize)?;
    let drawn = table.drawn(slot)?;
    let casts = drawn.geometry.is_some()
        && record.passes & shadow != 0
        && !DrawnFlags::of(record).unbounded
        && record.mobility == Mobility::Static as u32 + 1;
    if !casts {
        return None;
    }
    let matrix = globals.get(table.owner(slot)?)?.matrix()?;
    let low = Vec3::from_slice(&record.bounds_min[..3]);
    let high = Vec3::from_slice(&record.bounds_max[..3]);
    let center = matrix.transform_point3((low + high) * 0.5);
    let half = (high - low) * 0.5;
    let axes = Mat3::from_mat4(matrix);
    let reach =
        axes.x_axis.abs() * half.x + axes.y_axis.abs() * half.y + axes.z_axis.abs() * half.z;
    Some((center - reach, center + reach))
}

fn groups(table: &DrawnTable) -> Vec<Group> {
    table
        .batches()
        .iter()
        .filter_map(|batch| Some((batch.geometry?, batch.material)))
        .collect()
}

fn publish_areas(areas: &DrawnAreas, cursor: &mut Cursor, shadows: &mut ShadowChanges) {
    match areas.since(cursor) {
        Since::Same => {}
        Since::Listed(listed) => {
            for area in listed {
                shadows.push(match *area {
                    DrawnArea::Box { min, max, spacing } => ShadowChange::Box { min, max, spacing },
                    DrawnArea::All => ShadowChange::All,
                });
            }
        }
        Since::All => shadows.push(ShadowChange::All),
    }
}

pub(crate) fn publish_shadow_changes(
    (table, globals, passes): (
        Res<DrawnTable>,
        Res<GlobalTransforms>,
        Option<Res<ScenePasses>>,
    ),
    (settings, areas): (Res<ShadowSettings>, Res<DrawnAreas>),
    mut held: Local<HeldCasters>,
    mut shadows: ResMut<ShadowChanges>,
) {
    shadows
        .bypass_change_detection()
        .resize(settings.changes as usize);
    let held = &mut *held;
    publish_areas(&areas, &mut held.areas, &mut shadows);
    let shadow = passes
        .and_then(|passes| passes.bit(SHADOW_PASS))
        .map_or(0, |bit| bit.mask().word());
    let mut casters = false;
    if let PlanState::Built { generation } = table.plan()
        && held.generation != Some(generation)
    {
        held.generation = Some(generation);
        let next = groups(&table);
        if next != held.groups {
            held.groups = next;
            casters = true;
        }
    }
    held.boxes.resize(table.records().len(), None);
    let moved = match globals.static_moves_since(&mut held.moves) {
        Events::Listed(listed) => Some(!listed.is_empty()),
        Events::Missed(_) => None,
    };
    let listed = match table.changes_since(&mut held.rows) {
        Since::Same => Some(&[][..]),
        Since::Listed(listed) => Some(listed),
        Since::All => None,
    };
    let (Some(moved), Some(listed)) = (moved, listed) else {
        for (slot, kept) in held.boxes.iter_mut().enumerate() {
            *kept = caster_box(&table, &globals, shadow, slot as u32);
        }
        shadows.push(ShadowChange::All);
        shadows.push(ShadowChange::Casters);
        return;
    };
    let mut check = |slot: u32| {
        let next = caster_box(&table, &globals, shadow, slot);
        let Some(kept) = held.boxes.get_mut(slot as usize) else {
            return;
        };
        if *kept == next {
            return;
        }
        for (min, max) in kept.iter().chain(next.iter()) {
            shadows.push(ShadowChange::Box {
                min: *min,
                max: *max,
                spacing: f32::INFINITY,
            });
        }
        *kept = next;
        casters = true;
    };
    if moved {
        for slot in 0..table.records().len() as u32 {
            check(slot);
        }
    } else {
        for &slot in listed {
            check(slot);
        }
    }
    if casters {
        shadows.push(ShadowChange::Casters);
    }
}
