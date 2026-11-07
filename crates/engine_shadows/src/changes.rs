use bevy_ecs::prelude::*;
use engine_core::glam::{Mat3, Vec3};
use engine_core::{
    Cursor, DRAWN_UNBOUNDED, DrawKind, DrawnTable, Events, GlobalTransforms, MaterialHandle,
    MeshHandle, Mobility, PlanState, ScenePasses, Since,
};
use engine_places_rendering::shadows::{SHADOW_PASS, ShadowChange, ShadowChanges};

const MESH: DrawKind = DrawKind::MESH;

type Group = (MeshHandle, Option<MaterialHandle>);

type Bounds = (Vec3, Vec3);

#[derive(Default)]
pub(crate) struct HeldCasters {
    rows: Cursor,
    moves: Cursor,
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
    let casts = drawn.kind == MESH
        && record.passes & shadow != 0
        && record.flags & DRAWN_UNBOUNDED == 0
        && record.mobility == Mobility::Static as u32 + 1;
    if !casts {
        return None;
    }
    let matrix = globals.get(drawn.node)?.matrix()?;
    let low = Vec3::from_slice(&record.bounds_min[..3]);
    let high = Vec3::from_slice(&record.bounds_max[..3]);
    let center = matrix.transform_point3((low + high) * 0.5);
    let half = (high - low) * 0.5;
    let axes = Mat3::from_mat4(matrix);
    let reach = axes.x_axis.abs() * half.x + axes.y_axis.abs() * half.y + axes.z_axis.abs() * half.z;
    Some((center - reach, center + reach))
}

fn groups(table: &DrawnTable) -> Vec<Group> {
    table
        .batches()
        .iter()
        .filter(|batch| batch.kind == MESH)
        .filter_map(|batch| Some((batch.geometry?, batch.material)))
        .collect()
}

pub(crate) fn publish_shadow_changes(
    (table, globals, passes): (Res<DrawnTable>, Res<GlobalTransforms>, Option<Res<ScenePasses>>),
    mut held: Local<HeldCasters>,
    mut shadows: ResMut<ShadowChanges>,
) {
    let held = &mut *held;
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
