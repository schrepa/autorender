use bevy_ecs::prelude::*;
use engine_core::{Attribute, Cut, DrawnTable, Spawner};
use engine_places::meshes::{LastLevel, LastLevels, LevelBatches};
use engine_places_rendering::impostors::{
    IMPOSTOR_VERTICES, ImpostorAsk, ImpostorAsks, ImpostorRecords,
};

use crate::ImpostorsBrick;
use crate::settings::ImpostorStats;

#[derive(Component)]
pub(crate) struct MeshCards;

fn asks_of(batches: &LevelBatches, table: &DrawnTable) -> ImpostorAsks {
    ImpostorAsks {
        generation: batches.generation,
        hemisphere: false,
        asks: batches
            .batches
            .iter()
            .map(|batch| {
                let rigid = table
                    .kind(batch.kind)
                    .is_some_and(|declared| !declared.needs.has(Attribute::Joints));
                rigid.then(|| ImpostorAsk {
                    mesh: batch.mesh,
                    material: batch.material,
                    parts: Vec::new(),
                })
            })
            .collect(),
    }
}

pub(crate) fn ask_meshes(
    table: Res<DrawnTable>,
    meshes: Query<Ref<LevelBatches>>,
    own: Query<(Entity, &ImpostorAsks), With<MeshCards>>,
    mut spawner: Spawner<ImpostorsBrick>,
) {
    let Ok(batches) = meshes.single() else {
        return;
    };
    let held = own.single();
    if !batches.is_changed() && held.is_ok() {
        return;
    }
    let asks = asks_of(&batches, &table);
    match held {
        Ok((entity, held)) => {
            if *held != asks {
                spawner.insert(entity, asks);
            }
        }
        Err(_) => {
            spawner.spawn((MeshCards, asks, ImpostorRecords::default()));
        }
    }
}

pub(crate) fn fill_levels(
    own: Query<Ref<ImpostorRecords>, With<MeshCards>>,
    mut meshes: Query<(Ref<LevelBatches>, &mut LastLevels)>,
    mut stats: ResMut<ImpostorStats>,
) {
    let records = own.single().ok();
    let renewed = records.as_ref().is_some_and(|records| records.is_changed());
    for (batches, mut levels) in &mut meshes {
        if !renewed && !batches.is_changed() {
            continue;
        }
        let held = records
            .as_ref()
            .filter(|records| records.generation == batches.generation);
        levels.set_if_neq(LastLevels {
            generation: batches.generation,
            levels: held.map_or_else(Vec::new, |records| {
                records
                    .records
                    .iter()
                    .map(|record| LastLevel {
                        error: record.error(),
                        triangles: IMPOSTOR_VERTICES / 3,
                    })
                    .collect()
            }),
        });
    }
    let (live, cut) = records.map_or((0, Cut::default()), |records| {
        (
            records.live,
            Cut::of(records.asked, records.asked.saturating_sub(records.refused)),
        )
    });
    let next = ImpostorStats {
        mesh_impostors: live,
        mesh_cut: cut,
        ..stats.clone()
    };
    stats.set_if_neq(next);
}
