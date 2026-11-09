use bevy_ecs::prelude::*;
use engine_core::{DrawnTable, Spawner};
use engine_places::meshes::{LastLevels, LevelBatch, LevelBatches};

use crate::order::MeshBatches;
use crate::{MESH, MeshesBrick, SKIN};

#[derive(Component)]
pub(crate) struct MeshAssets;

fn listed(batches: &MeshBatches) -> LevelBatches {
    LevelBatches {
        generation: batches.generation(),
        batches: batches
            .batches
            .iter()
            .map(|batch| LevelBatch {
                kind: if batch.skinned { SKIN } else { MESH },
                mesh: batch.mesh,
                material: batch.material,
            })
            .collect(),
    }
}

pub(crate) fn publish_assets(
    table: Res<DrawnTable>,
    mut batches: Local<MeshBatches>,
    own: Query<(Entity, &LevelBatches), With<MeshAssets>>,
    mut spawner: Spawner<MeshesBrick>,
) {
    let held = own.single();
    if !batches.settle(&table) && held.is_ok() {
        return;
    }
    let listed = listed(&batches);
    match held {
        Ok((entity, held)) => {
            if *held != listed {
                spawner.insert(entity, listed);
            }
        }
        Err(_) => {
            spawner.spawn((MeshAssets, listed, LastLevels::default()));
        }
    }
}
