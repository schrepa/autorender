use bevy_ecs::prelude::*;
use engine_core::{AssetLibrary, DrawnTable, Spawner};
use engine_places::impostors::{
    IMPOSTOR_MIN_TRIANGLES, ImpostorAsk, ImpostorAsks, ImpostorRecords,
};

use crate::MeshesBrick;
use crate::order::MeshBatches;

#[derive(Component)]
pub(crate) struct MeshAssets;

fn asks(assets: &AssetLibrary, batches: &MeshBatches) -> ImpostorAsks {
    ImpostorAsks {
        generation: batches.generation(),
        hemisphere: false,
        asks: batches
            .batches
            .iter()
            .map(|batch| {
                assets
                    .mesh(batch.mesh)
                    .is_some_and(|mesh| {
                        mesh.takes_impostor() && mesh.indices().len() / 3 >= IMPOSTOR_MIN_TRIANGLES
                    })
                    .then(|| ImpostorAsk {
                        mesh: batch.mesh,
                        material: batch.material,
                        parts: Vec::new(),
                    })
            })
            .collect(),
    }
}

pub(crate) fn publish_assets(
    (assets, table): (Res<AssetLibrary>, Res<DrawnTable>),
    mut batches: Local<MeshBatches>,
    own: Query<(Entity, &ImpostorAsks), With<MeshAssets>>,
    mut spawner: Spawner<MeshesBrick>,
) {
    let held = own.single();
    if !batches.settle(&table) && held.is_ok() {
        return;
    }
    let asked = asks(&assets, &batches);
    match held {
        Ok((entity, held_asks)) => {
            if *held_asks != asked {
                spawner.insert(entity, asked);
            }
        }
        Err(_) => {
            spawner.spawn((MeshAssets, asked, ImpostorRecords::default()));
        }
    }
}
