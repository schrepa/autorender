use bevy_ecs::prelude::*;
use engine_core::{
    Asset, AssetLibrary, AssetResidency, AssetUses, IMPOSTOR_MIN_TRIANGLES, ImpostorAsk,
    ImpostorAsks, ImpostorPolicy, Spawner,
};

use crate::MeshesBrick;
use crate::batches::BatchPlan;
use crate::settings::Impostors;

#[derive(Component)]
pub(crate) struct MeshAssets;

fn uses(plan: &BatchPlan) -> Vec<Asset> {
    let mut wanted: Vec<Asset> = plan
        .batches
        .iter()
        .flat_map(|batch| {
            std::iter::once(Asset::Mesh(batch.mesh)).chain(batch.material.map(Asset::Material))
        })
        .collect();
    wanted.sort_unstable();
    wanted.dedup();
    wanted
}

fn asks(assets: &AssetLibrary, plan: &BatchPlan) -> ImpostorAsks {
    ImpostorAsks {
        generation: plan.generation(),
        hemisphere: false,
        asks: plan
            .batches
            .iter()
            .map(|batch| {
                assets
                    .mesh(batch.mesh)
                    .is_some_and(|mesh| {
                        !batch.first_person
                            && mesh.takes_impostor()
                            && mesh.indices().len() / 3 >= IMPOSTOR_MIN_TRIANGLES
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
    assets: Res<AssetLibrary>,
    residency: Res<AssetResidency>,
    plan: Res<BatchPlan>,
    own: Query<(Entity, &AssetUses, &ImpostorAsks), With<MeshAssets>>,
    mut spawner: Spawner<MeshesBrick>,
) {
    let held = own.single();
    if plan.refused.is_some() || (held.is_ok() && !plan.changed()) {
        return;
    }
    let wanted = uses(&plan);
    let asked = asks(&assets, &plan);
    match held {
        Ok((entity, held, held_asks)) => {
            if held.assets() != wanted.as_slice() {
                spawner.insert(entity, AssetUses::new(&residency, wanted));
            }
            if *held_asks != asked {
                spawner.insert(entity, asked);
            }
        }
        Err(_) => {
            spawner.spawn((MeshAssets, AssetUses::new(&residency, wanted), asked));
        }
    }
}

pub(crate) fn impostor_policy(impostors: Res<Impostors>, mut policy: ResMut<ImpostorPolicy>) {
    policy.set_if_neq(ImpostorPolicy {
        enabled: impostors.enabled,
        pixels: impostors.pixels,
        max: impostors.max,
    });
}
