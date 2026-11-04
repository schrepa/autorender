use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, AssetUses, IMPOSTOR_MIN_TRIANGLES, ImpostorAsk, ImpostorAsks, ImpostorPolicy,
    MaterialHandle, MeshHandle, Spawner, TextureHandle,
};

use crate::MeshesBrick;
use crate::batches::BatchPlan;
use crate::settings::Impostors;

#[derive(Component)]
pub(crate) struct MeshAssets;

fn uses(assets: &AssetLibrary, plan: &BatchPlan) -> AssetUses {
    let mut meshes: Vec<MeshHandle> = plan.batches.iter().map(|batch| batch.mesh).collect();
    meshes.dedup();
    let mut materials: Vec<MaterialHandle> = plan
        .batches
        .iter()
        .filter_map(|batch| batch.material)
        .collect();
    materials.sort_unstable();
    materials.dedup();
    let mut textures: Vec<TextureHandle> = materials
        .iter()
        .filter_map(|&handle| assets.material(handle))
        .flat_map(|material| material.textures())
        .collect();
    textures.sort_unstable();
    textures.dedup();
    AssetUses {
        meshes,
        materials,
        textures,
    }
}

fn asks(assets: &AssetLibrary, plan: &BatchPlan) -> ImpostorAsks {
    ImpostorAsks {
        hemisphere: false,
        asks: plan
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
    assets: Res<AssetLibrary>,
    plan: Res<BatchPlan>,
    own: Query<(Entity, &AssetUses, &ImpostorAsks), With<MeshAssets>>,
    mut spawner: Spawner<MeshesBrick>,
) {
    let held = own.single();
    if plan.refused.is_some() || (held.is_ok() && !plan.changed()) {
        return;
    }
    let used = uses(&assets, &plan);
    let asked = asks(&assets, &plan);
    match held {
        Ok((entity, held, held_asks)) => {
            if *held != used {
                spawner.insert(entity, used);
            }
            if *held_asks != asked {
                spawner.insert(entity, asked);
            }
        }
        Err(_) => {
            spawner.spawn((MeshAssets, used, asked));
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
