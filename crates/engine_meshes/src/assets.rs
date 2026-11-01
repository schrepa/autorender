use bevy_ecs::prelude::*;
use engine_core::{
    AssetUses, IMPOSTOR_MIN_TRIANGLES, ImpostorAsk, ImpostorAsks, ImpostorPolicy, Spawner,
};

use crate::MeshesBrick;
use crate::batches::BatchPlan;
use crate::retained::RetainedRender;
use crate::settings::Impostors;

#[derive(Component)]
pub(crate) struct MeshAssets;

fn uses(plan: &BatchPlan) -> AssetUses {
    AssetUses {
        meshes: plan.mesh_handles().to_vec(),
        materials: plan.material_handles().to_vec(),
        textures: plan.texture_handles().to_vec(),
    }
}

fn asks(retained: &RetainedRender, plan: &BatchPlan) -> ImpostorAsks {
    let snapshot = retained.snapshot();
    ImpostorAsks {
        hemisphere: false,
        asks: plan
            .batches
            .iter()
            .map(|batch| {
                let source = &snapshot.meshes[batch.source_index].mesh;
                (source.indices().len() / 3 >= IMPOSTOR_MIN_TRIANGLES).then(|| ImpostorAsk {
                    mesh: batch.mesh,
                    material: batch.material,
                    parts: Vec::new(),
                })
            })
            .collect(),
    }
}

pub(crate) fn publish_assets(
    retained: Res<RetainedRender>,
    plan: Res<BatchPlan>,
    own: Query<(Entity, &AssetUses, &ImpostorAsks), With<MeshAssets>>,
    mut spawner: Spawner<MeshesBrick>,
    mut published: Local<Option<u64>>,
) {
    let key = plan.rebuilds();
    if plan.refused.is_some() || (*published == Some(key) && own.single().is_ok()) {
        return;
    }
    *published = Some(key);
    let used = uses(&plan);
    let asked = asks(&retained, &plan);
    match own.single() {
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
