//! Animation's development tools: the inspectors of clip players and skins,
//! naming clips and entities by their authored IDs.

use engine_core::bevy_ecs::prelude::{Entity, World};
use engine_core::{AnimationPlayer, AssetLibrary, Skin};
use engine_scene::SceneEntity;
use serde_json::{Value, json};

use crate::inspect::Inspector;
use crate::views::{AnimationPlayerView, ClipView, SkinView};
use crate::{DevTools, InspectorDescriptor, InspectorKind};

/// Registers animation's inspectors.
pub(crate) fn register(tools: &mut DevTools, schemas: &Value) {
    // Animation: players and skins name clips
    // and entities by their authored IDs.
    let authored = |world: &World, entity: Option<Entity>| {
        entity
            .and_then(|entity| world.get::<SceneEntity>(entity))
            .map(|source| source.authored_id.clone())
    };
    tools.insert(Inspector {
        descriptor: InspectorDescriptor::generated(schemas, "AnimationPlayer"),
        kind: InspectorKind::Component,
        has: Box::new(|world, entity| {
            entity.is_some_and(|entity| world.get::<AnimationPlayer>(entity).is_some())
        }),
        read: Box::new(move |world, entity, context| {
            let scene = context.scene;
            let player = world.get::<AnimationPlayer>(entity?)?;
            let clip = world
                .get_resource::<AssetLibrary>()
                .and_then(|assets| assets.clip(player.clip))
                .map(|clip| {
                    let origin = scene.and_then(|scene| scene.clip_origin(player.clip));
                    ClipView {
                        id: origin.map(|(id, _)| id.to_owned()),
                        name: origin.map(|(_, origin)| origin.name.clone()),
                        targets: origin
                            .map(|(_, origin)| origin.targets.clone())
                            .unwrap_or_default(),
                        channels: clip.channels().len(),
                        uri: origin.and_then(|(_, origin)| origin.uri.clone()),
                        duration: clip.duration(),
                    }
                });
            Some(json!(AnimationPlayerView {
                clip,
                time: player.time,
                speed: player.speed,
                looping: player.looping,
                targets: player
                    .targets
                    .iter()
                    .map(|&target| authored(world, target))
                    .collect(),
            }))
        }),
    })
    .expect("built-in descriptor is bounded");
    tools.insert(Inspector {
        descriptor: InspectorDescriptor::generated(schemas, "Skin"),
        kind: InspectorKind::Component,
        has: Box::new(|world, entity| {
            entity.is_some_and(|entity| world.get::<Skin>(entity).is_some())
        }),
        read: Box::new(move |world, entity, context| {
            let skin = world.get::<Skin>(entity?)?;
            Some(json!(SkinView {
                skin: context
                    .scene
                    .and_then(|scene| scene.skin_id(skin))
                    .map(str::to_owned),
                joints: skin
                    .joints
                    .iter()
                    .map(|&joint| authored(world, Some(joint)))
                    .collect(),
                inverse_binds: skin.inverse_binds.len(),
            }))
        }),
    })
    .expect("built-in descriptor is bounded");
}

/// Adds the schemas of animation's inspectors.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn schemas(schemas: &mut serde_json::Map<String, Value>) {
    schemas.insert("AnimationPlayer".into(), super::schema::<crate::views::AnimationPlayerView>());
    schemas.insert("Skin".into(), super::schema::<crate::views::SkinView>());
}
