//! Animation's development tools: the inspectors of clip players and skins,
//! naming clips and entities by their authored IDs.

use engine_core::bevy_ecs::prelude::{Entity, World};
use engine_core::{AnimationPlayer, AssetLibrary, Engine, Skin};
use engine_scene::{SceneEntity, SceneInstance};
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::EngineControl;
use crate::inspect::Inspector;
use crate::views::{AnimationPlayerView, ClipView, SkinView};
use crate::{DevError, DevTools, InspectorDescriptor, InspectorKind};

/// Registers animation's inspectors.
pub(crate) fn register(tools: &mut DevTools, schemas: &Value) {
    // Animation: players and skins name clips
    // and entities by their authored IDs.
    let authored = |world: &World, entity: Option<Entity>| {
        entity
            .and_then(|entity| world.get::<SceneEntity>(entity))
            .map(|source| source.authored_id.clone())
    };
    tools
        .insert(Inspector {
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
    tools
        .insert(Inspector {
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
    schemas.insert(
        "AnimationPlayer".into(),
        super::schema::<crate::views::AnimationPlayerView>(),
    );
    schemas.insert("Skin".into(), super::schema::<crate::views::SkinView>());
}

/// Set an authored entity's animation player and render while paused:
/// its clip (an animation ID of the scene), time in seconds, speed and
/// looping; omitted values stay. Returns them before and after.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct AnimationControl {
    pub version: u32,
    pub entity: String,
    #[serde(default)]
    pub clip: Option<String>,
    #[serde(default)]
    pub time: Option<f32>,
    #[serde(default)]
    pub speed: Option<f32>,
    #[serde(default)]
    pub looping: Option<bool>,
}

impl EngineControl for AnimationControl {
    fn changed(&self) -> &'static str {
        "animation changed"
    }

    fn apply(&self, engine: &mut Engine, scene: Option<&SceneInstance>) -> Result<Value, DevError> {
        animate(engine, scene, self)
    }
}

/// Sets authored entity `change.entity`'s animation player as `change`
/// says: its values before and after.
fn animate(
    engine: &mut Engine,
    scene: Option<&SceneInstance>,
    change: &AnimationControl,
) -> Result<Value, DevError> {
    let id = change.entity.as_str();
    if !engine_core::has_feature(engine.app.world(), engine_core::Feature::Animation) {
        return Err(super::missing(engine_core::Feature::Animation));
    }
    let scene = scene.ok_or_else(|| {
        DevError::new(
            ErrorCode::NoScene,
            "$.entity",
            "animation players are authored entities",
        )
    })?;
    if let Some(time) = change.time
        && !(time.is_finite() && time >= 0.0)
    {
        return Err(DevError::new(
            ErrorCode::OutOfRange,
            "$.time",
            "must be finite and at least 0",
        ));
    }
    if let Some(speed) = change.speed
        && !(speed.is_finite() && speed.abs() <= 100.0)
    {
        return Err(DevError::new(
            ErrorCode::OutOfRange,
            "$.speed",
            "must be finite and within +/-100",
        ));
    }
    let clip = change
        .clip
        .as_deref()
        .map(|clip| {
            scene.clip(clip).ok_or_else(|| {
                DevError::new(
                    ErrorCode::UnknownReference,
                    "$.clip",
                    format!("the scene has no animation {clip}"),
                )
            })
        })
        .transpose()?;
    let world = engine.app.world_mut();
    let entity = crate::inspect::authored_entity(world, scene, id)
        .filter(|&entity| world.get::<AnimationPlayer>(entity).is_some())
        .ok_or_else(|| {
            DevError::new(
                ErrorCode::UnknownEntity,
                "$.entity",
                format!("no animation player {id} in the scene"),
            )
        })?;
    let mut player = world
        .get_mut::<AnimationPlayer>(entity)
        .expect("checked above");
    let values = |player: &AnimationPlayer| {
        let clip = scene
            .clips()
            .find(|&(_, handle)| handle == player.clip)
            .map(|(id, _)| id.to_owned());
        json!({"clip": clip, "time": player.time, "speed": player.speed, "looping": player.looping})
    };
    let before = values(&player);
    if let Some(clip) = clip {
        player.clip = clip;
    }
    if let Some(time) = change.time {
        player.time = time;
    }
    if let Some(speed) = change.speed {
        player.speed = speed;
    }
    if let Some(looping) = change.looping {
        player.looping = looping;
    }
    Ok(json!({"temporary": true, "before": before, "after": values(&player)}))
}
