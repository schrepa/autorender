//! The animation control: an authored animation player's clip, time,
//! speed and looping, set while paused.

use engine_core::{AnimationPlayer, Engine};
use engine_scene::SceneInstance;
use engine_settings::ErrorCode;
use serde_json::{Value, json};

use crate::DevError;

/// What the animation control changes of a player; omitted values stay.
pub struct Animate {
    pub clip: Option<String>,
    pub time: Option<f32>,
    pub speed: Option<f32>,
    pub looping: Option<bool>,
}

/// Sets authored entity `id`'s animation player as `change` says: its
/// values before and after.
pub fn animate(
    engine: &mut Engine,
    scene: Option<&SceneInstance>,
    id: &str,
    change: Animate,
) -> Result<Value, DevError> {
    if !engine_core::has_feature(engine.app.world(), engine_core::Feature::Animation) {
        return Err(crate::terrain::missing(engine_core::Feature::Animation));
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
