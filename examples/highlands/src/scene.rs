use engine::prelude::*;
use serde_json::{Value, json};

use crate::land;

pub(crate) const SCENE: &str = include_str!("../highlands.scene.json");

const SPEED: f32 = 160.0;

pub(crate) fn set(world: &mut World, key: &str, value: Value) -> engine::Result<()> {
    engine::host::apply_setting(world, key, &value)
}

pub(crate) fn furnish(
    world: &mut World,
    document: &SceneDocument,
    flying: bool,
) -> engine::Result<()> {
    land::populate(world, document)?;
    set(world, "collision.enabled", json!(false))?;
    set(world, "physics.enabled", json!(false))?;
    if flying {
        set(world, "fly_camera.speed", json!(SPEED))
    } else {
        set(world, "fly_camera.enabled", json!(false))
    }
}

pub(crate) fn start(world: &World) -> Option<(Transform, Camera)> {
    main_camera(world).and_then(|camera| {
        Some((*world.get::<Transform>(camera)?, *world.get::<Camera>(camera)?))
    })
}
