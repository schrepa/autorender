use engine::prelude::*;
use engine::scene::SceneFiles;
use engine_collision::CollisionConfig;
use engine_fly_camera::FlyCamera;
use engine_physics::Physics;

use crate::land;

pub(crate) const SCENE: &str = include_str!("../highlands.scene.json");

const SPEED: f32 = 160.0;

pub(crate) fn furnish(
    world: &mut World,
    document: &SceneDocument,
    files: &SceneFiles,
    flying: bool,
) -> engine::Result<SceneInstance> {
    let (instance, ()) = document.instantiate_then(world, files, |world, _| {
        land::populate(world, document)?;
        world
            .section::<CollisionConfig>()
            .set(|collision| collision.enabled = false)?;
        world
            .section::<Physics>()
            .set(|physics| physics.enabled = false)?;
        if flying {
            world.section::<FlyCamera>().set(|fly| fly.speed = SPEED)
        } else {
            world.section::<FlyCamera>().set(|fly| fly.enabled = false)
        }
    })?;
    Ok(instance)
}
