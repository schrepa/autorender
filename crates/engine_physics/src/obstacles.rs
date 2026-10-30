use bevy_ecs::prelude::*;
use engine_collision::{ColliderShape, CollisionWorld};
use engine_core::{WaterBodies, WaterObstacle, WaterObstacles};
use glam::Vec3;

use crate::PhysicsWorld;

#[derive(Default)]
pub(crate) struct Standing {
    revision: Option<u64>,
    reaches: Vec<(Vec3, Vec3)>,
    colliders: Vec<WaterObstacle>,
}

pub(crate) fn obstruct(world: &mut World, mut standing: Local<Standing>) {
    if !world.contains_resource::<WaterObstacles>() {
        return;
    }
    let reaches = match world.get_resource::<WaterBodies>() {
        Some(bodies) => bodies.reaches(world),
        None => Vec::new(),
    };
    world.resource_scope(|world, mut obstacles: Mut<WaterObstacles>| {
        if reaches.is_empty() {
            if !obstacles.obstacles.is_empty() {
                obstacles.obstacles.clear();
            }
            return;
        }
        let physics = world.resource::<PhysicsWorld>();
        let collision = world.get_resource::<CollisionWorld>();
        let revision = collision.map(|collision| collision.report().revision);
        if revision != standing.revision || reaches != standing.reaches {
            standing.revision = revision;
            standing.colliders.clear();
            if let Some(collision) = collision {
                let mut seen = Vec::new();
                for &(low, high) in &reaches {
                    for found in collision.collider_solids_in(low, high) {
                        if physics.holds(found.entity) || seen.contains(&found.entity) {
                            continue;
                        }
                        seen.push(found.entity);
                        standing.colliders.push(WaterObstacle {
                            center: found.solid.center,
                            axes: found.solid.axes,
                            round: matches!(found.solid.shape, ColliderShape::Sphere { .. }),
                            velocity: Vec3::ZERO,
                        });
                    }
                }
            }
            standing.reaches = reaches;
        }
        let listed = &mut obstacles.obstacles;
        listed.clear();
        listed.extend_from_slice(&standing.colliders);
        listed.extend(physics.solids());
    });
}
