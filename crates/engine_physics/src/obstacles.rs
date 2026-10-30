use bevy_ecs::prelude::*;
use engine_collision::{ColliderShape, CollisionWorld};
use engine_core::{WaterBodies, WaterObstacle, WaterObstacles};
use glam::Vec3;

use crate::PhysicsWorld;

#[derive(Resource, Default)]
pub(crate) struct WaterReaches(Vec<(Vec3, Vec3)>);

#[derive(Default)]
pub(crate) struct Standing {
    revision: Option<u64>,
    reaches: Vec<(Vec3, Vec3)>,
    colliders: Vec<WaterObstacle>,
}

pub(crate) fn reach_water(world: &mut World) {
    let reaches = match world.get_resource::<WaterBodies>() {
        Some(bodies) => bodies.reaches(world),
        None => Vec::new(),
    };
    world.resource_mut::<WaterReaches>().0 = reaches;
}

pub(crate) fn obstruct(
    physics: Res<PhysicsWorld>,
    collision: Option<Res<CollisionWorld>>,
    reaches: Res<WaterReaches>,
    obstacles: Option<ResMut<WaterObstacles>>,
    mut standing: Local<Standing>,
) {
    let Some(mut obstacles) = obstacles else {
        return;
    };
    let reaches = &reaches.0;
    if reaches.is_empty() {
        if !obstacles.obstacles.is_empty() {
            obstacles.obstacles.clear();
        }
        return;
    }
    let revision = collision
        .as_ref()
        .map(|collision| collision.report().revision);
    if revision != standing.revision || *reaches != standing.reaches {
        standing.revision = revision;
        standing.colliders.clear();
        if let Some(collision) = collision {
            let mut seen = Vec::new();
            for &(low, high) in reaches {
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
        standing.reaches.clone_from(reaches);
    }
    let listed = &mut obstacles.obstacles;
    listed.clear();
    listed.extend_from_slice(&standing.colliders);
    listed.extend(physics.solids());
}
