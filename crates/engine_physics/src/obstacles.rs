use std::cmp::Reverse;

use bevy_ecs::prelude::*;
use engine_places::physics::ColliderShape;
use engine_places::water::{WaterBody, WaterObstacle, WaterObstacles};
use glam::Vec3;

use crate::solids::Solids;
use crate::world::PhysicsWorld;

#[derive(Default)]
pub(crate) struct Standing {
    revision: Option<u64>,
    reaches: Vec<(Vec3, Vec3)>,
    wanted: Vec<(Vec3, Vec3)>,
    colliders: Vec<WaterObstacle>,
}

pub(crate) fn sorted<'a>(waters: impl Iterator<Item = &'a WaterBody>) -> Vec<&'a WaterBody> {
    let mut held: Vec<&WaterBody> = waters.collect();
    held.sort_by_key(|body| Reverse(body.precedence));
    held
}

pub(crate) fn obstruct(
    physics: Res<PhysicsWorld>,
    solids: Res<Solids>,
    waters: Query<&WaterBody>,
    mut obstacles: ResMut<WaterObstacles>,
    mut standing: Local<Standing>,
) {
    let standing = &mut *standing;
    standing.wanted.clear();
    for body in sorted(waters.iter()) {
        standing.wanted.extend(body.shape.reaches());
    }
    let reaches = &standing.wanted;
    if reaches.is_empty() {
        if !obstacles.obstacles.is_empty() {
            obstacles.obstacles.clear();
        }
        return;
    }
    let revision = Some(solids.revision());
    if revision != standing.revision || *reaches != standing.reaches {
        standing.revision = revision;
        standing.colliders.clear();
        let mut seen = Vec::new();
        for &(low, high) in reaches {
            for found in solids.colliders_in(low, high) {
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
        standing.reaches.clone_from(reaches);
    }
    let listed = &mut obstacles.obstacles;
    listed.clear();
    listed.extend_from_slice(&standing.colliders);
    listed.extend(physics.solids());
}
