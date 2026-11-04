use std::cmp::Reverse;

use bevy_ecs::prelude::*;
use engine_core::collections::HashSet;
use engine_places::physics::ColliderShape;
use engine_places::water::{WaterBody, WaterObstacle, WaterObstacles};
use glam::Vec3;

use crate::solids::Solids;
use crate::world::PhysicsWorld;

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
) {
    let reaches: Vec<(Vec3, Vec3)> = sorted(waters.iter())
        .into_iter()
        .flat_map(|body| body.shape.reaches())
        .collect();
    if reaches.is_empty() {
        if !obstacles.obstacles.is_empty() {
            obstacles.obstacles.clear();
        }
        return;
    }
    let listed = &mut obstacles.obstacles;
    listed.clear();
    if solids.collides() {
        let mut seen = HashSet::default();
        for &(low, high) in &reaches {
            for found in solids.colliders_in(low, high) {
                if physics.holds(found.entity) || !seen.insert(found.entity) {
                    continue;
                }
                listed.push(WaterObstacle {
                    center: found.solid.center,
                    axes: found.solid.axes,
                    round: matches!(found.solid.shape, ColliderShape::Sphere { .. }),
                    velocity: Vec3::ZERO,
                });
            }
        }
    }
    listed.extend(physics.solids());
}
