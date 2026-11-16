use std::cmp::Reverse;

use bevy_ecs::prelude::*;
use engine_core::collections::HashSet;
use engine_places_simulation::physics::{ColliderShape, Collision};
use engine_places_simulation::water::{WaterBody, WaterObstacle, WaterObstacles};
use glam::Vec3;

use crate::world::PhysicsWorld;

pub(crate) fn sorted<'a>(waters: impl Iterator<Item = &'a WaterBody>) -> Vec<&'a WaterBody> {
    let mut held: Vec<&WaterBody> = waters.collect();
    held.sort_by_key(|body| Reverse(body.precedence));
    held
}

pub(crate) fn obstruct(
    physics: Res<PhysicsWorld>,
    collision: Res<Collision>,
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
    let mut seen = HashSet::default();
    let mut found = Vec::new();
    for &(low, high) in &reaches {
        found.clear();
        collision.0.solids(low, high, &mut found);
        for solid in &found {
            if physics.holds(solid.entity) || !seen.insert(solid.entity) {
                continue;
            }
            listed.push(WaterObstacle {
                center: solid.center,
                axes: solid.axes,
                round: matches!(solid.shape, ColliderShape::Sphere { .. }),
            });
        }
    }
    listed.extend(physics.solids());
}
