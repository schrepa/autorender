use bevy_ecs::prelude::*;
use engine_collision::{ColliderShape, CollisionWorld};
use engine_core::{Terrains, WaterObstacle, WaterObstacles};
use glam::Vec3;

use crate::PhysicsWorld;

#[derive(Default)]
pub(crate) struct Standing {
    revision: Option<u64>,
    reaches: Vec<(Vec3, Vec3)>,
    colliders: Vec<WaterObstacle>,
}

pub(crate) fn obstruct(
    physics: Res<PhysicsWorld>,
    collision: Option<Res<CollisionWorld>>,
    terrains: Option<Res<Terrains>>,
    mut obstacles: ResMut<WaterObstacles>,
    mut standing: Local<Standing>,
) {
    let rivers = terrains
        .iter()
        .flat_map(|terrains| terrains.waters())
        .filter_map(|(_, origin, layers)| {
            let rivers = layers.hydrology()?.rivers();
            (!rivers.is_empty()).then(|| {
                let (low, high) = rivers.bounds();
                (
                    origin + Vec3::new(low.x, -1.0e5, low.y),
                    origin + Vec3::new(high.x, 1.0e5, high.y),
                )
            })
        });
    let live = terrains
        .iter()
        .flat_map(|terrains| terrains.water_windows())
        .map(|(low, high)| {
            (
                Vec3::new(low.x, -1.0e5, low.y),
                Vec3::new(high.x, 1.0e5, high.y),
            )
        });
    let reaches: Vec<(Vec3, Vec3)> = rivers.chain(live).collect();
    if reaches.is_empty() {
        if !obstacles.obstacles.is_empty() {
            obstacles.obstacles.clear();
        }
        return;
    }
    let revision = collision
        .as_ref()
        .map(|collision| collision.report().revision);
    if revision != standing.revision || reaches != standing.reaches {
        standing.revision = revision;
        standing.colliders.clear();
        if let Some(collision) = &collision {
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
}
