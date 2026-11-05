use bevy_ecs::prelude::*;
use engine_core::GlobalTransforms;
use engine_math::max;
use engine_places::ground::GroundObstacle;
use engine_places::physics::{Collider, ColliderShape, CollisionLayers, QueryFilter, SensorFilter};
use glam::{Mat3, Mat4, Vec2, Vec3};

use super::Navigation;

#[derive(SystemParam)]
pub struct ObstacleColliders<'w, 's> {
    settings: Res<'w, Navigation>,
    colliders: Query<'w, 's, (Entity, &'static Collider)>,
    globals: Res<'w, GlobalTransforms>,
}

impl ObstacleColliders<'_, '_> {
    pub(super) fn read(&self, corner: Vec2) -> Vec<GroundObstacle> {
        let filter = QueryFilter {
            layers: CollisionLayers::new(u32::MAX, self.settings.obstacle_layers),
            sensors: SensorFilter::Exclude,
            exclude: None,
        };
        self.colliders
            .iter()
            .filter(|&(entity, collider)| filter.accepts(entity, *collider))
            .filter_map(|(entity, collider)| {
                let world = self.globals.get(entity)?.matrix()?;
                Some(obstacle(collider.shape(), world, corner))
            })
            .collect()
    }
}

fn flat(direction: Vec3) -> Vec2 {
    Vec2::new(direction.x, direction.z)
}

fn obstacle(shape: ColliderShape, world: Mat4, corner: Vec2) -> GroundObstacle {
    let basis = Mat3::from_mat4(world);
    let center = flat(world.w_axis.truncate()) - corner;
    match shape {
        ColliderShape::Sphere { radius } => {
            let widest = max(
                max(basis.x_axis.length(), basis.y_axis.length()),
                basis.z_axis.length(),
            );
            GroundObstacle::Circle {
                center,
                radius: radius * widest,
            }
        }
        ColliderShape::Cuboid { half_extents } => {
            let arms = [
                flat(basis.x_axis) * half_extents.x,
                flat(basis.y_axis) * half_extents.y,
                flat(basis.z_axis) * half_extents.z,
            ];
            let longest = arms.iter().fold(Vec2::ZERO, |longest, &arm| {
                if arm.length_squared() > longest.length_squared() {
                    arm
                } else {
                    longest
                }
            });
            let unit_axis = longest.normalize_or(Vec2::X);
            let side = unit_axis.perp();
            let half_extents = arms.iter().fold(Vec2::ZERO, |sum, arm| {
                sum + Vec2::new(arm.dot(unit_axis).abs(), arm.dot(side).abs())
            });
            GroundObstacle::Box {
                center,
                half_extents,
                unit_axis,
            }
        }
    }
}
