use bevy_ecs::prelude::*;
use engine_core::Camera;
use engine_places_rendering::sky::EyeMedium;
use engine_places_simulation::ground::Ground;
use engine_places_simulation::water::{WaterBody, WaterSample};
use glam::{Vec2, Vec3};

pub(crate) fn publish_medium(
    camera: Res<Camera>,
    ground: Res<Ground>,
    bodies: Query<&WaterBody>,
    mut medium: ResMut<EyeMedium>,
) {
    medium.set_if_neq(EyeMedium {
        surface: surface_under(&ground, &bodies, camera.eye()),
    });
}

fn water_at(bodies: &Query<&WaterBody>, eye: Vec3) -> Option<WaterSample> {
    let mut found: Option<(i32, WaterSample)> = None;
    for body in bodies {
        if found.is_some_and(|(precedence, _)| precedence >= body.precedence) {
            continue;
        }
        if let Some(sample) = body.shape.water(eye).ready().flatten() {
            found = Some((body.precedence, sample));
        }
    }
    found.map(|(_, sample)| sample)
}

fn surface_under(ground: &Ground, bodies: &Query<&WaterBody>, eye: Vec3) -> Option<f32> {
    let water = water_at(bodies, eye);
    if let Some(water) = water.filter(|water| water.depth > 0.0) {
        return Some(water.surface);
    }
    ground
        .point(Vec2::new(eye.x, eye.z))
        .solid()
        .map(|ground| ground.height)
}
