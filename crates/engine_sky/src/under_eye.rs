use bevy_ecs::prelude::*;
use engine_core::Camera;
use engine_places::ground::Ground;
use engine_places::water::{WaterBody, WaterSample};
use glam::{Vec2, Vec3};

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct SurfaceUnderEye {
    pub(crate) height: Option<f32>,
}

pub(crate) fn find_surface_under_eye(
    camera: Res<Camera>,
    ground: Res<Ground>,
    bodies: Query<&WaterBody>,
    mut under: ResMut<SurfaceUnderEye>,
) {
    under.set_if_neq(SurfaceUnderEye {
        height: surface_under(&ground, &bodies, camera.position),
    });
}

fn water_at(bodies: &Query<&WaterBody>, eye: Vec3) -> Option<WaterSample> {
    let mut found: Option<(i32, WaterSample)> = None;
    for body in bodies {
        if found.is_some_and(|(precedence, _)| precedence >= body.precedence) {
            continue;
        }
        if let Some(sample) = body.shape.water(eye) {
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
        .sample(Vec2::new(eye.x, eye.z))
        .map(|ground| ground.height)
}
