use engine_core::RenderSnapshot;
use engine_core::bevy_ecs::prelude::World;
use engine_places::{Ground, WaterBodies};
use glam::Vec3;

#[derive(Clone, Copy)]
pub(crate) struct SurfaceUnderEye {
    height: Option<f32>,
}

impl SurfaceUnderEye {
    pub(crate) fn of(snapshot: &RenderSnapshot) -> Option<f32> {
        snapshot
            .extracted
            .get::<Self>()
            .and_then(|under| under.height)
    }
}

pub(crate) fn extract_surface_under_eye(world: &World, snapshot: &mut RenderSnapshot) {
    let under = SurfaceUnderEye {
        height: surface_under(world, snapshot.camera_position),
    };
    match snapshot.extracted.get_mut::<SurfaceUnderEye>() {
        Some(held) => *held = under,
        None => snapshot.extracted.insert(under),
    }
}

fn surface_under(world: &World, eye: Vec3) -> Option<f32> {
    let water = WaterBodies::water_in(world, eye);
    if let Some(water) = water.filter(|water| water.depth > 0.0) {
        return Some(water.surface);
    }
    let ground = Ground::sample_in(world, eye.x, eye.z);
    ground.map(|ground| ground.height)
}
