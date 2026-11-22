use brick_nanite::NaniteWorld;
use engine::prelude::*;

pub const HERO_ASSETS: [&str; 8] = [
    "sponza_lower_w",
    "sponza_lower_e",
    "sponza_upper_w",
    "sponza_upper_e",
    "sponza_roof",
    "sponza_curtains",
    "sponza_ivy_low",
    "sponza_ivy_high",
];

pub fn populate_sanctuary(world: &mut NaniteWorld) {
    world.instances.clear();
    world.total_virtual_tris = 0;
    let count = world.meshes.len() as u32;
    for mesh_id in 0..count {
        world.add_instance(mesh_id, Vec3::ZERO, Quat::IDENTITY, 1.0);
    }
}
