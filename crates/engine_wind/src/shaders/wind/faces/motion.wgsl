import package::wind::faces::wind::{slot_sway, wind_sway};

fn instance_moves(flags: u32) -> bool {
    return slot_sway(flags) > 0.0;
}

fn instance_offset(motion: vec4<f32>, pixels: f32, root: vec3<f32>, height: f32,
        local: vec3<f32>, flags: u32) -> vec3<f32> {
    return wind_sway(motion, pixels, root, height, local, slot_sway(flags));
}
