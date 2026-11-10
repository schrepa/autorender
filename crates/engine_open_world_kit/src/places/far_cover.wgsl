struct GroundAt {
    world: vec3<f32>,
    lush: f32,
    vegetated: f32,
    terrain: u32,
    material: u32,
    across: f32,
}

fn far_color(color: vec3<f32>, at: GroundAt) -> vec3<f32> {
    return color;
}
