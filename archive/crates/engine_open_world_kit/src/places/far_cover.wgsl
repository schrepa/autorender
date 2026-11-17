struct GroundAt {
    world: vec3<f32>,
    normal: vec3<f32>,
    toward: vec3<f32>,
    across: f32,
    footprint: f32,
    lush: f32,
    vegetated: f32,
}

struct FarGround {
    color: vec3<f32>,
    roughness: f32,
    normal: vec3<f32>,
    occlusion: f32,
    lift: f32,
    shown: f32,
    sun: f32,
}

fn far_cover(ground: FarGround, at: GroundAt) -> FarGround {
    return ground;
}
