struct CoverSite {
    terrain: u32,
    material: u32,
    ground: vec3<f32>,
    world: vec3<f32>,
    lush: f32,
    toward: vec3<f32>,
    normal: vec3<f32>,
    error: f32,
    footprint: f32,
}

struct CoverFar {
    color: vec3<f32>,
    roughness: f32,
    occlusion: f32,
    shown: f32,
    vegetated: f32,
    normal: vec3<f32>,
    through: f32,
    sun: f32,
}

fn cover_tint(lush: f32, vegetated: f32) -> vec3<f32> {
    return vec3<f32>(1.0);
}

fn cover_far(site: CoverSite) -> CoverFar {
    var far: CoverFar;
    far.occlusion = 1.0;
    far.sun = 1.0;
    return far;
}

fn cover_ground(site: CoverSite) -> CoverFar {
    var far: CoverFar;
    far.occlusion = 1.0;
    far.sun = 1.0;
    return far;
}
