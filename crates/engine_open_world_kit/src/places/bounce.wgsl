struct BounceLight {
    light: vec3<f32>,
    sky: f32,
    weight: f32,
}

fn bounce_light(world: vec3<f32>, normal: vec3<f32>, on_terrain: bool) -> BounceLight {
    var found: BounceLight;
    found.sky = 1.0;
    return found;
}
