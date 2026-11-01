struct EnvironmentLight {
    reflected: vec3<f32>,
    interior: vec2<f32>,
}

fn environment_light(direction: vec3<f32>, level: f32, world: vec3<f32>,
        reflected_sky: vec3<f32>) -> EnvironmentLight {
    return EnvironmentLight(reflected_sky, vec2<f32>(1.0));
}

fn in_interior(world: vec3<f32>) -> bool {
    return false;
}
