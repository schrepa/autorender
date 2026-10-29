struct EnvironmentLight {
    reflected: vec3<f32>,
    interior: vec2<f32>,
}

fn environment_light(direction: vec3<f32>, level: f32, world: vec3<f32>,
        reflected_sky: vec3<f32>) -> EnvironmentLight {
    var light = vec3<f32>(0.0);
    let covered = 0.0;
    light += (1.0 - covered) * reflected_sky;
    return EnvironmentLight(light, vec2<f32>(1.0));
}

fn in_interior(world: vec3<f32>) -> bool {
    return false;
}
