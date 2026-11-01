fn sky_average(normal: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.0);
}

fn sky_above(normal: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.0);
}

fn atmospheric(light: vec3<f32>, eye: vec3<f32>, point: vec3<f32>, direction: vec3<f32>)
        -> vec4<f32> {
    return vec4<f32>(light, 0.0);
}

fn atmospheric_cover(eye: vec3<f32>, point: vec3<f32>) -> f32 {
    return 0.0;
}

fn cloud_passing(world: vec3<f32>) -> f32 {
    return 1.0;
}

fn mirrored_along(start: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.0);
}

fn environment_levels() -> u32 {
    return 1u;
}

fn environment_size() -> u32 {
    return 1u;
}

fn environment_sample(direction: vec3<f32>, layer: u32, level: f32) -> vec3<f32> {
    return vec3<f32>(0.0);
}

fn zenith() -> vec3<f32> {
    return vec3<f32>(0.0);
}

fn horizon() -> vec3<f32> {
    return vec3<f32>(0.0);
}

fn ambient() -> f32 {
    return 0.0;
}

fn sunlight(authored: vec3<f32>) -> vec3<f32> {
    return authored;
}

fn adapted_exposure(authored: f32) -> f32 {
    return authored;
}
