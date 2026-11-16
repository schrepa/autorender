fn sky_average(normal: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.0);
}

fn sky_above(normal: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.0);
}

fn environment_levels() -> u32 {
    return 1u;
}

fn environment_sample(direction: vec3<f32>, level: f32) -> vec3<f32> {
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

fn sunlight() -> vec3<f32> {
    return vec3<f32>(1.0);
}
