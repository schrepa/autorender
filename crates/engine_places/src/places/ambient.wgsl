struct EnvironmentAt {
    world: vec3<f32>,
    normal: vec3<f32>,
    skyward: vec3<f32>,
    lobe: vec3<f32>,
    roughness: f32,
    alpha: f32,
    ground: bool,
}

struct EnvironmentLight {
    sky: vec3<f32>,
    upper: vec3<f32>,
    horizon: vec3<f32>,
    near: f32,
    near_open: f32,
    bounce: vec3<f32>,
    sky_specular: vec3<f32>,
    specular: vec3<f32>,
    covered: f32,
    open: f32,
    inside: vec2<f32>,
}

fn environment(light: EnvironmentLight, at: EnvironmentAt) -> EnvironmentLight {
    return light;
}
