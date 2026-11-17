import package::face::ambient::EnvironmentLight;

fn no_environment() -> EnvironmentLight {
    var light: EnvironmentLight;
    light.near_open = 1.0;
    light.open = 1.0;
    light.inside = vec2<f32>(1.0);
    return light;
}

fn ambient_light(light: EnvironmentLight) -> vec3<f32> {
    return mix(light.sky, light.upper * light.near_open, light.near);
}

fn environment_specular(light: EnvironmentLight) -> vec3<f32> {
    if light.covered >= 0.999 {
        return light.specular;
    }
    return light.specular + (1.0 - light.covered) * light.sky_specular;
}
