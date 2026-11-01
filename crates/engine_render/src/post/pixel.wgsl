struct PostPixel {
    position: vec2<f32>,
    uv: vec2<f32>,
    pixel: vec2<i32>,
    turn: f32,
    salt: u32,
}

struct SceneLight {
    color: vec3<f32>,
    ambient: f32,
}

fn ambient_blocked(light: SceneLight, visible: f32) -> SceneLight {
    let kept = 1.0 - light.ambient * (1.0 - visible);
    return SceneLight(light.color * kept, light.ambient * visible / max(kept, 1.0e-6));
}
