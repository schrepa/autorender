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
