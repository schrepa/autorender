struct Shaded {
    linear: vec3<f32>,
    opacity: f32,
}

struct PositionInput {
    local: vec3<f32>,
    world: vec3<f32>,
    origin: vec3<f32>,
    time: f32,
    pixels: f32,
    kind: u32,
    object: u32,
}

struct SurfaceInput {
    color: vec4<f32>,
    normal: vec3<f32>,
    world: vec3<f32>,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    uv: vec2<f32>,
    uv1: vec2<f32>,
    surface: vec4<f32>,
    pixel: vec2<f32>,
    front: bool,
    eye: vec3<f32>,
    bias: f32,
    time: f32,
    kind: u32,
    object: u32,
}

struct ViewPixel {
    clip: vec4<f32>,
    previous: vec4<f32>,
}

fn flat(color: vec4<f32>) -> Shaded {
    var shaded: Shaded;
    shaded.linear = color.rgb;
    shaded.opacity = color.a;
    return shaded;
}

fn unlit(point: SurfacePoint) -> Shaded {
    return flat(vec4<f32>(point.base_color + point.emissive, point.opacity));
}
