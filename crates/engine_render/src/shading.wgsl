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

struct LightInputs {
    ignore_face: bool,
    added_light: vec3<f32>,
    added_indirect: vec3<f32>,
    ambient_bend: vec3<f32>,
    smoother_across: f32,
    direct_softness: f32,
    receiver_lift_first: u32,
    receiver_lifts: vec4<f32>,
    thickness: f32,
    direct_blocked: f32,
    direct_known: bool,
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
