struct Shaded {
    color: vec3<f32>,
    linear: vec3<f32>,
    ambient: f32,
    glow: f32,
    reflect: f32,
    mirrored: f32,
    normal: vec3<f32>,
    roughness: f32,
    across: f32,
    body: u32,
}

struct SurfacePoint {
    albedo: vec3<f32>,
    opacity: f32,
    emission: f32,
    emissive: vec3<f32>,
    occlusion: f32,
    roughness: f32,
    alpha: f32,
    smoother_across: f32,
    f0: vec3<f32>,
    diffuse_color: vec3<f32>,
    normal: vec3<f32>,
    geometric: vec3<f32>,
    view: vec3<f32>,
    n_v: f32,
    world: vec3<f32>,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    distance: f32,
    shadow_offset: vec3<f32>,
    on_terrain: bool,
    bounce: vec3<f32>,
    sky_turn: vec3<f32>,
    sun_shade: f32,
    leaf: f32,
    crown: f32,
    body: u32,
}

const BODY_SOLID: u32 = 0u;
const BODY_THIN: u32 = 1u;
const BODY_GROUND: u32 = 2u;
const BODY_WATER: u32 = 3u;

struct ViewPixel {
    clip: vec4<f32>,
    previous: vec4<f32>,
}

fn flat(color: vec4<f32>) -> Shaded {
    var shaded: Shaded;
    shaded.color = color.rgb;
    shaded.linear = color.rgb;
    shaded.ambient = 0.0;
    shaded.glow = 0.0;
    shaded.reflect = 0.0;
    shaded.mirrored = 0.0;
    shaded.normal = vec3<f32>(0.0, 0.0, 1.0);
    shaded.roughness = 1.0;
    shaded.across = 0.0;
    shaded.body = BODY_SOLID;
    return shaded;
}
