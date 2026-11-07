import package::engine::surfaces::SURFACE_SOLID;

struct Shaded {
    linear: vec3<f32>,
    ambient: f32,
    reflect: f32,
    mirrored: f32,
    normal: vec3<f32>,
    roughness: f32,
    across: f32,
    kind: u32,
    opacity: f32,
}

struct PositionInput {
    local: vec3<f32>,
    world: vec3<f32>,
    origin: vec3<f32>,
    time: f32,
    pixels: f32,
    object: vec4<u32>,
}

struct LightInputs {
    shadow_shift: vec3<f32>,
    ignore_face: bool,
    bounce: vec3<f32>,
    sky_bend: vec3<f32>,
    sun_blocked: f32,
    smoother_across: f32,
    shadow_softness: f32,
    shadow_lift_first: u32,
    shadow_lifts: vec4<f32>,
}

struct SurfacePoint {
    base_color: vec3<f32>,
    opacity: f32,
    emissive: vec3<f32>,
    occlusion: f32,
    roughness: f32,
    alpha: f32,
    metalness: f32,
    transmission: f32,
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
    kind: u32,
    inputs: LightInputs,
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
    object: vec4<u32>,
    words: vec4<u32>,
    origin: vec3<f32>,
}

const SPECULAR_VARIANCE: f32 = 0.15;
const SPECULAR_LIMIT: f32 = 0.2;

struct ViewPixel {
    clip: vec4<f32>,
    previous: vec4<f32>,
}

fn flat(color: vec4<f32>) -> Shaded {
    var shaded: Shaded;
    shaded.linear = color.rgb;
    shaded.ambient = 0.0;
    shaded.reflect = 0.0;
    shaded.mirrored = 0.0;
    shaded.normal = vec3<f32>(0.0, 0.0, 1.0);
    shaded.roughness = 1.0;
    shaded.across = 0.0;
    shaded.kind = SURFACE_SOLID;
    shaded.opacity = color.a;
    return shaded;
}

fn derived(point: SurfacePoint) -> SurfacePoint {
    var result = point;
    result.roughness = max(point.roughness, 0.03);
    result.alpha = result.roughness * result.roughness;
    result.f0 = mix(vec3<f32>(0.04), point.base_color, point.metalness);
    result.diffuse_color = point.base_color * (1.0 - point.metalness);
    result.n_v = max(dot(point.normal, point.view), 1.0e-4);
    return result;
}

fn unlit(point: SurfacePoint) -> Shaded {
    var shaded = flat(vec4<f32>(point.base_color + point.emissive, point.opacity));
    shaded.normal = point.normal;
    shaded.roughness = point.roughness;
    shaded.kind = point.kind;
    return shaded;
}

fn world_plane(p: vec3<f32>, axis: u32) -> vec2<f32> {
    if axis == 0u {
        return vec2<f32>(p.z, -p.y);
    }
    if axis == 2u {
        return vec2<f32>(p.x, -p.y);
    }
    return p.xz;
}
