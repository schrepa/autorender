import package::engine::slots::LightInputs;
import package::engine::surfaces::SURFACE_SOLID;

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
    normal_dx: vec3<f32>,
    normal_dy: vec3<f32>,
    geometric: vec3<f32>,
    view: vec3<f32>,
    n_v: f32,
    world: vec3<f32>,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    distance: f32,
    kind: u32,
    thickness: f32,
    sun_blocked: f32,
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

fn flat_point(color: vec4<f32>) -> SurfacePoint {
    var point: SurfacePoint;
    point.base_color = color.rgb;
    point.opacity = color.a;
    point.normal = vec3<f32>(0.0, 0.0, 1.0);
    point.geometric = point.normal;
    point.view = point.normal;
    point.roughness = 1.0;
    point.occlusion = 1.0;
    point.kind = SURFACE_SOLID;
    return derived(point);
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
    return flat(vec4<f32>(point.base_color + point.emissive, point.opacity));
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
