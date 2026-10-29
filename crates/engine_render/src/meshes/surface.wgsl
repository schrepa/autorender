import package::frame::view::camera;
import package::meshes::declarations::{SceneTargets, TemporalTargets};

const MIRRORED_BITS: u32 = 14u;

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

const SPECULAR_VARIANCE: f32 = 0.15;
const SPECULAR_LIMIT: f32 = 0.2;

fn world_plane(p: vec3<f32>, axis: u32) -> vec2<f32> {
    if axis == 0u {
        return vec2<f32>(p.z, -p.y);
    }
    if axis == 2u {
        return vec2<f32>(p.x, -p.y);
    }
    return p.xz;
}

fn octahedral_coordinates(normal: vec3<f32>) -> vec2<f32> {
    let n = normal / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    let sign = select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0));
    return select(n.xy, (vec2<f32>(1.0) - abs(n.yx)) * sign, n.z < 0.0);
}

fn surface_texel(shaded: Shaded) -> vec4<u32> {
    let normal = pack2x16unorm(octahedral_coordinates(shaded.normal) * 0.5 + 0.5);
    let roughness = pack4x8unorm(vec4<f32>(shaded.roughness, shaded.across, 0.0, 0.0));
    let most = f32((1u << MIRRORED_BITS) - 1u);
    let mirrored = u32(floor(saturate(shaded.mirrored) * most + 0.5));
    return vec4<u32>(normal & 0xffffu, normal >> 16u, roughness,
        mirrored | (shaded.body << MIRRORED_BITS));
}

fn scene_targets(shaded: Shaded, color: vec3<f32>) -> SceneTargets {
    var targets: SceneTargets;
    targets.color = vec4<f32>(color, shaded.ambient);
    targets.glow = vec2<f32>(shaded.glow, shaded.reflect);
    targets.surface = surface_texel(shaded);
    return targets;
}

fn temporal_targets(shaded: Shaded, color: vec3<f32>, clip_position: vec4<f32>,
        previous: vec3<f32>) -> TemporalTargets {
    var targets: TemporalTargets;
    targets.color = vec4<f32>(color, shaded.ambient);
    targets.glow = vec2<f32>(shaded.glow, shaded.reflect);
    targets.surface = surface_texel(shaded);
    let current = vec2<f32>(clip_position.x * camera.jitter.z - 1.0,
        1.0 - clip_position.y * camera.jitter.w) - camera.jitter.xy;
    targets.motion = (previous.xy / previous.z - current) * vec2<f32>(0.5, -0.5);
    if !(previous.z > 0.0) {
        targets.motion = vec2<f32>(2.0);
    }
    return targets;
}
