import package::engine::drawn::drawn_shown;
import package::engine::shading::flat;
import package::engine::slots::flat_point;
import package::engine::view::{camera, stable_normalize, view_clip};
import package::generated::glimmer::GlimmerInstance;
import package::generated::writes::{Outputs, outputs};
@group(2) @binding(0) var<storage, read> glimmers: array<GlimmerInstance>;

const GLIMMER_UP: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
const GLIMMER_SIDE: vec3<f32> = vec3<f32>(1.0, 0.0, 0.0);
const GLIMMER_RING: f32 = 1.75;
const GLIMMER_RING_INNER: f32 = 0.8;
const GLIMMER_WHITE: u32 = 0xffffffffu;
const GLIMMER_ROUGH: u32 = 0xffu;
const GLIMMER_HIDDEN: vec4<f32> = vec4<f32>(2.0, 2.0, 2.0, 1.0);

struct GlimmerCorner {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
}

struct RingCorner { @invariant @builtin(position) clip: vec4<f32>, @location(0) unit: vec2<f32>, @location(1) color: vec4<f32> }

fn corner_unit(corner: u32) -> vec2<f32> {
    return vec2<f32>(f32((50u >> corner) & 1u), f32((44u >> corner) & 1u)) * 2.0 - 1.0;
}

fn facing_offset(glimmer: GlimmerInstance, unit: vec2<f32>, grow: f32) -> vec3<f32> {
    let toward = normalize(camera.position.xyz - glimmer.place.xyz);
    let across = cross(GLIMMER_UP, toward);
    let right = select(GLIMMER_SIDE, normalize(across), dot(across, across) > 1.0e-6);
    let up = cross(toward, right);
    return (right * unit.x + up * unit.y) * glimmer.place.w * grow;
}

fn glimmer_corner(corner: u32, glimmer: GlimmerInstance) -> GlimmerCorner {
    let unit = corner_unit(corner);
    let offset = facing_offset(glimmer, unit, 1.0);
    let world = glimmer.place.xyz + offset;
    let uv = unit * 0.5 + 0.5;
    var out: GlimmerCorner;
    out.clip_position = view_clip(world);
    out.normal_u = vec4<f32>(stable_normalize(camera.position.xyz - world), uv.x);
    out.world_v = vec4<f32>(world, uv.y);
    out.material = vec4<u32>(GLIMMER_WHITE, GLIMMER_WHITE, GLIMMER_ROUGH, 0u);
    out.color = glimmer.color;
    out.uv1 = uv;
    return out;
}

@vertex fn vs_glimmer(@builtin(vertex_index) corner: u32, @builtin(instance_index) index: u32)
        -> GlimmerCorner {
    let glimmer = glimmers[index];
    if !drawn_shown(glimmer.drawn.x) {
        var out: GlimmerCorner;
        out.clip_position = GLIMMER_HIDDEN;
        return out;
    }
    return glimmer_corner(corner, glimmer);
}

@vertex fn vs_ring(@builtin(vertex_index) corner: u32, @builtin(instance_index) index: u32) -> RingCorner {
    let glimmer = glimmers[index];
    var out: RingCorner;
    if !drawn_shown(glimmer.drawn.x) {
        out.clip = GLIMMER_HIDDEN;
        return out;
    }
    let unit = corner_unit(corner);
    let offset = facing_offset(glimmer, unit, GLIMMER_RING);
    out.clip = camera.view_projection * vec4<f32>(glimmer.place.xyz + offset, 1.0);
    out.unit = unit;
    out.color = glimmer.ring;
    return out;
}

@fragment fn fs_ring(corner: RingCorner) -> Outputs {
    let reach = length(corner.unit);
    if reach > 1.0 || reach < GLIMMER_RING_INNER { discard; }
    return outputs(flat_point(corner.color), flat(corner.color));
}
