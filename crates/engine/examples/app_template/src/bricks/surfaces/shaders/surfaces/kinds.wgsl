import package::surfaces::materials::alpha_test;
import package::surfaces::vertex::output_color;
import package::surfaces::shading::{Shaded, SurfaceInput, SurfacePoint};
import package::surfaces::surface::{masked_out, opacity_cut, surface_point};
import package::engine::view::{camera, view_mip_bias};
import package::surfaces::writes::{Outputs, blended_outputs, outputs, slot_shaded};

struct SurfaceVaryings {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
}

struct SurfaceExtraVaryings {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
}

struct DepthVaryings {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
}

struct DepthExtraVaryings {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
}

struct KindFragment {
    color: vec4<f32>,
    normal: vec3<f32>,
    world: vec3<f32>,
    uv: vec2<f32>,
    uv1: vec2<f32>,
    material: vec4<u32>,
    pixel: vec2<f32>,
    kind: u32,
    object: u32,
}

struct KindShaded {
    point: SurfacePoint,
    shaded: Shaded,
}

fn surface_input(input: KindFragment, front: bool) -> SurfaceInput {
    var surface: SurfaceInput;
    surface.color = input.color;
    surface.normal = input.normal;
    surface.world = input.world;
    surface.world_dx = dpdx(input.world);
    surface.world_dy = dpdy(input.world);
    surface.uv = input.uv;
    surface.uv1 = input.uv1;
    surface.surface = unpack4x8unorm(input.material.z);
    surface.pixel = input.pixel;
    surface.front = front;
    surface.eye = camera.position.xyz;
    surface.bias = view_mip_bias();
    surface.time = camera.position.w;
    surface.kind = input.kind;
    surface.object = input.object;
    return surface;
}

fn lit_point(surface: SurfaceInput) -> SurfacePoint {
    let point = surface_point(surface);
    if alpha_test() && masked_out(surface, point) {
        discard;
    }
    return point;
}

fn fragment_values(clip_position: vec4<f32>, normal_u: vec4<f32>, world_v: vec4<f32>,
        material: vec4<u32>) -> KindFragment {
    var values: KindFragment;
    values.color = output_color(material);
    values.normal = normal_u.xyz;
    values.world = world_v.xyz;
    values.uv = vec2<f32>(normal_u.w, world_v.w);
    values.uv1 = values.uv;
    values.material = material;
    values.pixel = clip_position.xy;
    return values;
}

fn extra_values(clip_position: vec4<f32>, normal_u: vec4<f32>, world_v: vec4<f32>,
        material: vec4<u32>, color: vec4<f32>, uv1: vec2<f32>) -> KindFragment {
    var values = fragment_values(clip_position, normal_u, world_v, material);
    values.color *= color;
    values.uv1 = uv1;
    return values;
}

fn surface_fragment(input: SurfaceVaryings) -> KindFragment {
    return fragment_values(input.clip_position, input.normal_u, input.world_v, input.material);
}

fn extra_fragment(input: SurfaceExtraVaryings) -> KindFragment {
    return extra_values(input.clip_position, input.normal_u, input.world_v, input.material,
        input.color, input.uv1);
}

fn masked_depth(input: KindFragment) {
    if alpha_test() && opacity_cut(surface_input(input, true)) {
        discard;
    }
}

fn shade(input: KindFragment, front: bool) -> KindShaded {
    let surface = surface_input(input, front);
    let point = lit_point(surface);
    return KindShaded(point, slot_shaded(surface, point, input.pixel));
}

@fragment
fn fs_kind_masked_depth(input: DepthVaryings) {
    masked_depth(fragment_values(input.clip_position, input.normal_u, input.world_v,
        input.material));
}

@fragment
fn fs_kind_masked_depth_extra(input: DepthExtraVaryings) {
    masked_depth(extra_values(input.clip_position, input.normal_u, input.world_v,
        input.material, input.color, input.uv1));
}

@fragment
fn fs_kind_lit(input: SurfaceVaryings, @builtin(front_facing) front: bool) -> Outputs {
    let shaded = shade(surface_fragment(input), front);
    return outputs(shaded.point, shaded.shaded);
}

@fragment
fn fs_kind_lit_extra(input: SurfaceExtraVaryings, @builtin(front_facing) front: bool)
        -> Outputs {
    let shaded = shade(extra_fragment(input), front);
    return outputs(shaded.point, shaded.shaded);
}

@fragment
fn fs_kind_blended(input: SurfaceVaryings, @builtin(front_facing) front: bool) -> Outputs {
    let shaded = shade(surface_fragment(input), front);
    return blended_outputs(shaded.point, shaded.shaded);
}

@fragment
fn fs_kind_blended_extra(input: SurfaceExtraVaryings, @builtin(front_facing) front: bool)
        -> Outputs {
    let shaded = shade(extra_fragment(input), front);
    return blended_outputs(shaded.point, shaded.shaded);
}
