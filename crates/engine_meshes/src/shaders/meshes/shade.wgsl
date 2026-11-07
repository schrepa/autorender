import package::engine::materials::{alpha_test, fade_list};
import package::engine::mesh::declarations::{MotionOutput, VertexOutput, fade_hidden};
import package::engine::mesh::vertex::output_color;
import package::engine::shading::{SurfaceInput, SurfacePoint};
import package::engine::slots::{shade_model, slot_output};
import package::engine::surface::{depth_surface, masked_out, opacity_cut, surface_point};
import package::engine::view::{camera, material_bias};
import package::meshes::vertex::{ExtendedOutput, ExtendedMotionOutput};
import package::place::instances::{SlotFragment, SlotShaded};

fn surface_input(input: SlotFragment, front: bool) -> SurfaceInput {
    var surface: SurfaceInput;
    surface.color = input.color;
    surface.normal = input.normal;
    surface.world = input.world;
    surface.world_dx = dpdx(input.world);
    surface.world_dy = dpdy(input.world);
    surface.uv = input.uv;
    surface.uv1 = input.uv1;
    surface.surface = unpack4x8unorm(input.object.z);
    surface.pixel = input.pixel;
    surface.front = front;
    surface.eye = camera.position.xyz;
    surface.bias = material_bias();
    surface.time = camera.position.w;
    surface.object = input.object;
    surface.words = input.words;
    surface.origin = input.origin;
    return surface;
}

fn lit_point(surface: SurfaceInput, tested: bool) -> SurfacePoint {
    let point = surface_point(surface);
    if alpha_test() && !tested && masked_out(surface, point) {
        discard;
    }
    return point;
}

fn mesh_point(input: SlotFragment, front: bool, tested: bool) -> SurfacePoint {
    return lit_point(surface_input(input, front), tested);
}

fn fragment_values(clip_position: vec4<f32>, normal_u: vec4<f32>, world_v: vec4<f32>,
        material: vec4<u32>, words: vec4<u32>, origin: vec3<f32>) -> SlotFragment {
    var values: SlotFragment;
    values.color = output_color(material);
    values.normal = normal_u.xyz;
    values.world = world_v.xyz;
    values.uv = vec2<f32>(normal_u.w, world_v.w);
    values.uv1 = values.uv;
    values.object = material;
    values.pixel = clip_position.xy;
    values.words = words;
    values.origin = origin;
    return values;
}

fn fragment(input: VertexOutput) -> SlotFragment {
    if fade_list() && fade_hidden(input.material.w, input.clip_position.xy) {
        discard;
    }
    return fragment_values(input.clip_position, input.normal_u, input.world_v, input.material,
        vec4<u32>(0u), vec3<f32>(0.0));
}

fn extended_fragment(input: ExtendedOutput) -> SlotFragment {
    if fade_list() && fade_hidden(input.material.w, input.clip_position.xy) {
        discard;
    }
    var values = fragment_values(input.clip_position, input.normal_u, input.world_v,
        input.material, vec4<u32>(0u), vec3<f32>(0.0));
    values.color *= input.color;
    values.uv1 = input.uv1;
    return values;
}

fn placed_fragment(clip_position: vec4<f32>, normal_u: vec4<f32>, world_v: vec4<f32>,
        material: vec4<u32>, words: vec4<u32>, origin: vec3<f32>) -> SlotFragment {
    if fade_hidden(material.w, clip_position.xy) {
        discard;
    }
    return fragment_values(clip_position, normal_u, world_v, material, words, origin);
}

fn masked_depth(input: SlotFragment) {
    if opacity_cut(surface_input(input, true)) {
        discard;
    }
}

fn depth_point(input: SlotFragment, front: bool) -> SurfacePoint {
    let surface = surface_input(input, front);
    let point = depth_surface(surface);
    if masked_out(surface, point) {
        discard;
    }
    return point;
}

fn shade(input: SlotFragment, front: bool, tested: bool) -> SlotShaded {
    let surface = surface_input(input, front);
    let point = lit_point(surface, tested);
    return SlotShaded(point, slot_output(surface, point, shade_model(point, input.pixel)));
}

fn motion_output(input: MotionOutput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = input.clip_position;
    output.normal_u = input.normal_u;
    output.world_v = input.world_v;
    output.material = input.material;
    return output;
}

fn extended_motion_output(input: ExtendedMotionOutput) -> ExtendedOutput {
    var output: ExtendedOutput;
    output.clip_position = input.clip_position;
    output.normal_u = input.normal_u;
    output.world_v = input.world_v;
    output.material = input.material;
    output.color = input.color;
    output.uv1 = input.uv1;
    return output;
}
