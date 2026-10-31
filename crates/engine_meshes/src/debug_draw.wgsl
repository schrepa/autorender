import package::face::lighting::sun_debug;
import package::meshes::debug::{DEBUG_FADING, DebugOutput, DebugTargets, debug_targets};
import package::meshes::declarations::MOVED_SLOT;
import package::meshes::impostor_frames::{IMPOSTOR_FADING, impostor_index};
import package::meshes::impostors::{ImpostorInput, ImpostorOutput, impostor_card, impostor_point,
    impostor_shaded};
import package::meshes::shade::{extended_fragment, surface_point};
import package::meshes::surface::SurfacePoint;
import package::meshes::vertex::ExtendedOutput;

fn point_targets(record: u32, object: u32, detail: u32, point: SurfacePoint) -> DebugTargets {
    let sun = sun_debug(point.world + point.shadow_offset, point.geometric, point.world_dx,
        point.world_dy, point.on_terrain);
    return debug_targets(record, object, detail, point.normal, sun);
}

@fragment
fn fs_debug_mesh(input: DebugOutput, @builtin(front_facing) front: bool) -> DebugTargets {
    var extended: ExtendedOutput;
    extended.clip_position = input.clip_position;
    extended.normal_u = input.normal_u;
    extended.world_v = input.world_v;
    extended.material = input.material;
    extended.color = input.color;
    extended.uv1 = input.uv1;
    let values = extended_fragment(extended);
    let point = surface_point(values.color, values.normal, values.world, values.uv, values.uv1,
        values.surface, front);
    return point_targets(input.object.x, input.object.y, input.object.z, point);
}

struct DebugImpostorOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) frames_ab: vec4<f32>,
    @location(1) frame_c: vec2<f32>,
    @location(2) world: vec3<f32>,
    @location(3) @interpolate(flat) normal_x: vec4<f32>,
    @location(4) @interpolate(flat) normal_y: vec4<f32>,
    @location(5) @interpolate(flat) normal_z: vec3<f32>,
    @location(6) @interpolate(flat) depth_axis: vec4<f32>,
    @location(7) @interpolate(flat) material: vec4<u32>,
    @location(8) @interpolate(flat) batch: u32,
    @location(9) @interpolate(flat) object: vec2<u32>,
    @location(10) crossings: vec3<f32>,
    @location(11) @interpolate(flat) slopes_ab: vec4<f32>,
    @location(12) @interpolate(flat) slope_c: vec2<f32>,
}

@vertex
fn vs_debug_impostor(input: ImpostorInput, @location(14) draw: vec4<u32>)
        -> DebugImpostorOutput {
    let card = impostor_card(input).output;
    var output: DebugImpostorOutput;
    output.clip_position = card.clip_position;
    output.frames_ab = card.frames_ab;
    output.frame_c = card.frame_c;
    output.world = card.world;
    output.normal_x = card.normal_x;
    output.normal_y = card.normal_y;
    output.normal_z = card.normal_z;
    output.depth_axis = card.depth_axis;
    output.material = card.material;
    output.batch = card.batch;
    output.object = vec2<u32>(draw.x, input.slot & ~MOVED_SLOT);
    output.crossings = card.crossings;
    output.slopes_ab = card.slopes_ab;
    output.slope_c = card.slope_c;
    return output;
}

struct DebugImpostorTargets {
    @location(0) ids: vec4<u32>,
    @location(1) count: f32,
    @location(2) terrain: vec2<u32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_debug_impostor(input: DebugImpostorOutput) -> DebugImpostorTargets {
    var card: ImpostorOutput;
    card.clip_position = input.clip_position;
    card.frames_ab = input.frames_ab;
    card.frame_c = input.frame_c;
    card.world = input.world;
    card.normal_x = input.normal_x;
    card.normal_y = input.normal_y;
    card.normal_z = input.normal_z;
    card.depth_axis = input.depth_axis;
    card.material = input.material;
    card.batch = input.batch;
    card.crossings = input.crossings;
    card.slopes_ab = input.slopes_ab;
    card.slope_c = input.slope_c;
    let point = impostor_point(card);
    let shaded = impostor_shaded(card, point, vec3<f32>(0.0));
    let fading = select(0u, DEBUG_FADING, (input.batch & IMPOSTOR_FADING) != 0u);
    let targets = point_targets(input.object.x | fading, input.object.y,
        impostor_index(input.batch), point);
    return DebugImpostorTargets(targets.ids, targets.count, targets.terrain, shaded.depth);
}
