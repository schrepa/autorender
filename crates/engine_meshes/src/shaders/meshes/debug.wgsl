import package::engine::shading::SurfacePoint;
import package::face::debug::{DebugTargets, debug_targets};
import package::face::lighting::{sun_debug, surface_sun};
import package::meshes::shade::{extended_fragment, surface_point};
import package::meshes::vertex::{ExtendedMeshOutput, ExtendedOutput, MeshOutput};

fn point_targets(record: u32, object: u32, detail: u32, point: SurfacePoint) -> DebugTargets {
    let sun = sun_debug(surface_sun(point));
    return debug_targets(record, object, detail, point.normal, sun);
}

fn debug_mesh(extended: ExtendedOutput, object: vec4<u32>, front: bool, tested: bool)
        -> DebugTargets {
    let point = surface_point(extended_fragment(extended), front, tested);
    return point_targets(object.x, object.y, object.z, point);
}

fn plain_output(input: MeshOutput) -> ExtendedOutput {
    var output: ExtendedOutput;
    output.clip_position = input.clip_position;
    output.normal_u = input.normal_u;
    output.world_v = input.world_v;
    output.material = input.material;
    output.color = vec4<f32>(1.0);
    output.uv1 = vec2<f32>(input.normal_u.w, input.world_v.w);
    return output;
}

fn extended_output(input: ExtendedMeshOutput) -> ExtendedOutput {
    var output: ExtendedOutput;
    output.clip_position = input.clip_position;
    output.normal_u = input.normal_u;
    output.world_v = input.world_v;
    output.material = input.material;
    output.color = input.color;
    output.uv1 = input.uv1;
    return output;
}

@fragment
fn fs_debug_mesh(input: MeshOutput, @builtin(front_facing) front: bool) -> DebugTargets {
    return debug_mesh(plain_output(input), input.object, front, false);
}

@fragment
fn fs_debug_mesh_pretested(input: MeshOutput, @builtin(front_facing) front: bool)
        -> DebugTargets {
    return debug_mesh(plain_output(input), input.object, front, true);
}

@fragment
fn fs_debug_mesh_extended(input: ExtendedMeshOutput, @builtin(front_facing) front: bool)
        -> DebugTargets {
    return debug_mesh(extended_output(input), input.object, front, false);
}

@fragment
fn fs_debug_mesh_extended_pretested(input: ExtendedMeshOutput,
        @builtin(front_facing) front: bool) -> DebugTargets {
    return debug_mesh(extended_output(input), input.object, front, true);
}
