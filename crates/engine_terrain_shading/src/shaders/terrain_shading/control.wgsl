import package::face::ground_surface::{SurfaceGround, surface_control, surface_inputs, surface_level,
    surface_river};
import package::generated::terrain_shading::{CONTROL_GROUP, ControlJob, RIVER_BEDS, RIVER_EDGES,
    RIVER_PLANE_FAR, TILE_STRIDE};
import package::terrain_shading::tile::{tile_height, tile_shape};

@group(1) @binding(0) var<storage, read> control_jobs: array<ControlJob>;
@group(1) @binding(1) var control: texture_storage_2d_array<r32uint, write>;
@group(1) @binding(2) var control_rivers: texture_2d_array<u32>;

fn river_plane(texel: vec2<u32>, layer: u32, plane: u32) -> u32 {
    return textureLoad(control_rivers, texel + vec2<u32>(plane * TILE_STRIDE, 0u), layer, 0).x;
}

fn river_planes(texel: vec2<u32>, job: ControlJob) -> vec4<u32> {
    var held = vec4<u32>(RIVER_PLANE_FAR, 0u, 0u, RIVER_PLANE_FAR);
    if (job.rivers & RIVER_EDGES) != 0u {
        held.x = river_plane(texel, job.layer, 0u);
    }
    if (job.rivers & RIVER_BEDS) != 0u {
        held.y = river_plane(texel, job.layer, 1u);
        held.z = river_plane(texel, job.layer, 2u);
        held.w = river_plane(texel, job.layer, 3u);
    }
    return held;
}

@compute @workgroup_size(CONTROL_GROUP, CONTROL_GROUP)
fn tile_control(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= TILE_STRIDE || id.y >= TILE_STRIDE {
        return;
    }
    let job = control_jobs[id.z];
    let spacing = job.spacing;
    let shape = tile_shape(id.x, id.y, spacing, job.layer);
    var ground: SurfaceGround;
    ground.x = f32(job.first.x + i32(id.x) - 1) * spacing;
    ground.z = f32(job.first.y + i32(id.y) - 1) * spacing;
    ground.height = tile_height(id.x, id.y, job.layer);
    ground.gradient = shape.xy;
    ground.curvature = shape.z;
    ground.spacing = spacing;
    let river = surface_river(river_planes(id.xy, job), job.rivers != 0u);
    let inputs = surface_inputs(job.terrain, ground, surface_level(job.terrain, spacing), river);
    let packed = surface_control(job.terrain, inputs);
    textureStore(control, id.xy, job.layer, vec4<u32>(packed, 0u, 0u, 0u));
}
