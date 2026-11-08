import package::engine::drawn::{drawn_flags_own_projection, drawn_in_view, drawn_records};
import package::engine::mesh::declarations::drawn_slot;
import package::engine::view::camera;
import package::generated::meshes::{BLEND_GROUP, BatchData as Batch, BlendedView};
import package::meshes::slots::instance_poses;

struct SortArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(1) @binding(0) var<uniform> sort_view: BlendedView;
@group(1) @binding(1) var<storage, read> sort_rows: array<vec4<u32>>;
@group(1) @binding(2) var<storage, read> sort_batches: array<Batch>;
@group(1) @binding(3) var<storage, read_write> sort_keys: array<f32>;
@group(1) @binding(4) var<storage, read_write> sort_ids: array<u32>;
@group(1) @binding(5) var<storage, read_write> sort_args: array<SortArgs>;

const UNSEEN: f32 = 3.0e38;

fn outside(center: vec3<f32>, radius: f32) -> bool {
    let m = camera.view_projection;
    let x = vec4<f32>(m[0].x, m[1].x, m[2].x, m[3].x);
    let y = vec4<f32>(m[0].y, m[1].y, m[2].y, m[3].y);
    let z = vec4<f32>(m[0].z, m[1].z, m[2].z, m[3].z);
    let w = vec4<f32>(m[0].w, m[1].w, m[2].w, m[3].w);
    var planes = array<vec4<f32>, 5>(w + x, w - x, w + y, w - y, w - z);
    for (var at = 0u; at < 5u; at++) {
        let plane = planes[at];
        if dot(plane.xyz, center) + plane.w < -radius * length(plane.xyz) {
            return true;
        }
    }
    return false;
}

fn row_depth(row: vec4<u32>) -> f32 {
    let slot = row.x;
    if !drawn_in_view(sort_view.view, slot) {
        return UNSEEN;
    }
    let pose = instance_poses[slot];
    let batch = sort_batches[row.y];
    let local = vec4<f32>((batch.bounds_min.xyz + batch.bounds_max.xyz) * 0.5, 1.0);
    let center = vec3<f32>(dot(pose.row_0, local), dot(pose.row_1, local),
        dot(pose.row_2, local));
    let scale = max(length(vec3<f32>(pose.row_0.x, pose.row_1.x, pose.row_2.x)),
        max(length(vec3<f32>(pose.row_0.y, pose.row_1.y, pose.row_2.y)),
            length(vec3<f32>(pose.row_0.z, pose.row_1.z, pose.row_2.z))));
    let radius = length(batch.bounds_max.xyz - batch.bounds_min.xyz) * 0.5 * scale;
    let own = drawn_flags_own_projection(drawn_records[slot].flags);
    if !own && outside(center, radius) {
        return UNSEEN;
    }
    let clip = camera.view_projection * vec4<f32>(center, 1.0);
    return clip.z / max(clip.w, 1.0e-6);
}

@compute @workgroup_size(BLEND_GROUP)
fn blended_keys(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= sort_view.rows {
        return;
    }
    sort_keys[id.x] = row_depth(sort_rows[id.x]);
}

@compute @workgroup_size(BLEND_GROUP)
fn blended_rank(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= sort_view.rows {
        return;
    }
    let row = sort_rows[id.x];
    let batch = sort_batches[row.y];
    let base = row.z;
    let own = sort_keys[id.x];
    var rank = 0u;
    var seen = 0u;
    for (var at = base; at < base + batch.count; at++) {
        let other = sort_keys[at];
        if other < own || (other == own && at < id.x) {
            rank += 1u;
        }
        if other < UNSEEN {
            seen += 1u;
        }
    }
    sort_ids[base + rank] = drawn_slot(row.x, false);
    if id.x == base {
        sort_args[row.w] = SortArgs(batch.index_count, seen, batch.first_index,
            batch.base_vertex, 0u);
    }
}
