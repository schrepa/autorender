import package::generated::nanite::NaniteUniforms;
import package::generated::nanite::NaniteInstance;
import package::generated::nanite::GpuClusterHeader;
import package::generated::nanite::VisibleCluster;
import package::nanite::common::PAGE_WORDS;
import package::nanite::common::CLUSTER_HEADER_WORDS;
import package::nanite::common::MAX_PHASE_CLUSTERS;
import package::nanite::common::quat_rotate;
import package::nanite::common::decode_pos;
import package::nanite::common::cluster_vert_count;
import package::nanite::common::cluster_tri_count;
import package::nanite::common::cluster_word_offset;

@group(1) @binding(0) var<uniform> hw_uniforms: NaniteUniforms;
@group(1) @binding(1) var<storage, read> hw_instances: array<NaniteInstance>;
@group(1) @binding(2) var<storage, read> hw_clusters: array<GpuClusterHeader>;
@group(1) @binding(3) var<storage, read> hw_page_heap: array<u32>;
@group(1) @binding(4) var<storage, read> hw_draw_list: array<VisibleCluster>;

struct HwOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) vis_id: u32,
}

fn hw_vertex(vid: u32, list_idx: u32) -> HwOut {
    let vc = hw_draw_list[list_idx];
    let c = hw_clusters[vc.cluster_id];
    let tri_id = vid / 3u;
    let tri_count = cluster_tri_count(c);
    var out: HwOut;
    if tri_id >= tri_count {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.vis_id = 0u;
        return out;
    }
    let corner = vid - tri_id * 3u;
    let vert_count = cluster_vert_count(c);
    let page_base = vc.heap_slot * PAGE_WORDS + cluster_word_offset(c);
    let origin = vec3<f32>(
        bitcast<f32>(hw_page_heap[page_base]),
        bitcast<f32>(hw_page_heap[page_base + 1u]),
        bitcast<f32>(hw_page_heap[page_base + 2u]),
    );
    let step = bitcast<f32>(hw_page_heap[page_base + 3u]);
    let verts_base = page_base + CLUSTER_HEADER_WORDS;
    let tri_word = hw_page_heap[verts_base + vert_count * 4u + tri_id];
    let local_vi = (tri_word >> (corner * 8u)) & 0xFFu;
    let v_base = verts_base + local_vi * 4u;
    let w0 = hw_page_heap[v_base];
    let w1 = hw_page_heap[v_base + 1u];
    let local_pos = decode_pos(w0, w1, origin, step);
    let pos_scale = hw_instances[vc.instance_id].pos_scale;
    let rot_quat = hw_instances[vc.instance_id].rot_quat;
    let world_pos = pos_scale.xyz + quat_rotate(rot_quat, local_pos * pos_scale.w);
    out.clip_pos = hw_uniforms.render_view_proj * vec4<f32>(world_pos, 1.0);
    out.vis_id = ((vc.vis_slot + 1u) << 7u) | (tri_id & 0x7Fu);
    return out;
}

@vertex
fn vs_hw_p1(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> HwOut {
    return hw_vertex(vid, iid);
}

@vertex
fn vs_hw_ds_p1(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> HwOut {
    return hw_vertex(vid, MAX_PHASE_CLUSTERS + iid);
}

@fragment
fn fs_hw_p1(in: HwOut) -> @location(0) u32 {
    return in.vis_id;
}

@vertex
fn vs_hw_p2(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> HwOut {
    return hw_vertex(vid, iid);
}

@vertex
fn vs_hw_ds_p2(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> HwOut {
    return hw_vertex(vid, MAX_PHASE_CLUSTERS + iid);
}

@fragment
fn fs_hw_p2(in: HwOut) -> @location(0) u32 {
    return in.vis_id;
}
