import package::generated::nanite::NaniteUniforms;
import package::generated::nanite::NaniteInstance;
import package::generated::nanite::GpuClusterHeader;
import package::generated::nanite::VisibleCluster;
import package::nanite::common::PAGE_WORDS;
import package::nanite::common::quat_rotate;
import package::nanite::common::decode_pos;

@group(1) @binding(0) var<uniform> hw_uniforms: NaniteUniforms;
@group(1) @binding(1) var<storage, read> hw_instances: array<NaniteInstance>;
@group(1) @binding(2) var<storage, read> hw_clusters: array<GpuClusterHeader>;
@group(1) @binding(3) var<storage, read> hw_page_heap: array<u32>;
@group(1) @binding(4) var<storage, read> hw_draw_list: array<VisibleCluster>;

struct HwOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) vis_id: u32,
}

fn hw_vertex(vid: u32, iid: u32) -> HwOut {
    let vc = hw_draw_list[iid];
    let cmeta = hw_clusters[vc.cluster_id].cone_and_meta;
    let tri_id = vid / 3u;
    let tri_count = (cmeta.y >> 8u) & 0xFFu;
    var out: HwOut;
    if tri_id >= tri_count {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.vis_id = 0u;
        return out;
    }
    let corner = vid - tri_id * 3u;
    let vert_count = cmeta.y & 0xFFu;
    let page_base = vc.heap_slot * PAGE_WORDS + cmeta.w;
    let tri_word = hw_page_heap[page_base + vert_count * 4u + tri_id];
    let local_vi = (tri_word >> (corner * 8u)) & 0xFFu;
    let v_base = page_base + local_vi * 4u;
    let w0 = hw_page_heap[v_base];
    let w1 = hw_page_heap[v_base + 1u];
    let aabb_min = hw_clusters[vc.cluster_id].aabb_min.xyz;
    let aabb_ext = hw_clusters[vc.cluster_id].aabb_extent.xyz;
    let local_pos = decode_pos(w0, w1, aabb_min, aabb_ext);
    let pos_scale = hw_instances[vc.instance_id].pos_scale;
    let rot_quat = hw_instances[vc.instance_id].rot_quat;
    let world_pos = pos_scale.xyz + quat_rotate(rot_quat, local_pos * pos_scale.w);
    out.clip_pos = hw_uniforms.render_view_proj * vec4<f32>(world_pos, 1.0);
    out.vis_id = ((vc.vis_slot + 1u) << 6u) | tri_id;
    return out;
}

@vertex
fn vs_hw_p1(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> HwOut {
    return hw_vertex(vid, iid);
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

@fragment
fn fs_hw_p2(in: HwOut) -> @location(0) u32 {
    return in.vis_id;
}
