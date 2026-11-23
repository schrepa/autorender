import package::generated::nanite::NaniteUniforms;
import package::generated::nanite::NaniteInstance;
import package::generated::nanite::GpuClusterHeader;
import package::generated::nanite::VisibleCluster;
import package::nanite::common::PAGE_WORDS;
import package::nanite::common::CLUSTER_HEADER_WORDS;
import package::nanite::common::MAX_PHASE_CLUSTERS;
import package::nanite::common::transform_pos;
import package::nanite::common::decode_pos_u16;

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
    let inst_id = hw_draw_list[list_idx].instance_id;
    let cluster_id = hw_draw_list[list_idx].cluster_id;
    let heap_slot = hw_draw_list[list_idx].heap_slot;
    let vis_slot = hw_draw_list[list_idx].vis_slot;
    let tri_id = vid / 3u;
    let ec_w = hw_clusters[cluster_id].errors_and_cone.w;
    let tri_count = (ec_w >> 8u) & 0xFFu;
    var out: HwOut;
    if tri_id >= tri_count {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.vis_id = 0u;
        return out;
    }
    let vert_count = ec_w & 0xFFu;
    let word_off = hw_clusters[cluster_id].page_and_fallback.y & 0xFFFFu;
    let page_base = heap_slot * PAGE_WORDS + word_off;
    let origin = vec3<f32>(
        bitcast<f32>(hw_page_heap[page_base]),
        bitcast<f32>(hw_page_heap[page_base + 1u]),
        bitcast<f32>(hw_page_heap[page_base + 2u]),
    );
    let step = bitcast<f32>(hw_page_heap[page_base + 3u]);
    let pos_base = page_base + CLUSTER_HEADER_WORDS;
    let tri_base = pos_base + ((vert_count * 3u + 1u) >> 1u);
    let tri_word = hw_page_heap[tri_base + (vid >> 2u)];
    let lane = vid & 3u;
    var local_vi = tri_word & 0xFFu;
    if lane == 1u {
        local_vi = (tri_word >> 8u) & 0xFFu;
    } else if lane == 2u {
        local_vi = (tri_word >> 16u) & 0xFFu;
    } else if lane == 3u {
        local_vi = tri_word >> 24u;
    }
    let pwi = pos_base + ((local_vi * 3u) >> 1u);
    let local_pos = decode_pos_u16(
        hw_page_heap[pwi],
        hw_page_heap[pwi + 1u],
        local_vi,
        origin,
        step,
    );
    let pos_scale = hw_instances[inst_id].pos_scale;
    let rot_quat = hw_instances[inst_id].rot_quat;
    let world_pos = transform_pos(pos_scale, rot_quat, local_pos);
    out.clip_pos = hw_uniforms.render_view_proj * vec4<f32>(world_pos, 1.0);
    out.vis_id = ((vis_slot + 1u) << 7u) | (tri_id & 0x7Fu);
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
