import package::generated::nanite::NaniteUniforms;
import package::generated::nanite::NaniteInstance;
import package::generated::nanite::GpuClusterHeader;
import package::generated::nanite::VisibleCluster;
import package::nanite::common::PAGE_WORDS;
import package::nanite::common::CLUSTER_HEADER_WORDS;
import package::nanite::common::MAX_PHASE_CLUSTERS;
import package::nanite::common::inst_to_world;
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

fn hw_vertex(vid: u32, list_idx: u32) -> HwOut {
    let vc = hw_draw_list[list_idx];
    let tri_id = vid / 3u;
    let ec_w = hw_clusters[vc.cluster_id].errors_and_cone.w;
    let tri_count = (ec_w >> 8u) & 0xFFu;
    if tri_id >= tri_count {
        return HwOut(vec4<f32>(2.0, 2.0, 2.0, 1.0), 0u);
    }
    let corner = vid - tri_id * 3u;
    let vert_count = ec_w & 0xFFu;
    let word_off = hw_clusters[vc.cluster_id].page_and_fallback.y & 0xFFFFu;
    let page_base = vc.heap_slot * PAGE_WORDS + word_off;
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
    let local_pos = decode_pos(hw_page_heap[v_base], hw_page_heap[v_base + 1u], origin, step);
    let inst = hw_instances[vc.instance_id];
    let world_pos = inst_to_world(inst, local_pos);
    let clip_pos = hw_uniforms.render_view_proj * vec4<f32>(world_pos, 1.0);
    let vis_id = ((vc.vis_slot + 1u) << 7u) | (tri_id & 0x7Fu);
    return HwOut(clip_pos, vis_id);
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
