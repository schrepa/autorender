import package::generated::nanite::NaniteUniforms;
import package::generated::nanite::NaniteInstance;
import package::generated::nanite::GpuClusterHeader;
import package::generated::nanite::VisibleCluster;
import package::nanite::common::PAGE_WORDS;
import package::nanite::common::MAX_PHASE_CLUSTERS;
import package::nanite::common::SW_DISPATCH_STRIDE;
import package::nanite::common::inst_to_world;
import package::nanite::common::decode_pos;
import package::nanite::common::cluster_vert_count;
import package::nanite::common::cluster_tri_count;
import package::nanite::common::edge_yup_i32;
import package::nanite::common::is_top_left_i32;

@group(1) @binding(0) var<uniform> sw_uniforms: NaniteUniforms;
@group(1) @binding(1) var<storage, read> sw_instances: array<NaniteInstance>;
@group(1) @binding(2) var<storage, read> sw_clusters: array<GpuClusterHeader>;
@group(1) @binding(3) var<storage, read> sw_page_heap: array<u32>;
@group(1) @binding(4) var<storage, read> sw_draw_list: array<VisibleCluster>;
@group(1) @binding(5) var sw_hw_depth_r32: texture_2d<f32>;
@group(1) @binding(6) var sw_hw_depth_tex: texture_depth_2d;
@group(1) @binding(7) var<storage, read_write> sw_depth_rw: array<atomic<u32>>;
@group(1) @binding(8) var<storage, read_write> sw_vis_rw: array<u32>;

var<workgroup> sh_screen: array<vec4<f32>, 64>;
var<workgroup> sh_valid: array<u32, 64>;

struct TriSetup {
    valid: u32,
    e0_start: i32,
    e1_start: i32,
    e2_start: i32,
    dx: vec3<i32>,
    dy: vec3<i32>,
    z_start: f32,
    dz_dx: f32,
    dz_dy: f32,
    bbox: vec4<i32>,
    vis_id: u32,
}

@compute @workgroup_size(8, 8, 1)
fn cs_seed_sw_depth_p1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(sw_uniforms.viewport_lod.x);
    let h = u32(sw_uniforms.viewport_lod.y);
    if gid.x >= w || gid.y >= h {
        return;
    }
    let idx = gid.y * w + gid.x;
    let d = textureLoad(sw_hw_depth_r32, vec2<i32>(i32(gid.x), i32(gid.y)), 0).x;
    atomicStore(&sw_depth_rw[idx], select(0u, bitcast<u32>(d), d > 0.0));
    sw_vis_rw[idx] = 0u;
}

@compute @workgroup_size(8, 8, 1)
fn cs_merge_hw_depth_p2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(sw_uniforms.viewport_lod.x);
    let h = u32(sw_uniforms.viewport_lod.y);
    if gid.x >= w || gid.y >= h {
        return;
    }
    let idx = gid.y * w + gid.x;
    let d = textureLoad(sw_hw_depth_tex, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let zb = select(0u, bitcast<u32>(d), d > 0.0);
    if zb > atomicLoad(&sw_depth_rw[idx]) {
        atomicStore(&sw_depth_rw[idx], zb);
        sw_vis_rw[idx] = 0u;
    }
}

fn load_cluster_verts(vc: VisibleCluster, c: GpuClusterHeader, lid: u32) {
    let vert_count = cluster_vert_count(c);
    if lid < vert_count {
        let page_base = vc.heap_slot * PAGE_WORDS + c.cone_and_meta.w;
        let v_base = page_base + lid * 4u;
        let w0 = sw_page_heap[v_base];
        let w1 = sw_page_heap[v_base + 1u];
        let local_pos = decode_pos(w0, w1, c.aabb_min.xyz, c.aabb_extent.xyz);
        let inst = sw_instances[vc.instance_id];
        let world_pos = inst_to_world(inst, local_pos);
        let clip = sw_uniforms.render_view_proj * vec4<f32>(world_pos, 1.0);
        if clip.w > 1.0e-5 {
            let inv_w = 1.0 / clip.w;
            let ndc = clip.xyz * inv_w;
            let vp = sw_uniforms.viewport_lod.xy;
            let sx = (ndc.x * 0.5 + 0.5) * vp.x;
            let sy = (0.5 - ndc.y * 0.5) * vp.y;
            sh_screen[lid] = vec4<f32>(sx, sy, ndc.z, inv_w);
            sh_valid[lid] = 1u;
        } else {
            sh_screen[lid] = vec4<f32>(0.0);
            sh_valid[lid] = 0u;
        }
    } else {
        sh_valid[lid] = 0u;
    }
}

fn setup_tri(vc: VisibleCluster, c: GpuClusterHeader, lid: u32) -> TriSetup {
    var s: TriSetup;
    s.valid = 0u;
    let tri_count = cluster_tri_count(c);
    if lid >= tri_count {
        return s;
    }
    let vert_count = cluster_vert_count(c);
    let page_base = vc.heap_slot * PAGE_WORDS + c.cone_and_meta.w;
    let tri_word = sw_page_heap[page_base + vert_count * 4u + lid];
    let i0 = tri_word & 0xFFu;
    let i1 = (tri_word >> 8u) & 0xFFu;
    let i2 = (tri_word >> 16u) & 0xFFu;
    if (sh_valid[i0] & sh_valid[i1] & sh_valid[i2]) == 0u {
        return s;
    }
    let p0 = sh_screen[i0];
    let p1 = sh_screen[i1];
    let p2 = sh_screen[i2];
    let raw_min_x = min(p0.x, min(p1.x, p2.x));
    let raw_max_x = max(p0.x, max(p1.x, p2.x));
    let raw_min_y = min(p0.y, min(p1.y, p2.y));
    let raw_max_y = max(p0.y, max(p1.y, p2.y));
    if (raw_max_x - raw_min_x) > 96.0 || (raw_max_y - raw_min_y) > 96.0 {
        return s;
    }
    let w_i = i32(sw_uniforms.viewport_lod.x);
    let h_i = i32(sw_uniforms.viewport_lod.y);
    if w_i <= 0 || h_i <= 0 {
        return s;
    }
    let min_x = max(i32(floor(raw_min_x)), 0);
    let max_x = min(i32(floor(raw_max_x)), w_i - 1);
    let min_y = max(i32(floor(raw_min_y)), 0);
    let max_y = min(i32(floor(raw_max_y)), h_i - 1);
    if min_x > max_x || min_y > max_y {
        return s;
    }
    let q0 = vec2<i32>(round(p0.xy * 16.0));
    let q1 = vec2<i32>(round(p1.xy * 16.0));
    let q2 = vec2<i32>(round(p2.xy * 16.0));
    let area = edge_yup_i32(q0, q1, q2);
    if area <= 0 {
        return s;
    }
    let b0 = select(-1, 0, is_top_left_i32(q1, q2));
    let b1 = select(-1, 0, is_top_left_i32(q2, q0));
    let b2 = select(-1, 0, is_top_left_i32(q0, q1));
    let qp0 = vec2<i32>(min_x * 16 + 8, min_y * 16 + 8);
    let e0_0 = edge_yup_i32(q1, q2, qp0);
    let e1_0 = edge_yup_i32(q2, q0, qp0);
    let e2_0 = edge_yup_i32(q0, q1, qp0);
    let e0_s = e0_0 + b0;
    let e1_s = e1_0 + b1;
    let e2_s = e2_0 + b2;
    if min_x == max_x && min_y == max_y {
        if (e0_s | e1_s | e2_s) < 0 {
            return s;
        }
        let z0 = (f32(e0_0) * p0.z + f32(e1_0) * p1.z + f32(e2_0) * p2.z) / f32(area);
        if z0 <= 0.0 || z0 > 1.0 {
            return s;
        }
        s.valid = 2u;
        s.z_start = z0;
        s.bbox = vec4<i32>(min_x, min_y, max_x, max_y);
        s.vis_id = ((vc.vis_slot + 1u) << 6u) | lid;
        return s;
    }
    let dx = vec3<i32>(
        (q2.y - q1.y) * 16,
        (q0.y - q2.y) * 16,
        (q1.y - q0.y) * 16,
    );
    let dy = vec3<i32>(
        (q1.x - q2.x) * 16,
        (q2.x - q0.x) * 16,
        (q0.x - q1.x) * 16,
    );
    let inv_area = 1.0 / f32(area);
    s.valid = 1u;
    s.e0_start = e0_s;
    s.e1_start = e1_s;
    s.e2_start = e2_s;
    s.dx = dx;
    s.dy = dy;
    s.z_start = (f32(e0_0) * p0.z + f32(e1_0) * p1.z + f32(e2_0) * p2.z) * inv_area;
    s.dz_dx = (f32(dx.x) * p0.z + f32(dx.y) * p1.z + f32(dx.z) * p2.z) * inv_area;
    s.dz_dy = (f32(dy.x) * p0.z + f32(dy.y) * p1.z + f32(dy.z) * p2.z) * inv_area;
    s.bbox = vec4<i32>(min_x, min_y, max_x, max_y);
    s.vis_id = ((vc.vis_slot + 1u) << 6u) | lid;
    return s;
}

@compute @workgroup_size(64)
fn cs_sw_depth(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let sw_idx = wid.y * SW_DISPATCH_STRIDE + wid.x;
    let sw_total = sw_draw_list[MAX_PHASE_CLUSTERS * 2u].cluster_id;
    if sw_idx >= sw_total {
        return;
    }
    let vc = sw_draw_list[MAX_PHASE_CLUSTERS + sw_idx];
    let c = sw_clusters[vc.cluster_id];
    load_cluster_verts(vc, c, lid);
    workgroupBarrier();
    let s = setup_tri(vc, c, lid);
    if s.valid == 0u {
        return;
    }
    let stride = u32(sw_uniforms.viewport_lod.x);
    if s.valid == 2u {
        let zb = bitcast<u32>(s.z_start);
        let idx = u32(s.bbox.y) * stride + u32(s.bbox.x);
        if zb > atomicLoad(&sw_depth_rw[idx]) {
            atomicMax(&sw_depth_rw[idx], zb);
        }
        return;
    }
    var e0_row = s.e0_start;
    var e1_row = s.e1_start;
    var e2_row = s.e2_start;
    var z_row = s.z_start;
    for (var y = s.bbox.y; y <= s.bbox.w; y = y + 1) {
        let row = u32(y) * stride;
        var e0 = e0_row;
        var e1 = e1_row;
        var e2 = e2_row;
        var z = z_row;
        for (var x = s.bbox.x; x <= s.bbox.z; x = x + 1) {
            if (e0 | e1 | e2) >= 0 && z > 0.0 && z <= 1.0 {
                let zb = bitcast<u32>(z);
                let idx = row + u32(x);
                if zb > atomicLoad(&sw_depth_rw[idx]) {
                    atomicMax(&sw_depth_rw[idx], zb);
                }
            }
            e0 += s.dx.x;
            e1 += s.dx.y;
            e2 += s.dx.z;
            z += s.dz_dx;
        }
        e0_row += s.dy.x;
        e1_row += s.dy.y;
        e2_row += s.dy.z;
        z_row += s.dz_dy;
    }
}

@compute @workgroup_size(64)
fn cs_sw_vis(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let sw_idx = wid.y * SW_DISPATCH_STRIDE + wid.x;
    let sw_total = sw_draw_list[MAX_PHASE_CLUSTERS * 2u].cluster_id;
    if sw_idx >= sw_total {
        return;
    }
    let vc = sw_draw_list[MAX_PHASE_CLUSTERS + sw_idx];
    let c = sw_clusters[vc.cluster_id];
    load_cluster_verts(vc, c, lid);
    workgroupBarrier();
    let s = setup_tri(vc, c, lid);
    if s.valid == 0u {
        return;
    }
    let stride = u32(sw_uniforms.viewport_lod.x);
    if s.valid == 2u {
        let zb = bitcast<u32>(s.z_start);
        let idx = u32(s.bbox.y) * stride + u32(s.bbox.x);
        if zb == atomicLoad(&sw_depth_rw[idx]) {
            sw_vis_rw[idx] = s.vis_id;
        }
        return;
    }
    var e0_row = s.e0_start;
    var e1_row = s.e1_start;
    var e2_row = s.e2_start;
    var z_row = s.z_start;
    for (var y = s.bbox.y; y <= s.bbox.w; y = y + 1) {
        let row = u32(y) * stride;
        var e0 = e0_row;
        var e1 = e1_row;
        var e2 = e2_row;
        var z = z_row;
        for (var x = s.bbox.x; x <= s.bbox.z; x = x + 1) {
            if (e0 | e1 | e2) >= 0 && z > 0.0 && z <= 1.0 {
                let zb = bitcast<u32>(z);
                let idx = row + u32(x);
                if zb == atomicLoad(&sw_depth_rw[idx]) {
                    sw_vis_rw[idx] = s.vis_id;
                }
            }
            e0 += s.dx.x;
            e1 += s.dx.y;
            e2 += s.dx.z;
            z += s.dz_dx;
        }
        e0_row += s.dy.x;
        e1_row += s.dy.y;
        e2_row += s.dy.z;
        z_row += s.dz_dy;
    }
}
