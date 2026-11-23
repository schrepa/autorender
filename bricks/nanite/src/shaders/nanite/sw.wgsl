import package::generated::nanite::NaniteUniforms;
import package::generated::nanite::NaniteInstance;
import package::generated::nanite::GpuClusterHeader;
import package::generated::nanite::VisibleCluster;
import package::nanite::common::PAGE_WORDS;
import package::nanite::common::CLUSTER_HEADER_WORDS;
import package::nanite::common::MAX_PHASE_CLUSTERS;
import package::nanite::common::SW_DISPATCH_STRIDE;
import package::nanite::common::transform_pos;
import package::nanite::common::decode_pos_u16;
import package::nanite::common::edge_yup_i32;
import package::nanite::common::is_top_left_i32;

@group(1) @binding(0) var<uniform> sw_uniforms: NaniteUniforms;
@group(1) @binding(1) var<storage, read> sw_instances: array<NaniteInstance>;
@group(1) @binding(2) var<storage, read> sw_clusters: array<GpuClusterHeader>;
@group(1) @binding(3) var<storage, read> sw_page_heap: array<u32>;
@group(1) @binding(4) var<storage, read> sw_draw_list: array<VisibleCluster>;
@group(1) @binding(5) var sw_hw_depth_tex: texture_depth_2d;
@group(1) @binding(6) var<storage, read_write> sw_depth_rw: array<atomic<u32>>;
@group(1) @binding(7) var<storage, read_write> sw_vis_rw: array<u32>;

var<workgroup> sh_screen: array<vec4<f32>, 128>;
var<workgroup> sh_valid: array<u32, 128>;

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
    let d = textureLoad(sw_hw_depth_tex, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
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

fn read_u24(base_word: u32, idx: u32) -> u32 {
    let byte_off = idx * 3u;
    let wi = base_word + (byte_off >> 2u);
    let rem = byte_off & 3u;
    let w0 = sw_page_heap[wi];
    var res = w0 & 0xFFFFFFu;
    if rem == 1u {
        res = w0 >> 8u;
    } else if rem == 2u {
        let w1 = sw_page_heap[wi + 1u];
        res = (w0 >> 16u) | ((w1 & 0xFFu) << 16u);
    } else if rem == 3u {
        let w1 = sw_page_heap[wi + 1u];
        res = (w0 >> 24u) | ((w1 & 0xFFFFu) << 8u);
    }
    return res;
}

fn load_cluster_vert(
    inst_id: u32,
    page_base: u32,
    vert_count: u32,
    vi: u32,
    origin: vec3<f32>,
    step: f32,
) {
    if vi < vert_count {
        let pos_base = page_base + CLUSTER_HEADER_WORDS;
        let pwi = pos_base + ((vi * 3u) >> 1u);
        let local_pos = decode_pos_u16(
            sw_page_heap[pwi],
            sw_page_heap[pwi + 1u],
            vi,
            origin,
            step,
        );
        let pos_scale = sw_instances[inst_id].pos_scale;
        let rot_quat = sw_instances[inst_id].rot_quat;
        let world_pos = transform_pos(pos_scale, rot_quat, local_pos);
        let clip = sw_uniforms.render_view_proj * vec4<f32>(world_pos, 1.0);
        if clip.w > 1.0e-5 {
            let inv_w = 1.0 / clip.w;
            let ndc = clip.xyz * inv_w;
            let vp = sw_uniforms.viewport_lod.xy;
            let sx = (ndc.x * 0.5 + 0.5) * vp.x;
            let sy = (0.5 - ndc.y * 0.5) * vp.y;
            sh_screen[vi] = vec4<f32>(sx, sy, ndc.z, inv_w);
            sh_valid[vi] = 1u;
        } else {
            sh_screen[vi] = vec4<f32>(0.0);
            sh_valid[vi] = 0u;
        }
    } else {
        sh_valid[vi] = 0u;
    }
}

fn setup_tri(page_base: u32, ec_w: u32, vis_slot: u32, tri_id: u32) -> TriSetup {
    var s: TriSetup;
    s.valid = 0u;
    let tri_count = (ec_w >> 8u) & 0xFFu;
    if tri_id >= tri_count {
        return s;
    }
    let vert_count = ec_w & 0xFFu;
    let tri_base = page_base + CLUSTER_HEADER_WORDS + ((vert_count * 3u + 1u) >> 1u);
    let tri_word = read_u24(tri_base, tri_id);
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
    var q0 = vec2<i32>(round(p0.xy * 16.0));
    var q1 = vec2<i32>(round(p1.xy * 16.0));
    var q2 = vec2<i32>(round(p2.xy * 16.0));
    var z0_v = p0.z;
    var z1_v = p1.z;
    var z2_v = p2.z;
    var area = edge_yup_i32(q0, q1, q2);
    let mat_bit = 1u << ((ec_w >> 24u) & 31u);
    let is_ds = (sw_uniforms.extra.x & mat_bit) != 0u;
    if area <= 0 {
        if !is_ds || area == 0 {
            return s;
        }
        let tq = q1;
        q1 = q2;
        q2 = tq;
        let tz = z1_v;
        z1_v = z2_v;
        z2_v = tz;
        area = -area;
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
    let vis_id = ((vis_slot + 1u) << 7u) | (tri_id & 0x7Fu);
    if min_x == max_x && min_y == max_y {
        if (e0_s | e1_s | e2_s) < 0 {
            return s;
        }
        let z0 = (f32(e0_0) * z0_v + f32(e1_0) * z1_v + f32(e2_0) * z2_v) / f32(area);
        if z0 <= 0.0 || z0 > 1.0 {
            return s;
        }
        s.valid = 2u;
        s.z_start = z0;
        s.bbox = vec4<i32>(min_x, min_y, max_x, max_y);
        s.vis_id = vis_id;
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
    s.z_start = (f32(e0_0) * z0_v + f32(e1_0) * z1_v + f32(e2_0) * z2_v) * inv_area;
    s.dz_dx = (f32(dx.x) * z0_v + f32(dx.y) * z1_v + f32(dx.z) * z2_v) * inv_area;
    s.dz_dy = (f32(dy.x) * z0_v + f32(dy.y) * z1_v + f32(dy.z) * z2_v) * inv_area;
    s.bbox = vec4<i32>(min_x, min_y, max_x, max_y);
    s.vis_id = vis_id;
    return s;
}

fn raster_depth_tri(s: TriSetup, stride: u32) {
    if s.valid == 0u {
        return;
    }
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

fn raster_vis_tri(s: TriSetup, stride: u32) {
    if s.valid == 0u {
        return;
    }
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

@compute @workgroup_size(64)
fn cs_sw_depth(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let sw_idx = wid.y * SW_DISPATCH_STRIDE + wid.x;
    let sw_total = sw_draw_list[MAX_PHASE_CLUSTERS * 3u].heap_slot;
    if sw_idx >= sw_total {
        return;
    }
    let list_idx = MAX_PHASE_CLUSTERS * 2u + sw_idx;
    let inst_id = sw_draw_list[list_idx].instance_id;
    let cluster_id = sw_draw_list[list_idx].cluster_id;
    let heap_slot = sw_draw_list[list_idx].heap_slot;
    let vis_slot = sw_draw_list[list_idx].vis_slot;
    let ec_w = sw_clusters[cluster_id].errors_and_cone.w;
    let word_off = sw_clusters[cluster_id].page_and_fallback.y & 0xFFFFu;
    let vert_count = ec_w & 0xFFu;
    let page_base = heap_slot * PAGE_WORDS + word_off;
    let origin = vec3<f32>(
        bitcast<f32>(sw_page_heap[page_base]),
        bitcast<f32>(sw_page_heap[page_base + 1u]),
        bitcast<f32>(sw_page_heap[page_base + 2u]),
    );
    let step = bitcast<f32>(sw_page_heap[page_base + 3u]);
    load_cluster_vert(inst_id, page_base, vert_count, lid, origin, step);
    load_cluster_vert(inst_id, page_base, vert_count, lid + 64u, origin, step);
    workgroupBarrier();
    let stride = u32(sw_uniforms.viewport_lod.x);
    raster_depth_tri(setup_tri(page_base, ec_w, vis_slot, lid), stride);
    raster_depth_tri(setup_tri(page_base, ec_w, vis_slot, lid + 64u), stride);
}

@compute @workgroup_size(64)
fn cs_sw_vis(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let sw_idx = wid.y * SW_DISPATCH_STRIDE + wid.x;
    let sw_total = sw_draw_list[MAX_PHASE_CLUSTERS * 3u].heap_slot;
    if sw_idx >= sw_total {
        return;
    }
    let list_idx = MAX_PHASE_CLUSTERS * 2u + sw_idx;
    let inst_id = sw_draw_list[list_idx].instance_id;
    let cluster_id = sw_draw_list[list_idx].cluster_id;
    let heap_slot = sw_draw_list[list_idx].heap_slot;
    let vis_slot = sw_draw_list[list_idx].vis_slot;
    let ec_w = sw_clusters[cluster_id].errors_and_cone.w;
    let word_off = sw_clusters[cluster_id].page_and_fallback.y & 0xFFFFu;
    let vert_count = ec_w & 0xFFu;
    let page_base = heap_slot * PAGE_WORDS + word_off;
    let origin = vec3<f32>(
        bitcast<f32>(sw_page_heap[page_base]),
        bitcast<f32>(sw_page_heap[page_base + 1u]),
        bitcast<f32>(sw_page_heap[page_base + 2u]),
    );
    let step = bitcast<f32>(sw_page_heap[page_base + 3u]);
    load_cluster_vert(inst_id, page_base, vert_count, lid, origin, step);
    load_cluster_vert(inst_id, page_base, vert_count, lid + 64u, origin, step);
    workgroupBarrier();
    let stride = u32(sw_uniforms.viewport_lod.x);
    raster_vis_tri(setup_tri(page_base, ec_w, vis_slot, lid), stride);
    raster_vis_tri(setup_tri(page_base, ec_w, vis_slot, lid + 64u), stride);
}
