import package::generated::nanite::NaniteUniforms;
import package::generated::nanite::NaniteInstance;
import package::generated::nanite::GpuClusterHeader;
import package::generated::nanite::VisibleCluster;
import package::nanite::common::PAGE_WORDS;
import package::nanite::common::UNMAPPED_PAGE;
import package::nanite::common::NO_FALLBACK;
import package::nanite::common::SHADOW_C0_DIM;
import package::nanite::common::SHADOW_C1_DIM;
import package::nanite::common::SHADOW_LIST_CAP;
import package::nanite::common::inst_to_world;
import package::nanite::common::decode_pos;
import package::nanite::common::cluster_vert_count;
import package::nanite::common::cluster_tri_count;
import package::nanite::common::edge_yup;

@group(1) @binding(0) var<uniform> sh_uniforms: NaniteUniforms;
@group(1) @binding(1) var<storage, read> sh_instances: array<NaniteInstance>;
@group(1) @binding(2) var<storage, read> sh_clusters: array<GpuClusterHeader>;
@group(1) @binding(3) var<storage, read> sh_page_table: array<u32>;
@group(1) @binding(4) var<storage, read> sh_page_heap: array<u32>;
@group(1) @binding(5) var<storage, read> sh_cuts: array<u32>;
@group(1) @binding(6) var<storage, read> sh_ranges: array<vec2<u32>>;
@group(1) @binding(7) var<storage, read_write> sh_list: array<VisibleCluster>;
@group(1) @binding(8) var<storage, read_write> sh_indirect: array<atomic<u32>>;
@group(1) @binding(9) var<storage, read_write> sh_depth: array<atomic<u32>>;

var<workgroup> sh_v: array<vec3<f32>, 64>;

fn shadow_collect(inst_id: u32, cascade: u32) {
    if inst_id >= sh_uniforms.counts.x {
        return;
    }
    let inst = sh_instances[inst_id];
    let bs = inst.bound_sphere;
    var vp = sh_uniforms.shadow_vp0;
    if cascade != 0u {
        vp = sh_uniforms.shadow_vp1;
    }
    let half_ext = select(sh_uniforms.shadow_center0.w, sh_uniforms.shadow_center1.w, cascade != 0u);
    let clip_c = vp * vec4<f32>(bs.xyz, 1.0);
    let r_ndc = bs.w / max(half_ext, 1.0);
    if abs(clip_c.x) > 1.0 + r_ndc || abs(clip_c.y) > 1.0 + r_ndc {
        return;
    }
    let mesh_id = inst.info.w;
    let range = sh_ranges[mesh_id * 2u + cascade];
    var last0: u32 = 0xFFFFFFFFu;
    var last1: u32 = 0xFFFFFFFFu;
    for (var i = 0u; i < range.y; i = i + 1u) {
        let cid = sh_cuts[range.x + i];
        let c0 = sh_clusters[cid];
        let slot = sh_page_table[c0.cone_and_meta.z];
        if slot != UNMAPPED_PAGE {
            let out_idx = atomicAdd(&sh_indirect[4u], 1u);
            if out_idx < SHADOW_LIST_CAP {
                sh_list[out_idx] = VisibleCluster(inst_id, cid, slot, cascade);
            }
            continue;
        }
        var fb_stack: array<u32, 16>;
        var sp: u32 = 0u;
        if c0.streaming_meta.w != NO_FALLBACK {
            fb_stack[sp] = c0.streaming_meta.w;
            sp = sp + 1u;
        }
        if c0.streaming_meta.y != NO_FALLBACK {
            fb_stack[sp] = c0.streaming_meta.y;
            sp = sp + 1u;
        }
        for (var s = 0u; s < 32u; s = s + 1u) {
            if sp == 0u {
                break;
            }
            sp = sp - 1u;
            let cur = fb_stack[sp];
            let fc = sh_clusters[cur];
            let fs = sh_page_table[fc.cone_and_meta.z];
            if fs != UNMAPPED_PAGE {
                if cur != last0 && cur != last1 {
                    last1 = last0;
                    last0 = cur;
                    let out_idx = atomicAdd(&sh_indirect[4u], 1u);
                    if out_idx < SHADOW_LIST_CAP {
                        sh_list[out_idx] = VisibleCluster(inst_id, cur, fs, cascade);
                    }
                }
            } else {
                if fc.streaming_meta.w != NO_FALLBACK && sp < 16u {
                    fb_stack[sp] = fc.streaming_meta.w;
                    sp = sp + 1u;
                }
                if fc.streaming_meta.y != NO_FALLBACK && sp < 16u {
                    fb_stack[sp] = fc.streaming_meta.y;
                    sp = sp + 1u;
                }
            }
        }
    }
}

@compute @workgroup_size(64)
fn cs_shadow_collect_c0(@builtin(global_invocation_id) gid: vec3<u32>) {
    shadow_collect(gid.x, 0u);
}

@compute @workgroup_size(64)
fn cs_shadow_collect_c1(@builtin(global_invocation_id) gid: vec3<u32>) {
    shadow_collect(gid.x, 1u);
}

@compute @workgroup_size(1)
fn cs_shadow_prep() {
    let cnt = min(atomicLoad(&sh_indirect[4u]), SHADOW_LIST_CAP);
    atomicStore(&sh_indirect[0u], cnt);
    atomicStore(&sh_indirect[1u], 1u);
    atomicStore(&sh_indirect[2u], 1u);
}

fn shadow_raster(wid_x: u32, lid: u32, cascade: u32) {
    let vc = sh_list[wid_x];
    let c = sh_clusters[vc.cluster_id];
    let vert_count = cluster_vert_count(c);
    let dim_u = select(SHADOW_C0_DIM, SHADOW_C1_DIM, cascade != 0u);
    let dim_f = f32(dim_u);
    var vp = sh_uniforms.shadow_vp0;
    if cascade != 0u {
        vp = sh_uniforms.shadow_vp1;
    }
    let page_base = vc.heap_slot * PAGE_WORDS + c.cone_and_meta.w;

    if lid < vert_count {
        let v_base = page_base + lid * 4u;
        let w0 = sh_page_heap[v_base];
        let w1 = sh_page_heap[v_base + 1u];
        let local_pos = decode_pos(w0, w1, c.aabb_min.xyz, c.aabb_extent.xyz);
        let inst = sh_instances[vc.instance_id];
        let world_pos = inst_to_world(inst, local_pos);
        let clip = vp * vec4<f32>(world_pos, 1.0);
        let sx = (clip.x * 0.5 + 0.5) * dim_f;
        let sy = (0.5 - clip.y * 0.5) * dim_f;
        sh_v[lid] = vec3<f32>(sx, sy, clip.z);
    }
    workgroupBarrier();

    let tri_count = cluster_tri_count(c);
    if lid >= tri_count {
        return;
    }
    let tri_word = sh_page_heap[page_base + vert_count * 4u + lid];
    let p0 = sh_v[tri_word & 0xFFu];
    let p1 = sh_v[(tri_word >> 8u) & 0xFFu];
    let p2 = sh_v[(tri_word >> 16u) & 0xFFu];
    let area = edge_yup(p0.xy, p1.xy, p2.xy);
    if abs(area) <= 1.0e-5 {
        return;
    }
    let sign_a = select(-1.0, 1.0, area > 0.0);
    let inv_area = 1.0 / abs(area);
    let dim_i = i32(dim_u);
    let min_x = max(i32(floor(min(p0.x, min(p1.x, p2.x)))), 0);
    let max_x = min(i32(floor(max(p0.x, max(p1.x, p2.x)))), dim_i - 1);
    let min_y = max(i32(floor(min(p0.y, min(p1.y, p2.y)))), 0);
    let max_y = min(i32(floor(max(p0.y, max(p1.y, p2.y)))), dim_i - 1);
    if min_x > max_x || min_y > max_y || (max_x - min_x) > 512 || (max_y - min_y) > 512 {
        return;
    }
    let p_start = vec2<f32>(f32(min_x) + 0.5, f32(min_y) + 0.5);
    var e0_row = edge_yup(p1.xy, p2.xy, p_start) * sign_a;
    var e1_row = edge_yup(p2.xy, p0.xy, p_start) * sign_a;
    var e2_row = edge_yup(p0.xy, p1.xy, p_start) * sign_a;
    let de0_dx = (p2.y - p1.y) * sign_a;
    let de1_dx = (p0.y - p2.y) * sign_a;
    let de2_dx = (p1.y - p0.y) * sign_a;
    let de0_dy = (p1.x - p2.x) * sign_a;
    let de1_dy = (p2.x - p0.x) * sign_a;
    let de2_dy = (p0.x - p1.x) * sign_a;
    var z_row = (e0_row * p0.z + e1_row * p1.z + e2_row * p2.z) * inv_area;
    let dz_dx = (de0_dx * p0.z + de1_dx * p1.z + de2_dx * p2.z) * inv_area;
    let dz_dy = (de0_dy * p0.z + de1_dy * p1.z + de2_dy * p2.z) * inv_area;
    let base_off = select(0u, SHADOW_C0_DIM * SHADOW_C0_DIM, cascade != 0u);
    for (var y = min_y; y <= max_y; y = y + 1) {
        let row = base_off + u32(y) * dim_u;
        var e0 = e0_row;
        var e1 = e1_row;
        var e2 = e2_row;
        var z = z_row;
        for (var x = min_x; x <= max_x; x = x + 1) {
            if e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0 && z > 0.0 && z <= 1.0 {
                let zb = bitcast<u32>(z);
                let idx = row + u32(x);
                if zb > atomicLoad(&sh_depth[idx]) {
                    atomicMax(&sh_depth[idx], zb);
                }
            }
            e0 += de0_dx;
            e1 += de1_dx;
            e2 += de2_dx;
            z += dz_dx;
        }
        e0_row += de0_dy;
        e1_row += de1_dy;
        e2_row += de2_dy;
        z_row += dz_dy;
    }
}

@compute @workgroup_size(64)
fn cs_shadow_raster_c0(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    shadow_raster(wid.x, lid, 0u);
}

@compute @workgroup_size(64)
fn cs_shadow_raster_c1(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    shadow_raster(wid.x, lid, 1u);
}
