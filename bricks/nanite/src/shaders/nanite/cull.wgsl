import package::generated::nanite::NaniteUniforms;
import package::generated::nanite::NaniteInstance;
import package::generated::nanite::GpuClusterHeader;
import package::generated::nanite::GpuBvhNode;
import package::generated::nanite::VisibleCluster;
import package::hiz::cull::hiz_project_sphere;
import package::hiz::cull::hiz_test_rect;
import package::nanite::common::MAX_PHASE_CLUSTERS;
import package::nanite::common::MAX_VIS_CLUSTERS;
import package::nanite::common::MAX_OCC_INSTANCES;
import package::nanite::common::MAX_OCC_CLUSTERS;
import package::nanite::common::DEDUP_MASK;
import package::nanite::common::BVH_QUEUE_CAP;
import package::nanite::common::CAND_CAP;
import package::nanite::common::SW_DISPATCH_STRIDE;
import package::nanite::common::UNMAPPED_PAGE;
import package::nanite::common::NO_FALLBACK;
import package::nanite::common::NO_CHILD;
import package::nanite::common::inst_to_world;
import package::nanite::common::frustum_visible;
import package::nanite::common::projected_error;
import package::nanite::common::projected_min_error;
import package::nanite::common::cone_backface_culled;
import package::nanite::common::cluster_mat;

const OCC_INST_BASE: u32 = 32u;
const OCC_CLUS_BASE: u32 = 4128u;
const DEDUP_BASE: u32 = 528416u;
const Q_A_BASE: u32 = 0u;
const Q_B_BASE: u32 = 524288u;
const CAND_BASE: u32 = 1048576u;

@group(1) @binding(0) var<uniform> cull_uniforms: NaniteUniforms;
@group(1) @binding(1) var cull_hiz_tex: texture_2d<f32>;
@group(1) @binding(2) var<storage, read> cull_instances: array<NaniteInstance>;
@group(1) @binding(3) var<storage, read> cull_clusters: array<GpuClusterHeader>;
@group(1) @binding(4) var<storage, read> cull_bvh: array<GpuBvhNode>;
@group(1) @binding(5) var<storage, read> cull_page_table: array<u32>;
@group(1) @binding(6) var<storage, read_write> cull_bvh_scratch: array<u32>;
@group(1) @binding(7) var<storage, read_write> cull_vis_clusters: array<VisibleCluster>;
@group(1) @binding(8) var<storage, read_write> cull_draw_lists: array<VisibleCluster>;
@group(1) @binding(9) var<storage, read_write> cull_state: array<atomic<u32>>;
@group(1) @binding(10) var<storage, read_write> cull_page_requests: array<atomic<u32>>;
@group(1) @binding(11) var<storage, read_write> cull_indirect: array<u32>;

fn hiz_occluded_sphere(world_c: vec3<f32>, world_r: f32) -> bool {
    if cull_uniforms.flags.w == 0u {
        return false;
    }
    let center_vs = (cull_uniforms.cull_view * vec4<f32>(world_c, 1.0)).xyz;
    let near_z = cull_uniforms.cull_pos_near.w;
    let d_min = -center_vs.z - world_r;
    if d_min <= near_z {
        return false;
    }
    let rect = hiz_project_sphere(
        center_vs,
        world_r,
        cull_uniforms.cull_proj[0].x,
        cull_uniforms.cull_proj[1].y,
        near_z,
    );
    let nearest_z = near_z / d_min;
    return hiz_test_rect(
        cull_hiz_tex,
        rect.xy,
        rect.zw,
        nearest_z,
        cull_uniforms.hiz_params.xy,
        u32(cull_uniforms.hiz_params.z),
    );
}

fn try_claim_dedup(inst_id: u32, cluster_id: u32) -> bool {
    let key = ((inst_id + 1u) << 21u) | (cluster_id & 0x1FFFFFu);
    var h = (inst_id * 2654435761u) ^ (cluster_id * 2246822519u);
    h = (h ^ (h >> 16u)) * 0x45D9F3Bu;
    var slot = h & DEDUP_MASK;
    for (var probe = 0u; probe < 24u; probe = probe + 1u) {
        let res = atomicCompareExchangeWeak(&cull_state[DEDUP_BASE + slot], 0u, key);
        if res.exchanged {
            return true;
        }
        if res.old_value == key {
            return false;
        }
        slot = (slot + 1u) & DEDUP_MASK;
    }
    return true;
}

@compute @workgroup_size(1)
fn cs_reset_frame() {
    cull_indirect[0] = 192u;
    cull_indirect[1] = 0u;
    cull_indirect[2] = 0u;
    cull_indirect[3] = 0u;
    cull_indirect[4] = 0u;
    cull_indirect[5] = 1u;
    cull_indirect[6] = 1u;
    cull_indirect[7] = 0u;
    for (var i = 8u; i < 32u; i = i + 1u) {
        cull_indirect[i] = 0u;
    }
}

@compute @workgroup_size(64)
fn cs_cull_instances_p1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= cull_uniforms.counts.x {
        return;
    }
    let inst = cull_instances[idx];
    let bs = inst.bound_sphere;
    if !frustum_visible(cull_uniforms, bs.xyz, bs.w) {
        return;
    }
    if hiz_occluded_sphere(bs.xyz, bs.w) {
        let occ_slot = atomicAdd(&cull_state[19u], 1u);
        if occ_slot < MAX_OCC_INSTANCES {
            atomicStore(&cull_state[OCC_INST_BASE + occ_slot], idx);
        }
        atomicAdd(&cull_state[1u], 1u);
        return;
    }
    let q_slot = atomicAdd(&cull_state[16u], 1u);
    if q_slot < BVH_QUEUE_CAP {
        let base = Q_A_BASE + q_slot * 2u;
        cull_bvh_scratch[base] = idx;
        cull_bvh_scratch[base + 1u] = inst.info.x;
    }
    atomicAdd(&cull_state[0u], 1u);
}

@compute @workgroup_size(64)
fn cs_cull_instances_p2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let occ_count = min(atomicLoad(&cull_state[19u]), MAX_OCC_INSTANCES);
    if gid.x >= occ_count {
        return;
    }
    let idx = atomicLoad(&cull_state[OCC_INST_BASE + gid.x]);
    let inst = cull_instances[idx];
    let bs = inst.bound_sphere;
    if hiz_occluded_sphere(bs.xyz, bs.w) {
        return;
    }
    let q_slot = atomicAdd(&cull_state[16u], 1u);
    if q_slot < BVH_QUEUE_CAP {
        let base = Q_A_BASE + q_slot * 2u;
        cull_bvh_scratch[base] = idx;
        cull_bvh_scratch[base + 1u] = inst.info.x;
    }
    atomicAdd(&cull_state[2u], 1u);
}

@compute @workgroup_size(1)
fn cs_prep_bvh_a() {
    let count = min(atomicLoad(&cull_state[16u]), BVH_QUEUE_CAP);
    cull_indirect[8] = (count + 63u) / 64u;
    cull_indirect[9] = 1u;
    cull_indirect[10] = 1u;
    atomicStore(&cull_state[17u], 0u);
}

@compute @workgroup_size(1)
fn cs_prep_bvh_b() {
    let count = min(atomicLoad(&cull_state[17u]), BVH_QUEUE_CAP);
    cull_indirect[12] = (count + 63u) / 64u;
    cull_indirect[13] = 1u;
    cull_indirect[14] = 1u;
    atomicStore(&cull_state[16u], 0u);
}

fn expand_step(gid_x: u32, in_cnt_idx: u32, out_cnt_idx: u32, in_base: u32, out_base: u32) {
    let count = min(atomicLoad(&cull_state[in_cnt_idx]), BVH_QUEUE_CAP);
    if gid_x >= count {
        return;
    }
    let item_off = in_base + gid_x * 2u;
    let inst_id = cull_bvh_scratch[item_off];
    let node_id = cull_bvh_scratch[item_off + 1u];
    let inst = cull_instances[inst_id];
    let node = cull_bvh[node_id];
    let scale = inst.pos_scale.w;
    let cull_pos = cull_uniforms.cull_pos_near.xyz;
    let near_z = cull_uniforms.cull_pos_near.w;
    let lod_scale = cull_uniforms.viewport_lod.z;
    let lod_thresh = cull_uniforms.viewport_lod.w;

    if node.info.y != 0u {
        let next_slot = atomicAdd(&cull_state[out_cnt_idx], 1u);
        if next_slot < BVH_QUEUE_CAP {
            let n_off = out_base + next_slot * 2u;
            cull_bvh_scratch[n_off] = inst_id;
            cull_bvh_scratch[n_off + 1u] = node_id;
        }
        return;
    }

    let child_cnt = min(node.info.x, 4u);
    for (var k = 0u; k < child_cnt; k = k + 1u) {
        let child_id = node.child_or_cluster[k];
        if child_id == NO_CHILD {
            continue;
        }
        let child = cull_bvh[child_id];
        let world_c = inst_to_world(inst, child.sphere.xyz);
        let world_r = child.sphere.w * scale;
        if !frustum_visible(cull_uniforms, world_c, world_r) {
            continue;
        }
        let max_pe = node.max_parent_error[k];
        if max_pe < 1.0e29 {
            let err_px = projected_error(cull_pos, near_z, lod_scale, world_c, world_r, max_pe, scale);
            if err_px <= lod_thresh {
                continue;
            }
        }
        let min_se = node.min_self_error[k];
        if min_se > 0.0 {
            let min_px = projected_min_error(cull_pos, near_z, lod_scale, world_c, world_r, min_se, scale);
            if min_px > lod_thresh {
                continue;
            }
        }
        let next_slot = atomicAdd(&cull_state[out_cnt_idx], 1u);
        if next_slot < BVH_QUEUE_CAP {
            let n_off = out_base + next_slot * 2u;
            cull_bvh_scratch[n_off] = inst_id;
            cull_bvh_scratch[n_off + 1u] = child_id;
        }
    }
}

@compute @workgroup_size(64)
fn cs_traverse_bvh_ab(@builtin(global_invocation_id) gid: vec3<u32>) {
    expand_step(gid.x, 16u, 17u, Q_A_BASE, Q_B_BASE);
}

@compute @workgroup_size(64)
fn cs_traverse_bvh_ba(@builtin(global_invocation_id) gid: vec3<u32>) {
    expand_step(gid.x, 17u, 16u, Q_B_BASE, Q_A_BASE);
}

@compute @workgroup_size(64)
fn cs_bvh_dfs_cut(@builtin(global_invocation_id) gid: vec3<u32>) {
    let count = min(atomicLoad(&cull_state[16u]), BVH_QUEUE_CAP);
    if gid.x >= count {
        return;
    }
    let item_off = Q_A_BASE + gid.x * 2u;
    let inst_id = cull_bvh_scratch[item_off];
    let root_node = cull_bvh_scratch[item_off + 1u];
    let inst = cull_instances[inst_id];
    let scale = inst.pos_scale.w;
    let cull_pos = cull_uniforms.cull_pos_near.xyz;
    let near_z = cull_uniforms.cull_pos_near.w;
    let lod_scale = cull_uniforms.viewport_lod.z;
    let lod_thresh = cull_uniforms.viewport_lod.w;

    var stack: array<u32, 64>;
    var sp: u32 = 1u;
    stack[0] = root_node;
    var local_tested: u32 = 0u;
    var local_frustum: u32 = 0u;
    var local_cone: u32 = 0u;

    while sp > 0u {
        sp = sp - 1u;
        let node = cull_bvh[stack[sp]];
        if node.info.y != 0u {
            let start = node.info.z;
            let cnt = node.info.w;
            for (var j = 0u; j < cnt; j = j + 1u) {
                let cid = start + j;
                var parent_proj = 1.0e30;
                let pe = cull_clusters[cid].aabb_extent.w;
                if pe < 1.0e29 {
                    let pls = cull_clusters[cid].parent_lod_sphere;
                    let pc = inst_to_world(inst, pls.xyz);
                    let pr = pls.w * scale;
                    parent_proj = projected_error(cull_pos, near_z, lod_scale, pc, pr, pe, scale);
                }
                if parent_proj <= lod_thresh {
                    continue;
                }
                let se = cull_clusters[cid].aabb_min.w;
                let ls = cull_clusters[cid].lod_sphere;
                let sc = inst_to_world(inst, ls.xyz);
                let sr = ls.w * scale;
                let self_proj = projected_error(cull_pos, near_z, lod_scale, sc, sr, se, scale);
                if self_proj > lod_thresh {
                    continue;
                }
                local_tested = local_tested + 1u;
                let c = cull_clusters[cid];
                let world_c = inst_to_world(inst, c.bound_sphere.xyz);
                let world_r = c.bound_sphere.w * scale;
                if !frustum_visible(cull_uniforms, world_c, world_r) {
                    local_frustum = local_frustum + 1u;
                    continue;
                }
                let mat_bit = 1u << (cluster_mat(c) & 31u);
                let is_ds = (cull_uniforms.extra.x & mat_bit) != 0u;
                if !is_ds && cone_backface_culled(c, inst, world_c, world_r, cull_pos) {
                    local_cone = local_cone + 1u;
                    continue;
                }
                let slot = atomicAdd(&cull_state[18u], 1u);
                if slot < CAND_CAP {
                    let c_off = CAND_BASE + slot * 2u;
                    cull_bvh_scratch[c_off] = inst_id;
                    cull_bvh_scratch[c_off + 1u] = cid;
                }
            }
            continue;
        }

        let child_cnt = min(node.info.x, 4u);
        for (var k = 0u; k < child_cnt; k = k + 1u) {
            let child_id = node.child_or_cluster[k];
            if child_id == NO_CHILD {
                continue;
            }
            let child = cull_bvh[child_id];
            let world_c = inst_to_world(inst, child.sphere.xyz);
            let world_r = child.sphere.w * scale;
            if !frustum_visible(cull_uniforms, world_c, world_r) {
                continue;
            }
            let max_pe = node.max_parent_error[k];
            if max_pe < 1.0e29 {
                let err_px = projected_error(cull_pos, near_z, lod_scale, world_c, world_r, max_pe, scale);
                if err_px <= lod_thresh {
                    continue;
                }
            }
            let min_se = node.min_self_error[k];
            if min_se > 0.0 {
                let min_px = projected_min_error(cull_pos, near_z, lod_scale, world_c, world_r, min_se, scale);
                if min_px > lod_thresh {
                    continue;
                }
            }
            if sp < 64u {
                stack[sp] = child_id;
                sp = sp + 1u;
            }
        }
    }
    if local_tested > 0u {
        atomicAdd(&cull_state[3u], local_tested);
    }
    if local_frustum > 0u {
        atomicAdd(&cull_state[13u], local_frustum);
    }
    if local_cone > 0u {
        atomicAdd(&cull_state[12u], local_cone);
    }
}

@compute @workgroup_size(1)
fn cs_prep_candidates() {
    let count = min(atomicLoad(&cull_state[18u]), CAND_CAP);
    cull_indirect[16] = (count + 63u) / 64u;
    cull_indirect[17] = 1u;
    cull_indirect[18] = 1u;
}

fn emit_visible_cluster(inst_id: u32, draw_cid: u32, slot: u32, is_p2: bool, is_fallback: u32) {
    let dc_bs = cull_clusters[draw_cid].bound_sphere;
    let dc_meta = cull_clusters[draw_cid].cone_and_meta;
    atomicMax(&cull_page_requests[dc_meta.z], 1u);

    if is_fallback != 0u {
        atomicAdd(&cull_state[11u], 1u);
    }

    let inst = cull_instances[inst_id];
    let scale = inst.pos_scale.w;
    let near_z = cull_uniforms.cull_pos_near.w;
    let dc_world_c = inst_to_world(inst, dc_bs.xyz);
    let dc_world_r = dc_bs.w * scale;
    let render_lod_scale = max(cull_uniforms.shadow_info.z, 1.0);
    let clip_c = cull_uniforms.render_view_proj * vec4<f32>(dc_world_c, 1.0);
    let near_straddle = clip_c.w <= dc_world_r * 1.45 + near_z * 2.0;
    let cluster_px = (2.0 * dc_world_r * render_lod_scale) / max(clip_c.w - dc_world_r, near_z);
    let tc = (dc_meta.y >> 8u) & 0xFFu;
    let tri_px = cluster_px * inverseSqrt(max(f32(tc), 1.0));
    let is_sw = !near_straddle
        && cull_uniforms.flags.y == 0u
        && tri_px <= cull_uniforms.hiz_params.w
        && cluster_px <= 16.0;

    let vis_idx = atomicAdd(&cull_state[25u], 1u);
    if vis_idx >= MAX_VIS_CLUSTERS {
        return;
    }
    let vflags = select(0u, 1u, is_sw)
        | select(0u, 2u, is_p2)
        | (is_fallback << 2u)
        | (((dc_meta.y >> 16u) & 0xFFu) << 8u);
    cull_vis_clusters[vis_idx] = VisibleCluster(inst_id, draw_cid, slot, vflags);
    atomicAdd(&cull_state[15u], 1u);

    let item = VisibleCluster(inst_id, draw_cid, slot, vis_idx);
    if is_sw {
        let cnt_idx = select(22u, 24u, is_p2);
        let sw_idx = atomicAdd(&cull_state[cnt_idx], 1u);
        if sw_idx < MAX_PHASE_CLUSTERS {
            cull_draw_lists[MAX_PHASE_CLUSTERS + sw_idx] = item;
        }
        atomicAdd(&cull_state[select(5u, 7u, is_p2)], 1u);
        atomicAdd(&cull_state[9u], tc);
    } else {
        let cnt_idx = select(21u, 23u, is_p2);
        let hw_idx = atomicAdd(&cull_state[cnt_idx], 1u);
        if hw_idx < MAX_PHASE_CLUSTERS {
            cull_draw_lists[hw_idx] = item;
        }
        atomicAdd(&cull_state[select(4u, 6u, is_p2)], 1u);
        atomicAdd(&cull_state[8u], tc);
    }
}

fn evaluate_cluster(inst_id: u32, cluster_id: u32, is_p2: bool) {
    let inst = cull_instances[inst_id];
    let c = cull_clusters[cluster_id];
    let scale = inst.pos_scale.w;
    let cull_pos = cull_uniforms.cull_pos_near.xyz;
    let near_z = cull_uniforms.cull_pos_near.w;
    let lod_scale = cull_uniforms.viewport_lod.z;

    let world_c = inst_to_world(inst, c.bound_sphere.xyz);
    let world_r = c.bound_sphere.w * scale;

    if hiz_occluded_sphere(world_c, world_r) {
        atomicAdd(&cull_state[14u], 1u);
        if !is_p2 {
            let occ_slot = atomicAdd(&cull_state[20u], 1u);
            if occ_slot < MAX_OCC_CLUSTERS {
                let off = OCC_CLUS_BASE + occ_slot * 2u;
                atomicStore(&cull_state[off], inst_id);
                atomicStore(&cull_state[off + 1u], cluster_id);
            }
        }
        return;
    }

    let orig_page = c.cone_and_meta.z;
    let slot = cull_page_table[orig_page];
    if slot != UNMAPPED_PAGE {
        emit_visible_cluster(inst_id, cluster_id, slot, is_p2, 0u);
        return;
    }

    var parent_proj = 1024.0;
    let pe = c.aabb_extent.w;
    if pe < 1.0e29 {
        let pc = inst_to_world(inst, c.parent_lod_sphere.xyz);
        let pr = c.parent_lod_sphere.w * scale;
        parent_proj = projected_error(cull_pos, near_z, lod_scale, pc, pr, pe, scale);
    }
    let prio = clamp(u32(min(parent_proj, 65535.0) * 64.0) + 1u, 2u, 0x7FFFFFFFu);
    atomicMax(&cull_page_requests[orig_page], prio);
    atomicAdd(&cull_state[10u], 1u);

    var fb_stack: array<u32, 32>;
    var sp: u32 = 0u;
    let fb0 = c.streaming_meta.y;
    let fb1 = c.streaming_meta.w;
    if fb1 != NO_FALLBACK {
        fb_stack[sp] = fb1;
        sp = sp + 1u;
    }
    if fb0 != NO_FALLBACK {
        fb_stack[sp] = fb0;
        sp = sp + 1u;
    }
    for (var step = 0u; step < 384u; step = step + 1u) {
        if sp == 0u {
            break;
        }
        sp = sp - 1u;
        let cur = fb_stack[sp];
        if !try_claim_dedup(inst_id, cur) {
            continue;
        }
        let fp = cull_clusters[cur].cone_and_meta.z;
        let fs = cull_page_table[fp];
        if fs != UNMAPPED_PAGE {
            emit_visible_cluster(inst_id, cur, fs, is_p2, 1u);
        } else {
            let fprio = clamp(prio + (384u - step), 2u, 0x7FFFFFFFu);
            atomicMax(&cull_page_requests[fp], fprio);
            let sm = cull_clusters[cur].streaming_meta;
            if sm.w != NO_FALLBACK && sp < 32u {
                fb_stack[sp] = sm.w;
                sp = sp + 1u;
            }
            if sm.y != NO_FALLBACK && sp < 32u {
                fb_stack[sp] = sm.y;
                sp = sp + 1u;
            }
        }
    }
}

@compute @workgroup_size(64)
fn cs_cull_clusters_p1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let count = min(atomicLoad(&cull_state[18u]), CAND_CAP);
    if gid.x >= count {
        return;
    }
    let off = CAND_BASE + gid.x * 2u;
    let inst_id = cull_bvh_scratch[off];
    let cluster_id = cull_bvh_scratch[off + 1u];
    evaluate_cluster(inst_id, cluster_id, false);
}

@compute @workgroup_size(64)
fn cs_cull_clusters_p2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let count = min(atomicLoad(&cull_state[18u]), CAND_CAP);
    if gid.x >= count {
        return;
    }
    let off = CAND_BASE + gid.x * 2u;
    let inst_id = cull_bvh_scratch[off];
    let cluster_id = cull_bvh_scratch[off + 1u];
    evaluate_cluster(inst_id, cluster_id, true);
}

@compute @workgroup_size(64)
fn cs_cull_occ_clusters_p2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let count = min(atomicLoad(&cull_state[20u]), MAX_OCC_CLUSTERS);
    if gid.x >= count {
        return;
    }
    let off = OCC_CLUS_BASE + gid.x * 2u;
    let inst_id = atomicLoad(&cull_state[off]);
    let cluster_id = atomicLoad(&cull_state[off + 1u]);
    evaluate_cluster(inst_id, cluster_id, true);
}

@compute @workgroup_size(1)
fn cs_prep_draws_p1() {
    let hw_cnt = min(atomicLoad(&cull_state[21u]), MAX_PHASE_CLUSTERS);
    let sw_cnt = min(atomicLoad(&cull_state[22u]), MAX_PHASE_CLUSTERS);
    let occ_i = min(atomicLoad(&cull_state[19u]), MAX_OCC_INSTANCES);
    let occ_c = min(atomicLoad(&cull_state[20u]), MAX_OCC_CLUSTERS);
    let sw_x = min(sw_cnt, SW_DISPATCH_STRIDE);
    let sw_y = select(0u, (sw_cnt + SW_DISPATCH_STRIDE - 1u) / SW_DISPATCH_STRIDE, sw_cnt > 0u);
    cull_draw_lists[MAX_PHASE_CLUSTERS * 2u] = VisibleCluster(hw_cnt, sw_cnt, 0u, 0u);
    cull_indirect[0] = 192u;
    cull_indirect[1] = hw_cnt;
    cull_indirect[2] = 0u;
    cull_indirect[3] = 0u;
    cull_indirect[4] = sw_x;
    cull_indirect[5] = sw_y;
    cull_indirect[6] = 1u;
    cull_indirect[7] = 0u;
    cull_indirect[20] = (occ_i + 63u) / 64u;
    cull_indirect[21] = 1u;
    cull_indirect[22] = 1u;
    cull_indirect[24] = (occ_c + 63u) / 64u;
    cull_indirect[25] = 1u;
    cull_indirect[26] = 1u;
}

@compute @workgroup_size(1)
fn cs_prep_draws_p2() {
    let hw_cnt = min(atomicLoad(&cull_state[23u]), MAX_PHASE_CLUSTERS);
    let sw_cnt = min(atomicLoad(&cull_state[24u]), MAX_PHASE_CLUSTERS);
    let sw_x = min(sw_cnt, SW_DISPATCH_STRIDE);
    let sw_y = select(0u, (sw_cnt + SW_DISPATCH_STRIDE - 1u) / SW_DISPATCH_STRIDE, sw_cnt > 0u);
    cull_draw_lists[MAX_PHASE_CLUSTERS * 2u] = VisibleCluster(hw_cnt, sw_cnt, 0u, 0u);
    cull_indirect[0] = 192u;
    cull_indirect[1] = hw_cnt;
    cull_indirect[2] = 0u;
    cull_indirect[3] = 0u;
    cull_indirect[4] = sw_x;
    cull_indirect[5] = sw_y;
    cull_indirect[6] = 1u;
    cull_indirect[7] = 0u;
}
