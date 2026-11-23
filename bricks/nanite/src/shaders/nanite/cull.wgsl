import package::generated::nanite::NaniteUniforms;
import package::generated::nanite::NaniteInstance;
import package::generated::nanite::GpuClusterHeader;
import package::generated::nanite::GpuBvhNode;
import package::generated::nanite::VisibleCluster;
import package::hiz::cull::hiz_project_sphere_raw;
import package::hiz::cull::hiz_test_rect;
import package::nanite::common::CLUSTER_DRAW_VERTS;
import package::nanite::common::MAX_PHASE_CLUSTERS;
import package::nanite::common::MAX_VIS_CLUSTERS;
import package::nanite::common::MAX_OCC_INSTANCES;
import package::nanite::common::MAX_OCC_CLUSTERS;
import package::nanite::common::DEDUP_MASK;
import package::nanite::common::BVH_QUEUE_CAP;
import package::nanite::common::CAND_CAP;
import package::nanite::common::SW_DISPATCH_STRIDE;
import package::nanite::common::UNMAPPED_PAGE;
import package::nanite::common::NO_CHILD;
import package::nanite::common::quat_rotate;
import package::nanite::common::projected_error;

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

fn inst_world_pos(pos_scale: vec4<f32>, rot: vec4<f32>, p: vec3<f32>) -> vec3<f32> {
    return pos_scale.xyz + quat_rotate(rot, p * pos_scale.w);
}

fn cull_frustum_visible(c: vec3<f32>, r: f32) -> bool {
    var m = dot(cull_uniforms.plane0.xyz, c) + cull_uniforms.plane0.w;
    m = min(m, dot(cull_uniforms.plane1.xyz, c) + cull_uniforms.plane1.w);
    m = min(m, dot(cull_uniforms.plane2.xyz, c) + cull_uniforms.plane2.w);
    m = min(m, dot(cull_uniforms.plane3.xyz, c) + cull_uniforms.plane3.w);
    m = min(m, dot(cull_uniforms.plane4.xyz, c) + cull_uniforms.plane4.w);
    m = min(m, dot(cull_uniforms.plane5.xyz, c) + cull_uniforms.plane5.w);
    return m >= -r;
}

fn unpack_i8(word: u32, shift: u32) -> i32 {
    let raw = i32((word >> shift) & 0xFFu);
    if raw >= 128 {
        return raw - 256;
    }
    return raw;
}

fn cone_culled_packed(
    packed: u32,
    rot_quat: vec4<f32>,
    world_c: vec3<f32>,
    world_r: f32,
    cull_pos: vec3<f32>,
) -> bool {
    let cutoff = unpack_i8(packed, 24u);
    if cutoff >= 127 {
        return false;
    }
    let to_c = world_c - cull_pos;
    let dist = length(to_c);
    if dist <= world_r {
        return false;
    }
    let ax = vec3<f32>(
        f32(unpack_i8(packed, 0u)),
        f32(unpack_i8(packed, 8u)),
        f32(unpack_i8(packed, 16u)),
    );
    let ax_l2 = dot(ax, ax);
    if ax_l2 <= 1.0e-4 {
        return false;
    }
    let local_axis = ax * inverseSqrt(ax_l2);
    let world_axis = quat_rotate(rot_quat, local_axis);
    let sin_cone = f32(cutoff) * (1.0 / 127.0);
    return dot(to_c, world_axis) >= sin_cone * dist + world_r;
}

fn hiz_occluded_sphere(world_c: vec3<f32>, world_r: f32, is_prev: bool) -> bool {
    if cull_uniforms.flags.w == 0u {
        return false;
    }
    var center_vs: vec3<f32>;
    if is_prev {
        let m = cull_uniforms.prev_cull_view;
        center_vs = m[0].xyz * world_c.x + m[1].xyz * world_c.y + m[2].xyz * world_c.z + m[3].xyz;
    } else {
        let m = cull_uniforms.cull_view;
        center_vs = m[0].xyz * world_c.x + m[1].xyz * world_c.y + m[2].xyz * world_c.z + m[3].xyz;
    }
    let near_z = cull_uniforms.cull_pos_near.w;
    let d_min = -center_vs.z - world_r;
    if d_min <= near_z {
        return false;
    }
    let p00 = select(cull_uniforms.cull_proj[0].x, cull_uniforms.prev_cull_proj[0].x, is_prev);
    let p11 = select(cull_uniforms.cull_proj[1].y, cull_uniforms.prev_cull_proj[1].y, is_prev);
    let raw_rect = hiz_project_sphere_raw(center_vs, world_r, p00, p11, near_z);
    if is_prev {
        let pad = vec2<f32>(1.0) / max(cull_uniforms.viewport_lod.xy, vec2<f32>(1.0));
        if raw_rect.x <= pad.x
            || raw_rect.y <= pad.y
            || raw_rect.z >= 1.0 - pad.x
            || raw_rect.w >= 1.0 - pad.y
        {
            return false;
        }
    }
    let nearest_z = near_z / d_min;
    return hiz_test_rect(
        cull_hiz_tex,
        raw_rect.xy,
        raw_rect.zw,
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
    for (var probe = 0u; probe < 32u; probe = probe + 1u) {
        let res = atomicCompareExchangeWeak(&cull_state[DEDUP_BASE + slot], 0u, key);
        if res.exchanged {
            return true;
        }
        if res.old_value == key {
            return false;
        }
        slot = (slot + 1u) & DEDUP_MASK;
    }
    return false;
}

@compute @workgroup_size(1)
fn cs_reset_frame() {
    cull_indirect[0] = CLUSTER_DRAW_VERTS;
    cull_indirect[1] = 0u;
    cull_indirect[2] = 0u;
    cull_indirect[3] = 0u;
    cull_indirect[4] = 0u;
    cull_indirect[5] = 1u;
    cull_indirect[6] = 1u;
    cull_indirect[7] = 0u;
    for (var i = 8u; i < 28u; i = i + 1u) {
        cull_indirect[i] = 0u;
    }
    cull_indirect[28] = CLUSTER_DRAW_VERTS;
    cull_indirect[29] = 0u;
    cull_indirect[30] = 0u;
    cull_indirect[31] = 0u;
}

@compute @workgroup_size(64)
fn cs_cull_instances_p1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= cull_uniforms.counts.x {
        return;
    }
    let bs = cull_instances[idx].bound_sphere;
    if !cull_frustum_visible(bs.xyz, bs.w) {
        return;
    }
    if hiz_occluded_sphere(bs.xyz, bs.w, true) {
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
        cull_bvh_scratch[base + 1u] = cull_instances[idx].info.x;
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
    let bs = cull_instances[idx].bound_sphere;
    if hiz_occluded_sphere(bs.xyz, bs.w, false) {
        return;
    }
    let q_slot = atomicAdd(&cull_state[16u], 1u);
    if q_slot < BVH_QUEUE_CAP {
        let base = Q_A_BASE + q_slot * 2u;
        cull_bvh_scratch[base] = idx;
        cull_bvh_scratch[base + 1u] = cull_instances[idx].info.x;
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

fn bvh_child_visible(
    node_id: u32,
    k: u32,
    child_id: u32,
    inst_pos_scale: vec4<f32>,
    inst_rot: vec4<f32>,
    cull_pos: vec3<f32>,
    near_z: f32,
    err_factor: f32,
    lod_thresh: f32,
) -> bool {
    let child_sphere = cull_bvh[child_id].sphere;
    let world_c = inst_world_pos(inst_pos_scale, inst_rot, child_sphere.xyz);
    let world_r = child_sphere.w * inst_pos_scale.w;
    if !cull_frustum_visible(world_c, world_r) {
        return false;
    }
    let max_pe = cull_bvh[node_id].max_parent_error[k];
    let min_se = cull_bvh[node_id].min_self_error[k];
    if max_pe < 1.0e29 || min_se > 0.0 {
        let dist_c = length(cull_pos - world_c);
        if max_pe < 1.0e29 && max_pe * err_factor <= lod_thresh * max(dist_c - world_r, near_z) {
            return false;
        }
        if min_se > 0.0 && min_se * err_factor > lod_thresh * max(dist_c + world_r, near_z) {
            return false;
        }
    }
    return true;
}

fn expand_step(gid_x: u32, in_cnt_idx: u32, out_cnt_idx: u32, in_base: u32, out_base: u32) {
    let count = min(atomicLoad(&cull_state[in_cnt_idx]), BVH_QUEUE_CAP);
    if gid_x >= count {
        return;
    }
    let item_off = in_base + gid_x * 2u;
    let inst_id = cull_bvh_scratch[item_off];
    let node_id = cull_bvh_scratch[item_off + 1u];
    let node_info = cull_bvh[node_id].info;

    if node_info.y != 0u {
        let next_slot = atomicAdd(&cull_state[out_cnt_idx], 1u);
        if next_slot < BVH_QUEUE_CAP {
            let n_off = out_base + next_slot * 2u;
            cull_bvh_scratch[n_off] = inst_id;
            cull_bvh_scratch[n_off + 1u] = node_id;
        }
        return;
    }

    let inst_pos_scale = cull_instances[inst_id].pos_scale;
    let inst_rot = cull_instances[inst_id].rot_quat;
    let cull_pos = cull_uniforms.cull_pos_near.xyz;
    let near_z = cull_uniforms.cull_pos_near.w;
    let err_factor = inst_pos_scale.w * cull_uniforms.viewport_lod.z;
    let lod_thresh = cull_uniforms.viewport_lod.w;

    let child_cnt = min(node_info.x, 4u);
    for (var k = 0u; k < child_cnt; k = k + 1u) {
        let child_id = cull_bvh[node_id].child_or_cluster[k];
        if child_id == NO_CHILD {
            continue;
        }
        if !bvh_child_visible(
            node_id,
            k,
            child_id,
            inst_pos_scale,
            inst_rot,
            cull_pos,
            near_z,
            err_factor,
            lod_thresh,
        ) {
            continue;
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
    let inst_pos_scale = cull_instances[inst_id].pos_scale;
    let inst_rot = cull_instances[inst_id].rot_quat;
    let scale = inst_pos_scale.w;
    let cull_pos = cull_uniforms.cull_pos_near.xyz;
    let near_z = cull_uniforms.cull_pos_near.w;
    let err_factor = scale * cull_uniforms.viewport_lod.z;
    let lod_thresh = cull_uniforms.viewport_lod.w;
    let ds_mask = cull_uniforms.extra.x;

    var stack: array<u32, 32>;
    stack[0] = root_node;
    var top: u32 = 1u;
    var local_tested: u32 = 0u;
    var local_frustum: u32 = 0u;
    var local_cone: u32 = 0u;

    while top > 0u {
        top = top - 1u;
        let node_id = stack[top];
        let node_info = cull_bvh[node_id].info;
        if node_info.y != 0u {
            let end = node_info.z + node_info.w;
            for (var cid = node_info.z; cid < end; cid = cid + 1u) {
                let ec = cull_clusters[cid].errors_and_cone;
                let pe = bitcast<f32>(ec.y);
                if pe < 1.0e29 {
                    let pls = cull_clusters[cid].parent_lod_sphere;
                    let pc = inst_world_pos(inst_pos_scale, inst_rot, pls.xyz);
                    let p_dist = max(length(cull_pos - pc) - pls.w * scale, near_z);
                    if pe * err_factor <= lod_thresh * p_dist {
                        continue;
                    }
                }
                let se = bitcast<f32>(ec.x);
                if se > 0.0 {
                    let ls = cull_clusters[cid].lod_sphere;
                    let sc = inst_world_pos(inst_pos_scale, inst_rot, ls.xyz);
                    let s_dist = max(length(cull_pos - sc) - ls.w * scale, near_z);
                    if se * err_factor > lod_thresh * s_dist {
                        continue;
                    }
                }
                local_tested = local_tested + 1u;
                let bs = cull_clusters[cid].bound_sphere;
                let world_c = inst_world_pos(inst_pos_scale, inst_rot, bs.xyz);
                let world_r = bs.w * scale;
                if !cull_frustum_visible(world_c, world_r) {
                    local_frustum = local_frustum + 1u;
                    continue;
                }
                let mat_bit = 1u << ((ec.w >> 24u) & 31u);
                if (ds_mask & mat_bit) == 0u
                    && cone_culled_packed(ec.z, inst_rot, world_c, world_r, cull_pos)
                {
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

        let child_cnt = min(node_info.x, 4u);
        for (var k = 0u; k < child_cnt; k = k + 1u) {
            let child_id = cull_bvh[node_id].child_or_cluster[k];
            if child_id == NO_CHILD {
                continue;
            }
            if !bvh_child_visible(
                node_id,
                k,
                child_id,
                inst_pos_scale,
                inst_rot,
                cull_pos,
                near_z,
                err_factor,
                lod_thresh,
            ) {
                continue;
            }
            if top < 32u {
                stack[top] = child_id;
                top = top + 1u;
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

fn emit_visible_cluster(
    inst_id: u32,
    draw_cid: u32,
    slot: u32,
    page_id: u32,
    c_meta: u32,
    dc_world_c: vec3<f32>,
    dc_world_r: f32,
    is_p2: bool,
    is_fallback: u32,
) {
    atomicMax(&cull_page_requests[page_id], 1u);

    if is_fallback != 0u {
        atomicAdd(&cull_state[11u], 1u);
    }

    let tc = (c_meta >> 8u) & 0xFFu;
    var is_sw = false;
    if cull_uniforms.flags.y == 0u {
        let rvp = cull_uniforms.render_view_proj;
        let clip_w = rvp[0].w * dc_world_c.x
            + rvp[1].w * dc_world_c.y
            + rvp[2].w * dc_world_c.z
            + rvp[3].w;
        let near_z = cull_uniforms.cull_pos_near.w;
        if clip_w > dc_world_r * 1.45 + near_z * 2.0 {
            let render_lod_scale = max(cull_uniforms.shadow_info.z, 1.0);
            let denom = max(clip_w - dc_world_r, near_z);
            let num = 2.0 * dc_world_r * render_lod_scale;
            is_sw = num <= 16.0 * denom
                && num * inverseSqrt(max(f32(tc), 1.0)) <= cull_uniforms.hiz_params.w * denom;
        }
    }

    let vis_idx = atomicAdd(&cull_state[25u], 1u);
    if vis_idx >= MAX_VIS_CLUSTERS {
        return;
    }
    let vflags = select(0u, 1u, is_sw)
        | select(0u, 2u, is_p2)
        | (is_fallback << 2u)
        | ((c_meta >> 8u) & 0xFF00u);
    cull_vis_clusters[vis_idx] = VisibleCluster(inst_id, draw_cid, slot, vflags);
    atomicAdd(&cull_state[15u], 1u);

    let item = VisibleCluster(inst_id, draw_cid, slot, vis_idx);
    if is_sw {
        var sw_cnt_idx = 22u;
        var sw_stat_idx = 5u;
        if is_p2 {
            sw_cnt_idx = 24u;
            sw_stat_idx = 7u;
        }
        let sw_idx = atomicAdd(&cull_state[sw_cnt_idx], 1u);
        if sw_idx < MAX_PHASE_CLUSTERS {
            cull_draw_lists[MAX_PHASE_CLUSTERS * 2u + sw_idx] = item;
        }
        atomicAdd(&cull_state[sw_stat_idx], 1u);
        atomicAdd(&cull_state[9u], tc);
    } else {
        let mat_bit = 1u << ((c_meta >> 24u) & 31u);
        if (cull_uniforms.extra.x & mat_bit) != 0u {
            var ds_cnt_idx = 26u;
            if is_p2 {
                ds_cnt_idx = 27u;
            }
            let ds_idx = atomicAdd(&cull_state[ds_cnt_idx], 1u);
            if ds_idx < MAX_PHASE_CLUSTERS {
                cull_draw_lists[MAX_PHASE_CLUSTERS + ds_idx] = item;
            }
        } else {
            var hw_cnt_idx = 21u;
            if is_p2 {
                hw_cnt_idx = 23u;
            }
            let hw_idx = atomicAdd(&cull_state[hw_cnt_idx], 1u);
            if hw_idx < MAX_PHASE_CLUSTERS {
                cull_draw_lists[hw_idx] = item;
            }
        }
        var hw_stat_idx = 4u;
        if is_p2 {
            hw_stat_idx = 6u;
        }
        atomicAdd(&cull_state[hw_stat_idx], 1u);
        atomicAdd(&cull_state[8u], tc);
    }
}

fn evaluate_cluster(inst_id: u32, cluster_id: u32, is_p2: bool) {
    let inst_pos_scale = cull_instances[inst_id].pos_scale;
    let inst_rot = cull_instances[inst_id].rot_quat;
    let scale = inst_pos_scale.w;
    let c_bs = cull_clusters[cluster_id].bound_sphere;
    let world_c = inst_world_pos(inst_pos_scale, inst_rot, c_bs.xyz);
    let world_r = c_bs.w * scale;

    if hiz_occluded_sphere(world_c, world_r, !is_p2) {
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

    let pf = cull_clusters[cluster_id].page_and_fallback;
    let orig_page = pf.x;
    let slot = cull_page_table[orig_page];
    let ec_w = cull_clusters[cluster_id].errors_and_cone.w;
    if slot != UNMAPPED_PAGE {
        emit_visible_cluster(inst_id, cluster_id, slot, orig_page, ec_w, world_c, world_r, is_p2, 0u);
        return;
    }

    let cull_pos = cull_uniforms.cull_pos_near.xyz;
    let near_z = cull_uniforms.cull_pos_near.w;
    let lod_scale = cull_uniforms.viewport_lod.z;
    var parent_proj = 1024.0;
    let pe = bitcast<f32>(cull_clusters[cluster_id].errors_and_cone.y);
    if pe < 1.0e29 {
        let pls = cull_clusters[cluster_id].parent_lod_sphere;
        let pc = inst_world_pos(inst_pos_scale, inst_rot, pls.xyz);
        let pr = pls.w * scale;
        parent_proj = projected_error(cull_pos, near_z, lod_scale, pc, pr, pe, scale);
    }
    let dist = max(length(cull_pos - world_c) - world_r, near_z);
    let fg_boost = clamp(12.0 / max(dist, 1.5), 1.0, 8.0);
    let weighted_proj = min(parent_proj * fg_boost, 65535.0);
    let prio = clamp(u32(weighted_proj * 64.0) + 1u, 2u, 0x7FFFFFFFu);
    atomicMax(&cull_page_requests[orig_page], prio);
    atomicAdd(&cull_state[10u], 1u);

    var stack: array<u32, 64>;
    var top: u32 = 0u;
    let fb_start = pf.z & 0x00FFFFFFu;
    let fb_count = pf.z >> 24u;
    for (var fi = 0u; fi < fb_count && top < 64u; fi = fi + 1u) {
        stack[top] = fb_start + fi;
        top = top + 1u;
    }
    for (var step = 0u; step < 1024u; step = step + 1u) {
        if top == 0u {
            break;
        }
        top = top - 1u;
        let cur = stack[top];
        if !try_claim_dedup(inst_id, cur) {
            continue;
        }
        let f_pf = cull_clusters[cur].page_and_fallback;
        let fp = f_pf.x;
        let fs = cull_page_table[fp];
        if fs != UNMAPPED_PAGE {
            let f_bs = cull_clusters[cur].bound_sphere;
            let f_meta = cull_clusters[cur].errors_and_cone.w;
            let f_wc = inst_world_pos(inst_pos_scale, inst_rot, f_bs.xyz);
            let f_wr = f_bs.w * scale;
            emit_visible_cluster(inst_id, cur, fs, fp, f_meta, f_wc, f_wr, is_p2, 1u);
        } else {
            let ps = f_pf.z & 0x00FFFFFFu;
            let pc = f_pf.z >> 24u;
            for (var fi = 0u; fi < pc && top < 64u; fi = fi + 1u) {
                stack[top] = ps + fi;
                top = top + 1u;
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
    let ds_cnt = min(atomicLoad(&cull_state[26u]), MAX_PHASE_CLUSTERS);
    let sw_cnt = min(atomicLoad(&cull_state[22u]), MAX_PHASE_CLUSTERS);
    let occ_i = min(atomicLoad(&cull_state[19u]), MAX_OCC_INSTANCES);
    let occ_c = min(atomicLoad(&cull_state[20u]), MAX_OCC_CLUSTERS);
    let sw_x = min(sw_cnt, SW_DISPATCH_STRIDE);
    let sw_y = select(0u, (sw_cnt + SW_DISPATCH_STRIDE - 1u) / SW_DISPATCH_STRIDE, sw_cnt > 0u);
    cull_draw_lists[MAX_PHASE_CLUSTERS * 3u] = VisibleCluster(hw_cnt, ds_cnt, sw_cnt, 0u);
    cull_indirect[0] = CLUSTER_DRAW_VERTS;
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
    cull_indirect[28] = CLUSTER_DRAW_VERTS;
    cull_indirect[29] = ds_cnt;
    cull_indirect[30] = 0u;
    cull_indirect[31] = 0u;
}

@compute @workgroup_size(1)
fn cs_prep_draws_p2() {
    let hw_cnt = min(atomicLoad(&cull_state[23u]), MAX_PHASE_CLUSTERS);
    let ds_cnt = min(atomicLoad(&cull_state[27u]), MAX_PHASE_CLUSTERS);
    let sw_cnt = min(atomicLoad(&cull_state[24u]), MAX_PHASE_CLUSTERS);
    let sw_x = min(sw_cnt, SW_DISPATCH_STRIDE);
    let sw_y = select(0u, (sw_cnt + SW_DISPATCH_STRIDE - 1u) / SW_DISPATCH_STRIDE, sw_cnt > 0u);
    cull_draw_lists[MAX_PHASE_CLUSTERS * 3u] = VisibleCluster(hw_cnt, ds_cnt, sw_cnt, 0u);
    cull_indirect[0] = CLUSTER_DRAW_VERTS;
    cull_indirect[1] = hw_cnt;
    cull_indirect[2] = 0u;
    cull_indirect[3] = 0u;
    cull_indirect[4] = sw_x;
    cull_indirect[5] = sw_y;
    cull_indirect[6] = 1u;
    cull_indirect[7] = 0u;
    cull_indirect[28] = CLUSTER_DRAW_VERTS;
    cull_indirect[29] = ds_cnt;
    cull_indirect[30] = 0u;
    cull_indirect[31] = 0u;
}
