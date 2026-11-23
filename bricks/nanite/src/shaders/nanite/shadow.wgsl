import package::generated::nanite::NaniteUniforms;
import package::generated::nanite::NaniteInstance;
import package::generated::nanite::GpuClusterHeader;
import package::generated::nanite::VisibleCluster;
import package::nanite::common::PAGE_WORDS;
import package::nanite::common::CLUSTER_HEADER_WORDS;
import package::nanite::common::CLUSTER_DRAW_VERTS;
import package::nanite::common::DEDUP_MASK;
import package::nanite::common::UNMAPPED_PAGE;
import package::nanite::common::SHADOW_LIST_CAP;
import package::nanite::common::transform_pos;
import package::nanite::common::decode_pos_u16;
import package::nanite::common::dedup_hash_pair;

const SH_DEDUP_BASE: u32 = 589856u;

@group(1) @binding(0) var<uniform> sh_uniforms: NaniteUniforms;
@group(1) @binding(1) var<storage, read> sh_instances: array<NaniteInstance>;
@group(1) @binding(2) var<storage, read> sh_clusters: array<GpuClusterHeader>;
@group(1) @binding(3) var<storage, read> sh_page_table: array<u32>;
@group(1) @binding(4) var<storage, read> sh_page_heap: array<u32>;
@group(1) @binding(5) var<storage, read> sh_jobs: array<vec2<u32>>;
@group(1) @binding(6) var<storage, read_write> sh_dedup: array<atomic<u32>>;
@group(1) @binding(7) var<storage, read_write> sh_list: array<VisibleCluster>;
@group(1) @binding(8) var<storage, read_write> sh_indirect: array<atomic<u32>>;
@group(1) @binding(9) var<storage, read> sh_list_ro: array<VisibleCluster>;

fn try_claim_sh_dedup(inst_id: u32, cluster_id: u32) -> bool {
    let ht = dedup_hash_pair(inst_id, cluster_id);
    var slot = ht.x;
    let tag_base = ht.y << 5u;
    for (var probe = 0u; probe < 32u; probe = probe + 1u) {
        let key = tag_base | probe;
        let res = atomicCompareExchangeWeak(&sh_dedup[SH_DEDUP_BASE + slot], 0u, key);
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

fn shadow_collect(job_idx: u32, cascade: u32) {
    let c0_cnt = sh_uniforms.extra.y;
    let c1_cnt = sh_uniforms.extra.z;
    let count = select(c0_cnt, c1_cnt, cascade != 0u);
    if job_idx >= count {
        return;
    }
    let job_base = select(0u, c0_cnt, cascade != 0u);
    let job = sh_jobs[job_base + job_idx];
    let inst_id = job.x;
    let cid = job.y;
    let inst_pos_scale = sh_instances[inst_id].pos_scale;
    let inst_rot = sh_instances[inst_id].rot_quat;
    let c_bs = sh_clusters[cid].bound_sphere;
    let world_c = transform_pos(inst_pos_scale, inst_rot, c_bs.xyz);
    let world_r = c_bs.w * inst_pos_scale.w;
    var clip_xy: vec2<f32>;
    if cascade != 0u {
        let vp = sh_uniforms.shadow_vp1;
        clip_xy = vp[0].xy * world_c.x + vp[1].xy * world_c.y + vp[2].xy * world_c.z + vp[3].xy;
    } else {
        let vp = sh_uniforms.shadow_vp0;
        clip_xy = vp[0].xy * world_c.x + vp[1].xy * world_c.y + vp[2].xy * world_c.z + vp[3].xy;
    }
    let half_ext = select(sh_uniforms.shadow_center0.w, sh_uniforms.shadow_center1.w, cascade != 0u);
    let safe_ext = max(half_ext, 1.0);
    if max(abs(clip_xy.x), abs(clip_xy.y)) * safe_ext > safe_ext + world_r {
        return;
    }
    let list_base = select(0u, SHADOW_LIST_CAP, cascade != 0u);
    let counter_idx = select(8u, 9u, cascade != 0u);
    let pf = sh_clusters[cid].page_and_fallback;
    let slot = sh_page_table[pf.x];
    if slot != UNMAPPED_PAGE {
        if try_claim_sh_dedup(inst_id, cid) {
            let out_idx = atomicAdd(&sh_indirect[counter_idx], 1u);
            if out_idx < SHADOW_LIST_CAP {
                sh_list[list_base + out_idx] = VisibleCluster(inst_id, cid, slot, cascade);
            }
        }
        return;
    }
    var stack: array<u32, 64>;
    var top: u32 = 0u;
    let fb_s = pf.z & 0x00FFFFFFu;
    let fb_c = pf.z >> 24u;
    for (var fi = 0u; fi < fb_c && top < 64u; fi = fi + 1u) {
        stack[top] = fb_s + fi;
        top = top + 1u;
    }
    for (var s = 0u; s < 1024u; s = s + 1u) {
        if top == 0u {
            break;
        }
        top = top - 1u;
        let cur = stack[top];
        if !try_claim_sh_dedup(inst_id, cur) {
            continue;
        }
        let f_pf = sh_clusters[cur].page_and_fallback;
        let fs = sh_page_table[f_pf.x];
        if fs != UNMAPPED_PAGE {
            let out_idx = atomicAdd(&sh_indirect[counter_idx], 1u);
            if out_idx < SHADOW_LIST_CAP {
                sh_list[list_base + out_idx] = VisibleCluster(inst_id, cur, fs, cascade);
            }
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
fn cs_shadow_collect_c0(@builtin(global_invocation_id) gid: vec3<u32>) {
    shadow_collect(gid.x, 0u);
}

@compute @workgroup_size(64)
fn cs_shadow_collect_c1(@builtin(global_invocation_id) gid: vec3<u32>) {
    shadow_collect(gid.x, 1u);
}

@compute @workgroup_size(1)
fn cs_shadow_prep() {
    let c0_cnt = min(atomicLoad(&sh_indirect[8u]), SHADOW_LIST_CAP);
    let c1_cnt = min(atomicLoad(&sh_indirect[9u]), SHADOW_LIST_CAP);
    atomicStore(&sh_indirect[0u], CLUSTER_DRAW_VERTS);
    atomicStore(&sh_indirect[1u], c0_cnt);
    atomicStore(&sh_indirect[2u], 0u);
    atomicStore(&sh_indirect[3u], 0u);
    atomicStore(&sh_indirect[4u], CLUSTER_DRAW_VERTS);
    atomicStore(&sh_indirect[5u], c1_cnt);
    atomicStore(&sh_indirect[6u], 0u);
    atomicStore(&sh_indirect[7u], 0u);
}

fn shadow_vertex(vid: u32, list_idx: u32, cascade: u32) -> vec4<f32> {
    let inst_id = sh_list_ro[list_idx].instance_id;
    let cluster_id = sh_list_ro[list_idx].cluster_id;
    let heap_slot = sh_list_ro[list_idx].heap_slot;
    let tri_id = vid / 3u;
    let ec_w = sh_clusters[cluster_id].errors_and_cone.w;
    let tri_count = (ec_w >> 8u) & 0xFFu;
    if tri_id >= tri_count {
        return vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    let vert_count = ec_w & 0xFFu;
    let word_off = sh_clusters[cluster_id].page_and_fallback.y & 0xFFFFu;
    let page_base = heap_slot * PAGE_WORDS + word_off;
    let origin = vec3<f32>(
        bitcast<f32>(sh_page_heap[page_base]),
        bitcast<f32>(sh_page_heap[page_base + 1u]),
        bitcast<f32>(sh_page_heap[page_base + 2u]),
    );
    let step = bitcast<f32>(sh_page_heap[page_base + 3u]);
    let pos_base = page_base + CLUSTER_HEADER_WORDS;
    let tri_base = pos_base + ((vert_count * 3u + 1u) >> 1u);
    let tri_word = sh_page_heap[tri_base + (vid >> 2u)];
    let raw_vi = (tri_word >> ((vid & 3u) * 8u)) & 0xFFu;
    let local_vi = min(raw_vi, max(vert_count, 1u) - 1u);
    let pwi = pos_base + ((local_vi * 3u) >> 1u);
    let local_pos = decode_pos_u16(
        sh_page_heap[pwi],
        sh_page_heap[pwi + 1u],
        local_vi,
        origin,
        step,
    );
    let pos_scale = sh_instances[inst_id].pos_scale;
    let rot_quat = sh_instances[inst_id].rot_quat;
    let world_pos = transform_pos(pos_scale, rot_quat, local_pos);
    if cascade == 1u {
        return sh_uniforms.shadow_vp1 * vec4<f32>(world_pos, 1.0);
    }
    return sh_uniforms.shadow_vp0 * vec4<f32>(world_pos, 1.0);
}

@vertex
fn vs_shadow_c0(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> @builtin(position) vec4<f32> {
    return shadow_vertex(vid, iid, 0u);
}

@vertex
fn vs_shadow_c1(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> @builtin(position) vec4<f32> {
    return shadow_vertex(vid, SHADOW_LIST_CAP + iid, 1u);
}
