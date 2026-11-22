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
import package::nanite::common::inst_to_world;
import package::nanite::common::decode_pos;
import package::nanite::common::cluster_vert_count;
import package::nanite::common::cluster_tri_count;
import package::nanite::common::cluster_page_id;
import package::nanite::common::cluster_word_offset;
import package::nanite::common::cluster_fb_start;
import package::nanite::common::cluster_fb_count;

const SH_DEDUP_BASE: u32 = 528416u;

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
    let key = ((inst_id + 1u) << 21u) | (cluster_id & 0x1FFFFFu);
    var h = (inst_id * 2654435761u) ^ (cluster_id * 2246822519u);
    h = (h ^ (h >> 16u)) * 0x45D9F3Bu;
    var slot = h & DEDUP_MASK;
    for (var probe = 0u; probe < 32u; probe = probe + 1u) {
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
    let inst = sh_instances[inst_id];
    let c0 = sh_clusters[cid];
    var vp = sh_uniforms.shadow_vp0;
    if cascade != 0u {
        vp = sh_uniforms.shadow_vp1;
    }
    let half_ext = select(sh_uniforms.shadow_center0.w, sh_uniforms.shadow_center1.w, cascade != 0u);
    let world_c = inst_to_world(inst, c0.bound_sphere.xyz);
    let world_r = c0.bound_sphere.w * inst.pos_scale.w;
    let clip_c = vp * vec4<f32>(world_c, 1.0);
    let r_ndc = world_r / max(half_ext, 1.0);
    if abs(clip_c.x) > 1.0 + r_ndc || abs(clip_c.y) > 1.0 + r_ndc {
        return;
    }
    let list_base = select(0u, SHADOW_LIST_CAP, cascade != 0u);
    let counter_idx = select(8u, 9u, cascade != 0u);
    let slot = sh_page_table[cluster_page_id(c0)];
    if slot != UNMAPPED_PAGE {
        if try_claim_sh_dedup(inst_id, cid) {
            let out_idx = atomicAdd(&sh_indirect[counter_idx], 1u);
            if out_idx < SHADOW_LIST_CAP {
                sh_list[list_base + out_idx] = VisibleCluster(inst_id, cid, slot, cascade);
            }
        }
        return;
    }
    var fb_stack: array<u32, 128>;
    var sp: u32 = 0u;
    let fb_s = cluster_fb_start(c0);
    let fb_c = cluster_fb_count(c0);
    for (var fi = 0u; fi < fb_c && sp < 128u; fi = fi + 1u) {
        fb_stack[sp] = fb_s + fi;
        sp = sp + 1u;
    }
    for (var s = 0u; s < 1024u; s = s + 1u) {
        if sp == 0u {
            break;
        }
        sp = sp - 1u;
        let cur = fb_stack[sp];
        if !try_claim_sh_dedup(inst_id, cur) {
            continue;
        }
        let fc = sh_clusters[cur];
        let fs = sh_page_table[cluster_page_id(fc)];
        if fs != UNMAPPED_PAGE {
            let out_idx = atomicAdd(&sh_indirect[counter_idx], 1u);
            if out_idx < SHADOW_LIST_CAP {
                sh_list[list_base + out_idx] = VisibleCluster(inst_id, cur, fs, cascade);
            }
        } else {
            let ps = cluster_fb_start(fc);
            let pc = cluster_fb_count(fc);
            for (var fi = 0u; fi < pc && sp < 128u; fi = fi + 1u) {
                fb_stack[sp] = ps + fi;
                sp = sp + 1u;
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
    let vc = sh_list_ro[list_idx];
    let c = sh_clusters[vc.cluster_id];
    let tri_id = vid / 3u;
    let tri_count = cluster_tri_count(c);
    if tri_id >= tri_count {
        return vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    let corner = vid - tri_id * 3u;
    let vert_count = cluster_vert_count(c);
    let page_base = vc.heap_slot * PAGE_WORDS + cluster_word_offset(c);
    let origin = vec3<f32>(
        bitcast<f32>(sh_page_heap[page_base]),
        bitcast<f32>(sh_page_heap[page_base + 1u]),
        bitcast<f32>(sh_page_heap[page_base + 2u]),
    );
    let step = bitcast<f32>(sh_page_heap[page_base + 3u]);
    let verts_base = page_base + CLUSTER_HEADER_WORDS;
    let tri_word = sh_page_heap[verts_base + vert_count * 4u + tri_id];
    let local_vi = (tri_word >> (corner * 8u)) & 0xFFu;
    let v_base = verts_base + local_vi * 4u;
    let w0 = sh_page_heap[v_base];
    let w1 = sh_page_heap[v_base + 1u];
    let local_pos = decode_pos(w0, w1, origin, step);
    let inst = sh_instances[vc.instance_id];
    let world_pos = inst_to_world(inst, local_pos);
    var vp = sh_uniforms.shadow_vp0;
    if cascade != 0u {
        vp = sh_uniforms.shadow_vp1;
    }
    return vp * vec4<f32>(world_pos, 1.0);
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
