import package::nanite::frame::nanite_frame_uniforms;
import package::nanite::frame::nanite_frame_instances;
import package::nanite::frame::nanite_frame_clusters;
import package::nanite::frame::nanite_frame_page_heap;
import package::nanite::frame::nanite_frame_vis_clusters;
import package::nanite::frame::nanite_frame_sw_depth;
import package::nanite::frame::nanite_frame_sw_vis;
import package::nanite::frame::nanite_frame_depth;
import package::nanite::frame::nanite_frame_shadow_c0;
import package::nanite::frame::nanite_frame_shadow_c1;
import package::nanite::common::PAGE_WORDS;
import package::nanite::common::CLUSTER_HEADER_WORDS;
import package::nanite::common::quat_rotate;
import package::nanite::common::decode_qpos_u16;
import package::nanite::common::decode_octahedral_10;
import package::nanite::common::decode_uv_12;
import package::nanite::common::ray_tri_bary_uv;
import package::nanite::common::clamp_uv_grad;
import package::nanite::common::build_tangent_frame;
import package::nanite::common::hash_color;
import package::nanite::common::lod_heat_color;

struct NaniteVisHit {
    vis_id: u32,
    depth: f32,
    use_sw: bool,
}

struct NaniteSurface {
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    geom_n: vec3<f32>,
    tangent: vec4<f32>,
    uv: vec2<f32>,
    duv_dx: vec2<f32>,
    duv_dy: vec2<f32>,
    vertex_ao: f32,
    normal_len2: f32,
    instance_id: u32,
    instance_custom_id: u32,
    cluster_id: u32,
    tri_id: u32,
    material_id: u32,
    lod_level: u32,
    page_id: u32,
    is_root: bool,
    use_sw: bool,
}

fn nanite_read_u24(base_word: u32, idx: u32) -> u32 {
    let byte_off = idx * 3u;
    let wi = base_word + (byte_off >> 2u);
    let rem = byte_off & 3u;
    let w0 = nanite_frame_page_heap[wi];
    let w1 = nanite_frame_page_heap[wi + 1u];
    let lo = w0 >> (rem * 8u);
    let hi_shift = ((4u - rem) & 3u) * 8u;
    let hi_mask = select(0u, 0xFFFFFFFFu, rem >= 2u);
    return (lo | ((w1 << hi_shift) & hi_mask)) & 0xFFFFFFu;
}

fn nanite_read_qpos_u16(pos_base: u32, vi: u32) -> vec3<f32> {
    let pwi = pos_base + ((vi * 3u) >> 1u);
    return decode_qpos_u16(
        nanite_frame_page_heap[pwi],
        nanite_frame_page_heap[pwi + 1u],
        vi,
    );
}

fn nanite_load_depth(px: vec2<i32>, size_x: u32) -> f32 {
    var d = textureLoad(nanite_frame_depth, px, 0);
    if nanite_frame_uniforms.flags.y == 0u {
        let idx = u32(px.y) * size_x + u32(px.x);
        let sw_d = bitcast<f32>(nanite_frame_sw_depth[idx]);
        if sw_d > d {
            d = sw_d;
        }
    }
    return d;
}

fn nanite_load_vis_hit(vis_hw_tex: texture_2d<u32>, px: vec2<u32>, size_x: u32) -> NaniteVisHit {
    let hw_d = textureLoad(nanite_frame_depth, vec2<i32>(px), 0);
    let hw_v = textureLoad(vis_hw_tex, vec2<i32>(px), 0).r;
    var vis_id = hw_v;
    var win_depth = hw_d;
    var use_sw = false;
    if nanite_frame_uniforms.flags.y == 0u {
        let idx = px.y * size_x + px.x;
        let sw_d = bitcast<f32>(nanite_frame_sw_depth[idx]);
        let sw_v = nanite_frame_sw_vis[idx];
        use_sw = (sw_v != 0u) && (sw_d > hw_d || hw_v == 0u);
        vis_id = select(hw_v, sw_v, use_sw);
        win_depth = select(hw_d, sw_d, use_sw);
    }
    return NaniteVisHit(vis_id, win_depth, use_sw);
}

fn nanite_decode_surface(hit: NaniteVisHit, frag_xy: vec2<f32>, vp: vec2<f32>) -> NaniteSurface {
    let cam_pos = nanite_frame_uniforms.render_pos_time.xyz;
    let inv_vp = nanite_frame_uniforms.inv_render_vp;
    let vis_slot = (hit.vis_id >> 7u) - 1u;
    let tri_id = hit.vis_id & 0x7Fu;
    let inst_id = nanite_frame_vis_clusters[vis_slot].instance_id;
    let cluster_id = nanite_frame_vis_clusters[vis_slot].cluster_id;
    let heap_slot = nanite_frame_vis_clusters[vis_slot].heap_slot;
    let ec_w = nanite_frame_clusters[cluster_id].errors_and_cone.w;
    let pf = nanite_frame_clusters[cluster_id].page_and_fallback;
    let word_off = pf.y & 0xFFFFu;
    let page_base = heap_slot * PAGE_WORDS + word_off;
    let pos_base = page_base + CLUSTER_HEADER_WORDS;
    let vert_count = ec_w & 0xFFu;
    let tri_count = (ec_w >> 8u) & 0xFFu;
    let tri_word = nanite_read_u24(pos_base + ((vert_count * 3u + 1u) >> 1u), tri_id);

    let origin = vec3<f32>(
        bitcast<f32>(nanite_frame_page_heap[page_base]),
        bitcast<f32>(nanite_frame_page_heap[page_base + 1u]),
        bitcast<f32>(nanite_frame_page_heap[page_base + 2u]),
    );
    let inst_ps = nanite_frame_instances[inst_id].pos_scale;
    let inst_rq = nanite_frame_instances[inst_id].rot_quat;
    let inst_custom = nanite_frame_instances[inst_id].info.y;
    let q_scale = bitcast<f32>(nanite_frame_page_heap[page_base + 3u]) * inst_ps.w;

    let max_vi = max(vert_count, 1u) - 1u;
    let i0 = min(tri_word & 0xFFu, max_vi);
    let i1 = min((tri_word >> 8u) & 0xFFu, max_vi);
    let i2 = min((tri_word >> 16u) & 0xFFu, max_vi);
    let q0 = nanite_read_qpos_u16(pos_base, i0);
    let p0 = inst_ps.xyz + quat_rotate(inst_rq, origin * inst_ps.w + q0 * q_scale);
    let e1 = quat_rotate(inst_rq, (nanite_read_qpos_u16(pos_base, i1) - q0) * q_scale);
    let e2 = quat_rotate(inst_rq, (nanite_read_qpos_u16(pos_base, i2) - q0) * q_scale);

    let o = cam_pos - p0;
    let d_vec = cross(e2, e1);
    let m_vec = cross(e2, o);
    let q_vec = cross(o, e1);
    let ndc_p = vec2<f32>((frag_xy.x / vp.x) * 2.0 - 1.0, 1.0 - (frag_xy.y / vp.y) * 2.0);
    let h_c = inv_vp * vec4<f32>(ndc_p, 0.5, 1.0);
    let r_c = h_c.xyz - cam_pos * h_c.w;
    let dr_x = (inv_vp[0].xyz - cam_pos * inv_vp[0].w) * (2.0 / vp.x);
    let dr_y = -(inv_vp[1].xyz - cam_pos * inv_vp[1].w) * (2.0 / vp.y);
    let bary_uv = ray_tri_bary_uv(d_vec, m_vec, q_vec, r_c);
    let db_dx = ray_tri_bary_uv(d_vec, m_vec, q_vec, r_c + dr_x) - bary_uv;
    let db_dy = ray_tri_bary_uv(d_vec, m_vec, q_vec, r_c + dr_y) - bary_uv;

    let bary_w = 1.0 - bary_uv.x - bary_uv.y;
    let world_pos = p0 + e1 * bary_uv.x + e2 * bary_uv.y;
    let norm_base = pos_base + ((vert_count * 3u + 1u) >> 1u) + ((tri_count * 3u + 3u) >> 2u);
    let n0_p = nanite_read_u24(norm_base, i0);
    let n1_p = nanite_read_u24(norm_base, i1);
    let n2_p = nanite_read_u24(norm_base, i2);
    let local_n = decode_octahedral_10(n0_p) * bary_w
        + decode_octahedral_10(n1_p) * bary_uv.x
        + decode_octahedral_10(n2_p) * bary_uv.y;
    let ao_raw = f32((n0_p >> 20u) & 0xFu) * bary_w
        + f32((n1_p >> 20u) & 0xFu) * bary_uv.x
        + f32((n2_p >> 20u) & 0xFu) * bary_uv.y;
    let ln_l2 = dot(local_n, local_n);
    let world_n_raw = quat_rotate(inst_rq, local_n);
    let wn_l2 = dot(world_n_raw, world_n_raw);
    let world_n = select(vec3<f32>(0.0, 1.0, 0.0), world_n_raw * inverseSqrt(wn_l2), wn_l2 > 1.0e-8);
    let geom_n = select(world_n, -world_n, dot(world_n, cam_pos - world_pos) < 0.0);

    let uv_base = norm_base + ((vert_count * 3u + 3u) >> 2u);
    let uv_origin = vec2<f32>(
        bitcast<f32>(nanite_frame_page_heap[page_base + 4u]),
        bitcast<f32>(nanite_frame_page_heap[page_base + 5u]),
    );
    let uv_step = bitcast<f32>(nanite_frame_page_heap[page_base + 6u]);
    let uv0 = decode_uv_12(nanite_read_u24(uv_base, i0), uv_origin, uv_step);
    let du1 = decode_uv_12(nanite_read_u24(uv_base, i1), uv_origin, uv_step) - uv0;
    let du2 = decode_uv_12(nanite_read_u24(uv_base, i2), uv_origin, uv_step) - uv0;
    let uv_interp = uv0 + du1 * bary_uv.x + du2 * bary_uv.y;
    let duv_dx = clamp_uv_grad(du1 * db_dx.x + du2 * db_dx.y);
    let duv_dy = clamp_uv_grad(du1 * db_dy.x + du2 * db_dy.y);
    let t_frame = build_tangent_frame(e1, e2, du1, du2, geom_n);

    return NaniteSurface(
        world_pos,
        world_n,
        geom_n,
        t_frame,
        uv_interp,
        duv_dx,
        duv_dy,
        ao_raw * (1.0 / 15.0),
        ln_l2,
        inst_id,
        inst_custom,
        cluster_id,
        tri_id,
        ec_w >> 24u,
        (ec_w >> 16u) & 0xFFu,
        pf.x,
        ((pf.y >> 16u) & 1u) == 1u,
        hit.use_sw,
    );
}

fn nanite_debug_color(view_mode: u32, s: NaniteSurface, light_dir: vec3<f32>) -> vec3<f32> {
    let ndl = 0.42 + 0.58 * clamp(dot(s.world_n, light_dir) * 0.5 + 0.5, 0.0, 1.0);
    if view_mode == 1u {
        return hash_color(s.cluster_id ^ (s.instance_id * 1664525u + 1013904223u)) * ndl;
    }
    if view_mode == 2u {
        return hash_color((s.cluster_id * 128u + s.tri_id) ^ (s.instance_id * 2246822519u)) * ndl;
    }
    if view_mode == 3u {
        return lod_heat_color(s.lod_level) * ndl;
    }
    if view_mode == 4u {
        let hw_col = vec3<f32>(0.06, 0.80, 0.96);
        let sw_col = vec3<f32>(0.98, 0.52, 0.08);
        return select(hw_col, sw_col, s.use_sw) * ndl;
    }
    let pcol = select(hash_color(s.page_id + 17u), vec3<f32>(0.98, 0.25, 0.72), s.is_root);
    return pcol * ndl;
}

fn nanite_sample_cascade(
    shadow_tex: texture_depth_2d,
    shadow_sampler: sampler_comparison,
    shadow_vp: mat4x4<f32>,
    tex_dim: f32,
    radius_texels: f32,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    normal_bias: f32,
    depth_bias: f32,
    cs: vec2<f32>,
) -> vec2<f32> {
    let p = world_pos + n * normal_bias;
    let clip = shadow_vp * vec4<f32>(p, 1.0);
    let margin = max(abs(clip.x), abs(clip.y));
    if margin >= 0.98 || clip.z <= 0.0005 || clip.z >= 0.9995 {
        return vec2<f32>(1.0, 0.0);
    }
    let uv = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
    let cmp_z = clamp(clip.z + depth_bias, 0.0, 1.0);
    let r_cs = cs * (radius_texels / max(tex_dim, 1.0));
    let r_cs_p = vec2<f32>(-r_cs.y, r_cs.x);
    var acc = textureSampleCompareLevel(shadow_tex, shadow_sampler, uv + r_cs * 0.35355339, cmp_z);
    acc += textureSampleCompareLevel(shadow_tex, shadow_sampler, uv + r_cs * -0.45154436 + r_cs_p * 0.41365165, cmp_z);
    acc += textureSampleCompareLevel(shadow_tex, shadow_sampler, uv + r_cs * 0.06911606 + r_cs_p * -0.78754236, cmp_z);
    acc += textureSampleCompareLevel(shadow_tex, shadow_sampler, uv + r_cs * 0.56914251 + r_cs_p * 0.74234548, cmp_z);
    let coverage = smoothstep(0.98, 0.84, margin);
    return vec2<f32>(acc * 0.25, coverage);
}

fn nanite_sample_shadow(
    world_pos: vec3<f32>,
    geom_n: vec3<f32>,
    shadow_sampler: sampler_comparison,
    light_dir: vec3<f32>,
    rot_01: f32,
) -> f32 {
    if nanite_frame_uniforms.shadow_info.w < 0.5 {
        return 1.0;
    }
    let rot = rot_01 * 6.2831853;
    let cs = vec2<f32>(cos(rot), sin(rot));
    let ndl = max(dot(geom_n, light_dir), 0.0);
    let slope = clamp(1.0 - ndl, 0.0, 1.0);
    let c0 = nanite_sample_cascade(
        nanite_frame_shadow_c0,
        shadow_sampler,
        nanite_frame_uniforms.shadow_vp0,
        nanite_frame_uniforms.shadow_info.x,
        1.65,
        world_pos,
        geom_n,
        0.014 + slope * 0.022,
        0.00022 + slope * 0.00032,
        cs,
    );
    if c0.y >= 0.99 {
        return c0.x;
    }
    let c1 = nanite_sample_cascade(
        nanite_frame_shadow_c1,
        shadow_sampler,
        nanite_frame_uniforms.shadow_vp1,
        nanite_frame_uniforms.shadow_info.y,
        1.85,
        world_pos,
        geom_n,
        0.028 + slope * 0.042,
        0.00036 + slope * 0.00050,
        cs,
    );
    let far_val = mix(1.0, c1.x, c1.y);
    return mix(far_val, c0.x, c0.y);
}
