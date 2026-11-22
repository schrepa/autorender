import package::generated::frame::FrameUniforms;
import package::nanite::frame::nanite_frame_uniforms;
import package::nanite::frame::nanite_frame_depth;
import package::nanite::frame::nanite_frame_sw_depth;

@group(2) @binding(0) var curr_hdr: texture_2d<f32>;
@group(2) @binding(1) var prev_hdr: texture_2d<f32>;
@group(2) @binding(2) var taa_sampler: sampler;
@group(2) @binding(3) var<uniform> frame_params: FrameUniforms;
@group(2) @binding(4) var taa_src: texture_2d<f32>;
@group(2) @binding(5) var taa_dst: texture_storage_2d<rgba16float, write>;

fn luma_hdr(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn karis_compress(c: vec3<f32>) -> vec3<f32> {
    let clamped = max(c, vec3<f32>(0.0));
    return clamped / (1.0 + luma_hdr(clamped));
}

fn karis_expand(c: vec3<f32>) -> vec3<f32> {
    let clamped = max(c, vec3<f32>(0.0));
    return clamped / max(1.0 - luma_hdr(clamped), 0.002);
}

fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    let co = rgb.r - rgb.b;
    let tmp = rgb.b + co * 0.5;
    let cg = rgb.g - tmp;
    let y = tmp + cg * 0.5;
    return vec3<f32>(y, co, cg);
}

fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    let tmp = ycocg.x - ycocg.z * 0.5;
    let g = ycocg.z + tmp;
    let b = tmp - ycocg.y * 0.5;
    let r = b + ycocg.y;
    return vec3<f32>(r, g, b);
}

fn sample_history_catmull_rom(uv: vec2<f32>, tex_size: vec2<f32>) -> vec3<f32> {
    let inv_size = 1.0 / tex_size;
    let sample_pos = uv * tex_size;
    let tc1 = floor(sample_pos - 0.5) + 0.5;
    let f = sample_pos - tc1;
    let f2 = f * f;
    let f3 = f2 * f;

    let w0 = f2 - 0.5 * (f3 + f);
    let w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    let w3 = 0.5 * (f3 - f2);
    let w2 = 1.0 - w0 - w1 - w3;

    let w12 = w1 + w2;
    let tc0 = (tc1 - 1.0) * inv_size;
    let tc3 = (tc1 + 2.0) * inv_size;
    let tc12 = (tc1 + w2 / max(w12, vec2<f32>(1.0e-5))) * inv_size;

    let c0 = textureSampleLevel(prev_hdr, taa_sampler, vec2<f32>(tc12.x, tc0.y), 0.0).rgb;
    let c1 = textureSampleLevel(prev_hdr, taa_sampler, vec2<f32>(tc0.x, tc12.y), 0.0).rgb;
    let c2 = textureSampleLevel(prev_hdr, taa_sampler, vec2<f32>(tc12.x, tc12.y), 0.0).rgb;
    let c3 = textureSampleLevel(prev_hdr, taa_sampler, vec2<f32>(tc3.x, tc12.y), 0.0).rgb;
    let c4 = textureSampleLevel(prev_hdr, taa_sampler, vec2<f32>(tc12.x, tc3.y), 0.0).rgb;

    let wt0 = w12.x * w0.y;
    let wt1 = w0.x * w12.y;
    let wt2 = w12.x * w12.y;
    let wt3 = w3.x * w12.y;
    let wt4 = w12.x * w3.y;
    let sum = c0 * wt0 + c1 * wt1 + c2 * wt2 + c3 * wt3 + c4 * wt4;
    let wsum = wt0 + wt1 + wt2 + wt3 + wt4;
    return max(sum / max(wsum, 1.0e-4), vec3<f32>(0.0));
}

fn clip_aabb(aabb_min: vec3<f32>, aabb_max: vec3<f32>, history: vec3<f32>) -> vec3<f32> {
    let center = 0.5 * (aabb_max + aabb_min);
    let extents = 0.5 * (aabb_max - aabb_min) + vec3<f32>(1.0e-5);
    let offset = history - center;
    let units = abs(offset / extents);
    let max_unit = max(units.x, max(units.y, units.z));
    if max_unit > 1.0 {
        return center + offset / max_unit;
    }
    return history;
}

fn load_scene_depth(px: vec2<i32>, size_x: u32) -> f32 {
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

@fragment
fn fs_taa(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size_u = max(textureDimensions(curr_hdr), vec2<u32>(1u));
    let size_i = vec2<i32>(size_u);
    let size_f = vec2<f32>(size_u);
    let px = clamp(vec2<i32>(pos.xy), vec2<i32>(0), size_i - vec2<i32>(1));
    let center_raw = textureLoad(curr_hdr, px, 0);

    if frame_params.fog_params.w > 0.5 || frame_params.taa_params.z > 0.5 {
        return center_raw;
    }

    var best_d = load_scene_depth(px, size_u.x);
    var best_px = px;
    let depth_offs = array<vec2<i32>, 4>(
        vec2<i32>(-1, -1),
        vec2<i32>(1, -1),
        vec2<i32>(-1, 1),
        vec2<i32>(1, 1),
    );
    for (var i = 0; i < 4; i = i + 1) {
        let sp = clamp(px + depth_offs[i], vec2<i32>(0), size_i - vec2<i32>(1));
        let d = load_scene_depth(sp, size_u.x);
        if d > best_d {
            best_d = d;
            best_px = sp;
        }
    }

    let uv = pos.xy / size_f;
    let best_uv = (vec2<f32>(best_px) + vec2<f32>(0.5)) / size_f;
    let best_ndc = vec2<f32>(best_uv.x * 2.0 - 1.0, 1.0 - best_uv.y * 2.0);
    var vel_uv = vec2<f32>(0.0);
    if best_d > 1.0e-5 {
        let h_w = nanite_frame_uniforms.inv_render_vp * vec4<f32>(best_ndc, best_d, 1.0);
        let world_p = h_w.xyz / max(abs(h_w.w), 1.0e-5) * sign(h_w.w);
        let c_clip = frame_params.curr_unjittered_vp * vec4<f32>(world_p, 1.0);
        let p_clip = frame_params.prev_unjittered_vp * vec4<f32>(world_p, 1.0);
        if c_clip.w > 1.0e-4 && p_clip.w > 1.0e-4 {
            let c_ndc = c_clip.xy / c_clip.w;
            let p_ndc = p_clip.xy / p_clip.w;
            let c_uv = vec2<f32>(c_ndc.x * 0.5 + 0.5, 0.5 - c_ndc.y * 0.5);
            let p_uv = vec2<f32>(p_ndc.x * 0.5 + 0.5, 0.5 - p_ndc.y * 0.5);
            vel_uv = c_uv - p_uv;
        }
    } else {
        let far_h = nanite_frame_uniforms.inv_render_vp * vec4<f32>(best_ndc, 0.0001, 1.0);
        let far_w = far_h.xyz / max(abs(far_h.w), 1.0e-5) * sign(far_h.w);
        let ray_d = normalize(far_w - nanite_frame_uniforms.render_pos_time.xyz);
        let c_clip = frame_params.curr_unjittered_vp * vec4<f32>(ray_d, 0.0);
        let p_clip = frame_params.prev_unjittered_vp * vec4<f32>(ray_d, 0.0);
        if c_clip.w > 1.0e-4 && p_clip.w > 1.0e-4 {
            let c_ndc = c_clip.xy / c_clip.w;
            let p_ndc = p_clip.xy / p_clip.w;
            let c_uv = vec2<f32>(c_ndc.x * 0.5 + 0.5, 0.5 - c_ndc.y * 0.5);
            let p_uv = vec2<f32>(p_ndc.x * 0.5 + 0.5, 0.5 - p_ndc.y * 0.5);
            vel_uv = c_uv - p_uv;
        }
    }

    let prev_uv = uv - vel_uv;
    if prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0 {
        return center_raw;
    }

    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    var n_min = vec3<f32>(1.0e6);
    var n_max = vec3<f32>(-1.0e6);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let sp = clamp(px + vec2<i32>(dx, dy), vec2<i32>(0), size_i - vec2<i32>(1));
            let c_k = karis_compress(textureLoad(curr_hdr, sp, 0).rgb);
            let ycc = rgb_to_ycocg(c_k);
            m1 += ycc;
            m2 += ycc * ycc;
            n_min = min(n_min, ycc);
            n_max = max(n_max, ycc);
        }
    }
    let mean = m1 * (1.0 / 9.0);
    let stddev = sqrt(max(m2 * (1.0 / 9.0) - mean * mean, vec3<f32>(0.0)));

    let vel_px = length(vel_uv * size_f);
    let motion_t = clamp(vel_px * 0.22, 0.0, 1.0);
    let gamma = mix(1.35, 1.05, motion_t);
    let clip_min = max(n_min, mean - stddev * gamma);
    let clip_max = min(n_max, mean + stddev * gamma);

    let curr_k = karis_compress(center_raw.rgb);
    let hist_raw = sample_history_catmull_rom(prev_uv, size_f);
    let hist_ycc = rgb_to_ycocg(karis_compress(hist_raw));
    let clamped_hist_k = max(ycocg_to_rgb(clip_aabb(clip_min, clip_max, hist_ycc)), vec3<f32>(0.0));

    let blend = mix(0.075, 0.22, motion_t);
    let blended_k = mix(clamped_hist_k, curr_k, blend);
    let final_hdr = karis_expand(blended_k);
    return vec4<f32>(final_hdr, center_raw.a);
}

@compute @workgroup_size(8, 8, 1)
fn cs_taa_store(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(taa_src);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let val = textureLoad(taa_src, vec2<i32>(gid.xy), 0);
    textureStore(taa_dst, vec2<i32>(gid.xy), val);
}
