import package::generated::post::PostParams;
import package::post::tonemap::POST_FLAG_ACES;
import package::post::tonemap::POST_FLAG_AGX_PUNCH;
import package::post::tonemap::POST_FLAG_DISABLE_BLOOM;
import package::post::tonemap::POST_FLAG_DISABLE_DITHER;
import package::post::tonemap::POST_FLAG_DISABLE_SHAFTS;
import package::post::tonemap::POST_FLAG_LINEAR_OUT;
import package::post::tonemap::aces_tonemap;
import package::post::tonemap::agx_tonemap_look;
import package::post::tonemap::apply_film_grain;
import package::post::tonemap::apply_vignette;
import package::post::tonemap::interleaved_gradient_noise;
import package::post::tonemap::linear_to_srgb;
import package::post::tonemap::post_luma;

@group(1) @binding(0) var<uniform> post_params: PostParams;
@group(1) @binding(1) var hdr_color: texture_2d<f32>;
@group(1) @binding(2) var bloom_tex: texture_2d<f32>;
@group(1) @binding(3) var post_sampler: sampler;

struct FullscreenOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> FullscreenOut {
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    var out: FullscreenOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

@fragment
fn fs_composite(in: FullscreenOut) -> @location(0) vec4<f32> {
    var hdr = max(textureSampleLevel(hdr_color, post_sampler, in.uv, 0.0).rgb, vec3<f32>(0.0));
    let bloom_active = (post_params.flags & POST_FLAG_DISABLE_BLOOM) == 0u
        && post_params.bloom_intensity > 0.0;
    let shafts_active = (post_params.flags & POST_FLAG_DISABLE_SHAFTS) == 0u
        && post_params.shaft_intensity > 0.0
        && post_params.sun_visibility > 0.0;
    if bloom_active || shafts_active {
        let bloom_rgb = max(
            textureSampleLevel(bloom_tex, post_sampler, in.uv, 0.0).rgb,
            vec3<f32>(0.0),
        );
        hdr = hdr + bloom_rgb;
    }
    hdr = apply_vignette(hdr, in.uv, post_params.vignette);
    var mapped: vec3<f32>;
    if (post_params.flags & POST_FLAG_ACES) != 0u {
        mapped = aces_tonemap(hdr * max(post_params.exposure, 0.0));
        let luma = post_luma(mapped);
        mapped = saturate(mix(vec3<f32>(luma), mapped, post_params.saturation));
    } else {
        let punch = (post_params.flags & POST_FLAG_AGX_PUNCH) != 0u;
        mapped = agx_tonemap_look(hdr, post_params.exposure, post_params.saturation, punch);
    }
    var out_rgb = mapped;
    if (post_params.flags & POST_FLAG_LINEAR_OUT) == 0u {
        out_rgb = linear_to_srgb(out_rgb);
    }
    if post_params.grain_intensity > 0.0 {
        out_rgb = apply_film_grain(
            out_rgb,
            in.pos.xy,
            post_params.time,
            post_params.grain_intensity,
        );
    } else if (post_params.flags & POST_FLAG_DISABLE_DITHER) == 0u
        && (post_params.flags & POST_FLAG_LINEAR_OUT) == 0u
    {
        let dither = interleaved_gradient_noise(in.pos.xy, post_params.time) - 0.5;
        out_rgb = saturate(out_rgb + vec3<f32>(dither * (1.0 / 255.0)));
    }
    return vec4<f32>(out_rgb, 1.0);
}
