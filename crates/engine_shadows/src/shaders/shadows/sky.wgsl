import package::generated::shadows::{SKY_DIRECTIONS, SkyViews};

@group(0) @binding(0) var sky_maps: texture_depth_2d_array;
@group(0) @binding(0) var sky_sampler: sampler_comparison;
@group(0) @binding(0) var<uniform> sky_views: SkyViews;

fn sky_seen(layer: u32, p: vec3<f32>, normal: vec3<f32>, texel: f32) -> vec2<f32> {
    let toward = sky_views.directions[layer];
    if toward.w <= 0.0 {
        return vec2<f32>(1.0, 0.0);
    }
    let lifted = p + normal * (1.5 * texel) + toward.xyz * texel;
    let clip = sky_views.clips[layer] * vec4<f32>(lifted, 1.0);
    let edge = max(abs(clip.x), abs(clip.y));
    let counted = toward.w * (1.0 - smoothstep(0.85, 1.0, edge)) * select(1.0, 0.0, clip.z >= 1.0);
    if counted <= 0.0 {
        return vec2<f32>(1.0, 0.0);
    }
    let uv = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
    let seen = textureSampleCompareLevel(sky_maps, sky_sampler, uv, layer, clip.z);
    return vec2<f32>(seen, counted);
}

fn sky_occlusion(p: vec3<f32>, normal: vec3<f32>) -> f32 {
    let count = u32(sky_views.counts.x);
    if sky_views.occlusion.w == 0.0 || count == 0u {
        return 1.0;
    }
    var sum = 0.0;
    var total = 0.0;
    for (var direction = 0u; direction < count; direction++) {
        let weight = max(dot(normal, sky_views.directions[direction].xyz), 0.0) + 0.05;
        let near = sky_seen(direction, p, normal, sky_views.counts.y);
        var seen = near.x;
        if near.y < 1.0 {
            let far = sky_seen(direction + SKY_DIRECTIONS, p, normal, sky_views.counts.z);
            seen = mix(mix(1.0, far.x, far.y), near.x, near.y);
        }
        sum += weight * seen;
        total += weight;
    }
    let visible = max(sum / total, sky_views.occlusion.y);
    return 1.0 - sky_views.occlusion.x * (1.0 - visible);
}
