// Fragment lighting copied from crates/engine_render/src/meshes.wgsl.
// Bevy provides the mesh vertex stage, transforms and material binding.
#import bevy_pbr::forward_io::VertexOutput

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> material_color: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var base_color_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var base_color_sampler: sampler;

fn stable_normalize(value: vec3<f32>) -> vec3<f32> {
    let scale = max(max(abs(value.x), abs(value.y)), abs(value.z));
    if scale == 0.0 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(value / scale);
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let low = color * 12.92;
    let high = 1.055 * pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, color <= vec3<f32>(0.0031308));
}

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let normal = stable_normalize(mesh.world_normal);
    let sun = max(dot(normal, normalize(vec3<f32>(-0.5, 0.85, 0.35))), 0.0);
    let sky = normal.y * 0.5 + 0.5;
    let ambient = mix(vec3<f32>(0.12, 0.14, 0.18), vec3<f32>(0.30, 0.38, 0.46), sky);
    let albedo = textureSample(base_color_texture, base_color_sampler, mesh.uv).rgb * material_color.rgb;
    let lit = albedo * (ambient + vec3<f32>(1.0, 0.91, 0.76) * sun * 0.85);
    // Fragment position.w is reciprocal clip.w; the source shader interpolates clip.w.
    let distance = 1.0 / mesh.position.w;
    let fog = 1.0 - exp(-max(distance - 35.0, 0.0) * 0.006);
    let color = mix(lit, vec3<f32>(0.009, 0.015, 0.025), min(fog, 0.85));
    // Bevy's default SDR compositing target is UNORM. Encode as our renderer does.
    return vec4<f32>(linear_to_srgb(color), 1.0);
}
