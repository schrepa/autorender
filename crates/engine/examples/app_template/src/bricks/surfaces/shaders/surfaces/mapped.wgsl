import package::surfaces::materials::{MAP_BASE_COLOR, MAP_EMISSIVE, MAP_NORMAL, MASK_MIP_SCALE,
    SURFACE_EMISSION, alpha_cutoff, base_color_map, base_color_sampler, emissive_factor,
    emissive_map, emissive_sampler, map_present, map_uv, normal_map, normal_sampler, normal_scale};
import package::surfaces::shading::{SurfaceInput, SurfacePoint};
import package::engine::view::stable_normalize;

fn base_texel(input: SurfaceInput) -> vec4<f32> {
    return textureSampleBias(base_color_map, base_color_sampler,
        map_uv(MAP_BASE_COLOR, input.uv, input.uv1), input.bias);
}

fn deflected_normal(normal: vec3<f32>, dp_dx: vec3<f32>, dp_dy: vec3<f32>, uv_dx: vec2<f32>,
        uv_dy: vec2<f32>, sampled: vec3<f32>, scale: f32) -> vec3<f32> {
    let dp_dy_perp = cross(dp_dy, normal);
    let dp_dx_perp = cross(normal, dp_dx);
    let tangent = dp_dy_perp * uv_dx.x + dp_dx_perp * uv_dy.x;
    let bitangent = dp_dy_perp * uv_dx.y + dp_dx_perp * uv_dy.y;
    let frame = inverseSqrt(max(max(dot(tangent, tangent), dot(bitangent, bitangent)), 1.0e-30));
    let deflection = sampled.xy * (scale * frame);
    return stable_normalize(tangent * deflection.x - bitangent * deflection.y + normal * sampled.z);
}

fn emissive_texel(input: SurfaceInput) -> vec3<f32> {
    if !map_present(MAP_EMISSIVE) {
        return vec3<f32>(1.0);
    }
    return textureSampleBias(emissive_map, emissive_sampler,
        map_uv(MAP_EMISSIVE, input.uv, input.uv1), input.bias).rgb;
}

fn mask_coverage(input: SurfaceInput) -> f32 {
    let uv = map_uv(MAP_BASE_COLOR, input.uv, input.uv1);
    let size = vec2<f32>(textureDimensions(base_color_map)) * exp2(input.bias);
    let x = dpdx(uv) * size;
    let y = dpdy(uv) * size;
    return 1.0 + MASK_MIP_SCALE * max(0.5 * log2(max(dot(x, x), dot(y, y))), 0.0);
}

fn base_color(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    return base_texel(input).rgb * input.color.rgb;
}

fn opacity(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return base_texel(input).a * input.color.a;
}

fn cutoff(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return alpha_cutoff() / mask_coverage(input);
}

fn normal(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    var shading = point.geometric;
    if map_present(MAP_NORMAL) {
        let uv = map_uv(MAP_NORMAL, input.uv, input.uv1);
        let sampled = textureSampleBias(normal_map, normal_sampler, uv, input.bias).xyz
            * 2.0 - 1.0;
        shading = deflected_normal(shading, input.world_dx, input.world_dy, dpdx(uv), dpdy(uv),
            sampled, normal_scale());
    }
    return shading;
}

fn emissive(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    let glow = input.surface.z * SURFACE_EMISSION * base_texel(input).a;
    return point.base_color * glow + emissive_factor() * emissive_texel(input);
}
