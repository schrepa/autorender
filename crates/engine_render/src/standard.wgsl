import package::engine::materials::{MAP_BASE_COLOR, MAP_EMISSIVE, MAP_METALLIC_ROUGHNESS,
    MAP_NORMAL, MAP_OCCLUSION, SURFACE_EMISSION, base_color_map, base_color_sampler,
    emissive_factor, emissive_map, emissive_sampler, map_present, map_uv, metallic_roughness_map,
    metallic_roughness_sampler, normal_map, normal_sampler, normal_scale, occlusion_map,
    occlusion_sampler, occlusion_strength, transmission_factor};
import package::engine::shading::{Shaded, SurfaceInput, SurfacePoint};
import package::engine::view::stable_normalize;

fn base_texel(input: SurfaceInput) -> vec4<f32> {
    return textureSampleBias(base_color_map, base_color_sampler,
        map_uv(MAP_BASE_COLOR, input.uv, input.uv1), input.bias);
}

fn metal_rough(input: SurfaceInput) -> vec4<f32> {
    if !map_present(MAP_METALLIC_ROUGHNESS) {
        return vec4<f32>(1.0);
    }
    return textureSampleBias(metallic_roughness_map, metallic_roughness_sampler,
        map_uv(MAP_METALLIC_ROUGHNESS, input.uv, input.uv1), input.bias);
}

fn mapped_normal(normal: vec3<f32>, dp_dx: vec3<f32>, dp_dy: vec3<f32>, uv_dx: vec2<f32>,
        uv_dy: vec2<f32>, sampled: vec3<f32>, scale: f32) -> vec3<f32> {
    let dp_dy_perp = cross(dp_dy, normal);
    let dp_dx_perp = cross(normal, dp_dx);
    let tangent = dp_dy_perp * uv_dx.x + dp_dx_perp * uv_dy.x;
    let bitangent = dp_dy_perp * uv_dx.y + dp_dx_perp * uv_dy.y;
    let frame = inverseSqrt(max(max(dot(tangent, tangent), dot(bitangent, bitangent)), 1.0e-30));
    let deflection = sampled.xy * (scale * frame);
    return stable_normalize(tangent * deflection.x - bitangent * deflection.y + normal * sampled.z);
}

fn rippled(normal: vec3<f32>, world: vec3<f32>, strength: f32, time: f32) -> vec3<f32> {
    let p = world.xz;
    let a = dot(p, vec2<f32>(0.8, 0.6)) * 0.7 + time * 1.1;
    let b = dot(p, vec2<f32>(-0.6, 0.8)) * 1.6 + time * 1.7;
    let c = dot(p, vec2<f32>(0.1, -1.0)) * 3.3 + time * 2.3;
    let slope = vec2<f32>(0.8, 0.6) * (0.126 * cos(a))
        + vec2<f32>(-0.6, 0.8) * (0.112 * cos(b))
        + vec2<f32>(0.1, -1.0) * (0.083 * cos(c));
    return normalize(normal - vec3<f32>(slope.x, 0.0, slope.y) * strength);
}

fn base_color(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    return base_texel(input).rgb * input.color.rgb;
}

fn opacity(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return base_texel(input).a * input.color.a;
}

fn normal(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    var shading = point.geometric;
    if map_present(MAP_NORMAL) {
        let uv = map_uv(MAP_NORMAL, input.uv, input.uv1);
        let sampled = textureSampleBias(normal_map, normal_sampler, uv, input.bias).xyz
            * 2.0 - 1.0;
        shading = mapped_normal(shading, input.world_dx, input.world_dy, dpdx(uv), dpdy(uv),
            sampled, normal_scale());
    }
    if input.surface.w > 0.0 {
        shading = rippled(shading, input.world, input.surface.w, input.time);
    }
    return shading;
}

fn roughness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return input.surface.x * metal_rough(input).g;
}

fn metalness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return input.surface.y * metal_rough(input).b;
}

fn emissive_texel(input: SurfaceInput) -> vec3<f32> {
    if !map_present(MAP_EMISSIVE) {
        return vec3<f32>(1.0);
    }
    return textureSampleBias(emissive_map, emissive_sampler,
        map_uv(MAP_EMISSIVE, input.uv, input.uv1), input.bias).rgb;
}

fn emissive(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    let glow = input.surface.z * SURFACE_EMISSION * base_texel(input).a;
    return point.base_color * glow + emissive_factor() * emissive_texel(input);
}

fn occlusion(input: SurfaceInput, point: SurfacePoint) -> f32 {
    if !map_present(MAP_OCCLUSION) {
        return 1.0;
    }
    let sampled = textureSampleBias(occlusion_map, occlusion_sampler,
        map_uv(MAP_OCCLUSION, input.uv, input.uv1), input.bias).r;
    return 1.0 + occlusion_strength() * (sampled - 1.0);
}

fn transmission(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return transmission_factor();
}

fn output(input: SurfaceInput, point: SurfacePoint, shaded: Shaded) -> Shaded {
    return shaded;
}
