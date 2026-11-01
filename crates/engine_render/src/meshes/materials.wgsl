import package::engine::view::stable_normalize;

@group(1) @binding(0) var base_color_texture: texture_2d<f32>;
@group(1) @binding(1) var base_color_sampler: sampler;
@group(1) @binding(2) var metallic_roughness_texture: texture_2d<f32>;
@group(1) @binding(3) var metallic_roughness_sampler: sampler;
@group(1) @binding(4) var normal_texture: texture_2d<f32>;
@group(1) @binding(5) var normal_sampler: sampler;
@group(1) @binding(6) var occlusion_texture: texture_2d<f32>;
@group(1) @binding(7) var occlusion_sampler: sampler;
@group(1) @binding(8) var emissive_texture: texture_2d<f32>;
@group(1) @binding(9) var emissive_sampler: sampler;
@group(1) @binding(10) var<uniform> material_maps: MaterialUniform;

override MATERIAL_MAPS: bool = false;
override NORMAL_MAP: bool = false;
override ALPHA_MASK: bool = false;
override DOUBLE_SIDED: bool = false;
override METALLIC_ROUGHNESS_MAP: bool = true;
override OCCLUSION_MAP: bool = true;
override EMISSIVE_MAP: bool = true;
override PACKED_OCCLUSION: bool = false;
override WORLD_MAPPING: bool = false;
override TOP_LAYER: bool = false;
override GROUNDING: bool = false;
override FADE_LIST: bool = false;
override UBER_MATERIALS: bool = false;
override FOLIAGE: bool = false;
override PRETESTED: bool = false;

fn material_flag(bit: u32, specialized: bool) -> bool {
    if UBER_MATERIALS {
        return (bitcast<u32>(material_maps.variant.x) & bit) != 0u;
    }
    return specialized;
}

fn map_uv(index: u32, uv: vec2<f32>, uv1: vec2<f32>) -> vec2<f32> {
    let second = ((u32(material_maps.params.w) >> index) & 1u) == 1u;
    let point = vec3<f32>(select(uv, uv1, second), 1.0);
    return vec2<f32>(dot(material_maps.transforms[index * 2u].xyz, point),
        dot(material_maps.transforms[index * 2u + 1u].xyz, point));
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

struct MaterialTexels {
    base: vec4<f32>,
    factors: vec2<f32>,
    occlusion: f32,
    emissive: vec3<f32>,
    normal: vec3<f32>,
    normal_dx: vec2<f32>,
    normal_dy: vec2<f32>,
    coverage: f32,
    world_normal: vec3<f32>,
    slopes: f32,
    height: f32,
}

fn untextured() -> MaterialTexels {
    var texels: MaterialTexels;
    texels.factors = vec2<f32>(1.0);
    texels.occlusion = 1.0;
    texels.normal = vec3<f32>(0.0, 0.0, 1.0);
    texels.coverage = 1.0;
    texels.height = 1.0;
    return texels;
}

fn mask_coverage(dx: vec2<f32>, dy: vec2<f32>) -> f32 {
    let size = vec2<f32>(textureDimensions(base_color_texture));
    let x = dx * size;
    let y = dy * size;
    return 1.0 + MASK_MIP_SCALE * max(0.5 * log2(max(dot(x, x), dot(y, y))), 0.0);
}

fn material_texels(uv: vec2<f32>, uv1: vec2<f32>, bias: f32) -> MaterialTexels {
    var texels = untextured();
    var base_uv = uv;
    if material_flag(VARIANT_MAPS, MATERIAL_MAPS) {
        base_uv = map_uv(0u, uv, uv1);
    }
    texels.base = textureSampleBias(base_color_texture, base_color_sampler, base_uv, bias);
    if ALPHA_MASK {
        let detail = exp2(bias);
        texels.coverage = mask_coverage(dpdx(base_uv) * detail, dpdy(base_uv) * detail);
    }
    if material_flag(VARIANT_MAPS, MATERIAL_MAPS) {
        var metallic_roughness = vec4<f32>(1.0);
        if material_flag(VARIANT_METALLIC_ROUGHNESS_MAP, METALLIC_ROUGHNESS_MAP) {
            metallic_roughness = textureSampleBias(metallic_roughness_texture,
                metallic_roughness_sampler, map_uv(1u, uv, uv1), bias);
        }
        texels.factors = metallic_roughness.gb;
        texels.height = metallic_roughness.a;
        if material_flag(VARIANT_PACKED_OCCLUSION, PACKED_OCCLUSION) {
            texels.occlusion = metallic_roughness.r;
        } else if material_flag(VARIANT_OCCLUSION_MAP, OCCLUSION_MAP) {
            texels.occlusion = textureSampleBias(occlusion_texture, occlusion_sampler,
                map_uv(3u, uv, uv1), bias).r;
        }
        texels.emissive = vec3<f32>(1.0);
        if material_flag(VARIANT_EMISSIVE_MAP, EMISSIVE_MAP) {
            texels.emissive = textureSampleBias(emissive_texture, emissive_sampler,
                map_uv(4u, uv, uv1), bias).rgb;
        }
    }
    if material_flag(VARIANT_NORMAL_MAP, NORMAL_MAP) {
        let normal_uv = map_uv(2u, uv, uv1);
        texels.normal = textureSampleBias(normal_texture, normal_sampler, normal_uv, bias).xyz
            * 2.0 - 1.0;
        texels.normal_dx = dpdx(normal_uv);
        texels.normal_dy = dpdy(normal_uv);
    }
    return texels;
}

fn geometric_normal(surface_normal: vec3<f32>, front: bool) -> vec3<f32> {
    let geometric = stable_normalize(surface_normal);
    if material_flag(VARIANT_DOUBLE_SIDED, DOUBLE_SIDED) && !front {
        return -geometric;
    }
    return geometric;
}

fn top_layer_amount(up: f32, breakup: f32, texels: MaterialTexels) -> f32 {
    var hollow = 1.0 - dot(texels.base.rgb, vec3<f32>(0.299, 0.587, 0.114));
    if material_flag(VARIANT_METALLIC_ROUGHNESS_MAP, METALLIC_ROUGHNESS_MAP) {
        hollow = mix(hollow, 1.0 - texels.height, 0.6);
    }
    let lift = up + 0.5 * (breakup - 0.5) + 0.4 * (hollow - 0.5);
    let reach = material_maps.top.w;
    return smoothstep(1.0 - reach, 1.2 - reach, lift) * step(1.0e-4, reach);
}
