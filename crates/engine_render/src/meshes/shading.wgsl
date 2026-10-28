fn rippled(normal: vec3<f32>, world: vec3<f32>, strength: f32) -> vec3<f32> {
    let time = camera.position.w;
    let p = world.xz;
    let a = dot(p, vec2<f32>(0.8, 0.6)) * 0.7 + time * 1.1;
    let b = dot(p, vec2<f32>(-0.6, 0.8)) * 1.6 + time * 1.7;
    let c = dot(p, vec2<f32>(0.1, -1.0)) * 3.3 + time * 2.3;
    let slope = vec2<f32>(0.8, 0.6) * (0.126 * cos(a))
        + vec2<f32>(-0.6, 0.8) * (0.112 * cos(b))
        + vec2<f32>(0.1, -1.0) * (0.083 * cos(c));
    return normalize(normal - vec3<f32>(slope.x, 0.0, slope.y) * strength);
}

struct SurfacePoint {
    albedo: vec3<f32>,
    opacity: f32,
    emission: f32,
    emissive: vec3<f32>,
    occlusion: f32,
    roughness: f32,
    alpha: f32,
    smoother_across: f32,
    f0: vec3<f32>,
    diffuse_color: vec3<f32>,
    normal: vec3<f32>,
    geometric: vec3<f32>,
    view: vec3<f32>,
    n_v: f32,
    world: vec3<f32>,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    distance: f32,
    shadow_offset: vec3<f32>,
    on_terrain: bool,
    bounce: vec3<f32>,
    sky_turn: vec3<f32>,
    sun_shade: f32,
    leaf: f32,
    crown: f32,
    body: u32,
}

const BODY_SOLID: u32 = 0u;
const BODY_THIN: u32 = 1u;
const BODY_GROUND: u32 = 2u;
const BODY_WATER: u32 = 3u;
const MIRRORED_BITS: u32 = 14u;

struct Shaded {
    color: vec3<f32>,
    linear: vec3<f32>,
    ambient: f32,
    glow: f32,
    reflect: f32,
    mirrored: f32,
    normal: vec3<f32>,
    roughness: f32,
    across: f32,
    body: u32,
    opacity: f32,
}

var<private> fragment_plant: u32;
var<private> fragment_pixel: vec2<f32>;
var<private> fragment_slot: u32;

fn leafiness(albedo: vec3<f32>) -> f32 {
    return saturate(4.0 * (albedo.g - max(albedo.r, albedo.b)) / max(albedo.g, 1.0e-4));
}

fn plant_tint(plant: u32, leaf: f32) -> vec3<f32> {
    let strength = lighting.foliage.y * leaf;
    if strength <= 0.0 {
        return vec3<f32>(1.0);
    }
    let along = f32(plant >> 24u) / 255.0;
    let gradient = mix(vec3<f32>(1.16, 1.05, 0.64), vec3<f32>(0.80, 0.97, 1.08), along);
    return mix(vec3<f32>(1.0), gradient, strength);
}

const TRANSMISSION_BEND: f32 = 0.3;
const TRANSMISSION_POWER: f32 = 4.0;

fn transmitted(point: SurfacePoint) -> vec3<f32> {
    let strength = lighting.foliage.x * point.leaf;
    if strength <= 0.0 {
        return vec3<f32>(0.0);
    }
    let bent = lighting.sun.xyz + point.normal * TRANSMISSION_BEND;
    let through = saturate(dot(point.view, -normalize(bent)));
    return point.diffuse_color * (strength * pow(through, TRANSMISSION_POWER));
}

fn foliage_occlusion(point: SurfacePoint) -> f32 {
    return 1.0 - 0.5 * saturate(lighting.foliage.x) * point.leaf;
}

fn alpha_cutoff(cutoff: f32) -> f32 {
    if lighting.foliage.z == 0.0 {
        return cutoff;
    }
    return max(fade_noise(fragment_pixel), 1.0 / 255.0);
}

fn masked_out(opacity: f32, coverage: f32) -> bool {
    return opacity * coverage < alpha_cutoff(material_maps.params.z);
}

fn map_uv(index: u32, uv: vec2<f32>, uv1: vec2<f32>) -> vec2<f32> {
    let second = ((u32(material_maps.params.w) >> index) & 1u) == 1u;
    let point = vec3<f32>(select(uv, uv1, second), 1.0);
    return vec2<f32>(dot(material_maps.transforms[index * 2u].xyz, point),
        dot(material_maps.transforms[index * 2u + 1u].xyz, point));
}

const SPECULAR_VARIANCE: f32 = 0.15;
const SPECULAR_LIMIT: f32 = 0.2;

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

const MASK_MIP_SCALE: f32 = 0.25;

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

fn world_plane(p: vec3<f32>, axis: u32) -> vec2<f32> {
    if axis == 0u {
        return vec2<f32>(p.z, -p.y);
    }
    if axis == 2u {
        return vec2<f32>(p.x, -p.y);
    }
    return p.xz;
}

fn world_shares(normal: vec3<f32>) -> vec3<f32> {
    let sharp = pow(abs(normal), vec3<f32>(4.0));
    let shares = max(sharp / max(sharp.x + sharp.y + sharp.z, 1.0e-6) - 0.03, vec3<f32>(0.0));
    return shares / max(shares.x + shares.y + shares.z, 1.0e-6);
}

fn map_mean(map: texture_2d<f32>, map_sampler: sampler) -> vec4<f32> {
    return textureSampleLevel(map, map_sampler, vec2<f32>(0.5), f32(textureNumLevels(map) - 1u));
}

fn world_map_texel(map: texture_2d<f32>, map_sampler: sampler, index: u32, world: vec3<f32>,
        dx: vec3<f32>, dy: vec3<f32>, axis: u32) -> vec4<f32> {
    let scale = 1.0 / max(material_maps.emissive.w, 1.0e-6);
    let uv = world_plane(world, axis) * scale;
    let across = world_plane(dx, axis) * scale;
    let down = world_plane(dy, axis) * scale;
    let grad_x = map_gradient(index, across, across);
    let grad_y = map_gradient(index, down, down);
    if min(dot(grad_x, grad_x), dot(grad_y, grad_y)) >= 1.0 { // filter: the map's last level, its mean, which filtering reads there
        return map_mean(map, map_sampler);
    }
    return textureSampleGrad(map, map_sampler, map_uv(index, uv, uv), grad_x, grad_y);
}

const NORMAL_ROUNDING: f32 = 1.7320508 / 255.0;

fn hidden_slopes(sampled: vec3<f32>, scale: f32) -> f32 {
    let mean = max(length(sampled), 1.0e-4);
    return 2.0 * max(1.0 - mean - NORMAL_ROUNDING, 0.0) / mean * scale * scale;
}

fn world_plane_normal(sampled: vec3<f32>, axis: u32, normal: vec3<f32>, scale: f32)
        -> vec3<f32> {
    let n = vec3<f32>(sampled.xy * scale, sampled.z);
    if axis == 0u {
        return stable_normalize(vec3<f32>(abs(n.z) * normal.x, n.y + normal.y, n.x + normal.z));
    }
    if axis == 2u {
        return stable_normalize(vec3<f32>(n.x + normal.x, n.y + normal.y, abs(n.z) * normal.z));
    }
    return stable_normalize(vec3<f32>(n.x + normal.x, abs(n.z) * normal.y, normal.z - n.y));
}

fn world_texels(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>)
        -> MaterialTexels {
    var texels = untextured();
    let shares = world_shares(normal);
    var base = vec4<f32>(0.0);
    var metallic_roughness = vec4<f32>(0.0);
    var occlusion = 0.0;
    var emissive = vec3<f32>(0.0);
    var mapped = vec3<f32>(0.0);
    var hidden = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let share = shares[axis];
        if share <= 0.0 {
            continue;
        }
        base += share * world_map_texel(base_color_texture, base_color_sampler, 0u, world, dx, dy,
            axis);
        if material_flag(VARIANT_METALLIC_ROUGHNESS_MAP, METALLIC_ROUGHNESS_MAP) {
            metallic_roughness += share * world_map_texel(metallic_roughness_texture,
                metallic_roughness_sampler, 1u, world, dx, dy, axis);
        }
        if material_flag(VARIANT_OCCLUSION_MAP, OCCLUSION_MAP) {
            occlusion += share * world_map_texel(occlusion_texture, occlusion_sampler, 3u, world,
                dx, dy, axis).r;
        }
        if material_flag(VARIANT_EMISSIVE_MAP, EMISSIVE_MAP) {
            emissive += share * world_map_texel(emissive_texture, emissive_sampler, 4u, world, dx,
                dy, axis).rgb;
        }
        if material_flag(VARIANT_NORMAL_MAP, NORMAL_MAP) {
            let sampled = world_map_texel(normal_texture, normal_sampler, 2u, world, dx, dy, axis)
                .xyz * 2.0 - 1.0;
            mapped += share * world_plane_normal(sampled, axis, normal, material_maps.params.x);
            hidden += share * hidden_slopes(sampled, material_maps.params.x);
        }
    }
    texels.base = base;
    if material_flag(VARIANT_METALLIC_ROUGHNESS_MAP, METALLIC_ROUGHNESS_MAP) {
        texels.factors = metallic_roughness.gb;
        texels.height = metallic_roughness.a;
    }
    if material_flag(VARIANT_PACKED_OCCLUSION, PACKED_OCCLUSION) {
        texels.occlusion = metallic_roughness.r;
    } else if material_flag(VARIANT_OCCLUSION_MAP, OCCLUSION_MAP) {
        texels.occlusion = occlusion;
    }
    texels.emissive = vec3<f32>(1.0);
    if material_flag(VARIANT_EMISSIVE_MAP, EMISSIVE_MAP) {
        texels.emissive = emissive;
    }
    if material_flag(VARIANT_NORMAL_MAP, NORMAL_MAP) {
        texels.world_normal = mapped;
        texels.slopes = hidden;
    }
    return texels;
}

struct UvGradients {
    dx: vec2<f32>,
    dy: vec2<f32>,
    dx1: vec2<f32>,
    dy1: vec2<f32>,
}

fn map_gradient(index: u32, d: vec2<f32>, d1: vec2<f32>) -> vec2<f32> {
    let second = ((u32(material_maps.params.w) >> index) & 1u) == 1u;
    let change = vec3<f32>(select(d, d1, second), 0.0);
    return vec2<f32>(dot(material_maps.transforms[index * 2u].xyz, change),
        dot(material_maps.transforms[index * 2u + 1u].xyz, change));
}

fn material_texels_grad(uv: vec2<f32>, uv1: vec2<f32>, gradients: UvGradients)
        -> MaterialTexels {
    var texels = untextured();
    var base_uv = uv;
    var base_dx = gradients.dx;
    var base_dy = gradients.dy;
    if material_flag(VARIANT_MAPS, MATERIAL_MAPS) {
        base_uv = map_uv(0u, uv, uv1);
        base_dx = map_gradient(0u, gradients.dx, gradients.dx1);
        base_dy = map_gradient(0u, gradients.dy, gradients.dy1);
    }
    texels.base = textureSampleGrad(base_color_texture, base_color_sampler, base_uv, base_dx,
        base_dy);
    if ALPHA_MASK {
        texels.coverage = mask_coverage(base_dx, base_dy);
    }
    if material_flag(VARIANT_MAPS, MATERIAL_MAPS) {
        var metallic_roughness = vec4<f32>(1.0);
        if material_flag(VARIANT_METALLIC_ROUGHNESS_MAP, METALLIC_ROUGHNESS_MAP) {
            metallic_roughness = textureSampleGrad(metallic_roughness_texture,
                metallic_roughness_sampler, map_uv(1u, uv, uv1),
                map_gradient(1u, gradients.dx, gradients.dx1),
                map_gradient(1u, gradients.dy, gradients.dy1));
        }
        texels.factors = metallic_roughness.gb;
        texels.height = metallic_roughness.a;
        if material_flag(VARIANT_PACKED_OCCLUSION, PACKED_OCCLUSION) {
            texels.occlusion = metallic_roughness.r;
        } else if material_flag(VARIANT_OCCLUSION_MAP, OCCLUSION_MAP) {
            texels.occlusion = textureSampleGrad(occlusion_texture, occlusion_sampler,
                map_uv(3u, uv, uv1), map_gradient(3u, gradients.dx, gradients.dx1),
                map_gradient(3u, gradients.dy, gradients.dy1)).r;
        }
        texels.emissive = vec3<f32>(1.0);
        if material_flag(VARIANT_EMISSIVE_MAP, EMISSIVE_MAP) {
            texels.emissive = textureSampleGrad(emissive_texture, emissive_sampler,
                map_uv(4u, uv, uv1), map_gradient(4u, gradients.dx, gradients.dx1),
                map_gradient(4u, gradients.dy, gradients.dy1)).rgb;
        }
    }
    if material_flag(VARIANT_NORMAL_MAP, NORMAL_MAP) {
        texels.normal_dx = map_gradient(2u, gradients.dx, gradients.dx1);
        texels.normal_dy = map_gradient(2u, gradients.dy, gradients.dy1);
        texels.normal = textureSampleGrad(normal_texture, normal_sampler, map_uv(2u, uv, uv1),
            texels.normal_dx, texels.normal_dy).xyz * 2.0 - 1.0;
    }
    return texels;
}

fn shading_normal(geometric: vec3<f32>, world: vec3<f32>, world_dx: vec3<f32>,
        world_dy: vec3<f32>, sampled: vec3<f32>, normal_dx: vec2<f32>, normal_dy: vec2<f32>,
        ripples: f32) -> vec3<f32> {
    var normal = geometric;
    if material_flag(VARIANT_NORMAL_MAP, NORMAL_MAP) {
        normal = mapped_normal(geometric, world_dx, world_dy, normal_dx, normal_dy, sampled,
            material_maps.params.x);
    }
    if ripples > 0.0 {
        normal = rippled(normal, world, ripples);
    }
    return normal;
}

fn geometric_normal(surface_normal: vec3<f32>, front: bool) -> vec3<f32> {
    let geometric = stable_normalize(surface_normal);
    if material_flag(VARIANT_DOUBLE_SIDED, DOUBLE_SIDED) && !front {
        return -geometric;
    }
    return geometric;
}

fn textured_point(color: vec4<f32>, surface_normal: vec3<f32>, world: vec3<f32>,
        world_dx: vec3<f32>, world_dy: vec3<f32>, texels: MaterialTexels, packed: u32,
        front: bool) -> SurfacePoint {
    let surface = unpack4x8unorm(packed);
    var point: SurfacePoint;
    point.albedo = texels.base.rgb * color.rgb;
    point.opacity = surface_opacity(color, texels);
    point.emission = surface.z * 16.0 * texels.base.a;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = 1.0;
    var roughness = surface.x;
    var metalness = surface.y;
    if material_flag(VARIANT_MAPS, MATERIAL_MAPS) {
        roughness *= texels.factors.x;
        metalness *= texels.factors.y;
        point.occlusion = 1.0 + material_maps.params.y * (texels.occlusion - 1.0);
        point.emissive = material_maps.emissive.rgb * texels.emissive;
    }
    if texels.slopes > 0.0 {
        let alpha = roughness * roughness;
        roughness = sqrt(sqrt(min(alpha * alpha + texels.slopes, 1.0)));
    }
    point.roughness = max(roughness, 0.03);
    point.alpha = point.roughness * point.roughness;
    if ALPHA_MASK || material_flag(VARIANT_FOLIAGE, FOLIAGE) {
        point.leaf = leafiness(point.albedo);
        point.crown = point.leaf;
        point.albedo *= plant_tint(fragment_plant, point.leaf);
        point.body = BODY_THIN;
    }
    point.f0 = mix(vec3<f32>(0.04), point.albedo, metalness);
    point.diffuse_color = point.albedo * (1.0 - metalness);
    point.geometric = geometric_normal(surface_normal, front);
    point.normal = shading_normal(point.geometric, world, world_dx, world_dy, texels.normal,
        texels.normal_dx, texels.normal_dy, surface.w);
    if material_flag(VARIANT_WORLD_MAPPING, WORLD_MAPPING) && material_flag(VARIANT_NORMAL_MAP, NORMAL_MAP)
        && dot(texels.world_normal, texels.world_normal) > 0.0 {
        var mapped = stable_normalize(texels.world_normal);
        if dot(mapped, point.geometric) < 0.0 {
            mapped = -mapped;
        }
        point.normal = mapped;
    }
    if material_flag(VARIANT_TOP_LAYER, TOP_LAYER) {
        let breakup = variety_noise(world.xz * 2.3 + vec2<f32>(world.y * 0.7, 3.1)) * 0.6
            + variety_noise(world.xz * 0.55 - vec2<f32>(world.y * 0.4, 7.7)) * 0.4;
        let amount = top_layer_amount(point.normal.y, breakup, texels);
        point.albedo = mix(point.albedo, material_maps.top.rgb, amount);
        point.diffuse_color = mix(point.diffuse_color, material_maps.top.rgb, amount);
        point.f0 = mix(point.f0, vec3<f32>(0.04), amount);
        roughness = mix(roughness, material_maps.layer.x, amount);
        point.roughness = max(roughness, 0.03);
        point.alpha = point.roughness * point.roughness;
        point.normal = stable_normalize(mix(point.normal, point.geometric, 0.5 * amount));
    }
    let to_eye = camera.position.xyz - world;
    point.distance = length(to_eye);
    point.view = stable_normalize(to_eye);
    point.n_v = max(dot(point.normal, point.view), 1.0e-4);
    point.world = world;
    point.world_dx = world_dx;
    point.world_dy = world_dy;
    return point;
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

fn widened(point: SurfacePoint, turn_dx: vec3<f32>, turn_dy: vec3<f32>) -> SurfacePoint {
    var result = point;
    let widening = min(2.0 * SPECULAR_VARIANCE * (dot(turn_dx, turn_dx) + dot(turn_dy, turn_dy)),
        SPECULAR_LIMIT);
    result.alpha = sqrt(saturate(point.alpha * point.alpha + widening));
    result.roughness = sqrt(result.alpha);
    return result;
}

fn surface_texels(surface_normal: vec3<f32>, world: vec3<f32>, world_dx: vec3<f32>,
        world_dy: vec3<f32>, uv: vec2<f32>, uv1: vec2<f32>) -> MaterialTexels {
    var texels: MaterialTexels;
    if material_flag(VARIANT_WORLD_MAPPING, WORLD_MAPPING) {
        let detail = material_detail();
        texels = world_texels(world, stable_normalize(surface_normal), world_dx * detail,
            world_dy * detail);
    } else {
        texels = material_texels(uv, uv1, material_bias());
    }
    return texels;
}

fn surface_opacity(color: vec4<f32>, texels: MaterialTexels) -> f32 {
    if material_flag(VARIANT_MAPS, MATERIAL_MAPS) {
        return color.a * texels.base.a;
    }
    return color.a;
}

fn surface_point(color: vec4<f32>, surface_normal: vec3<f32>, world: vec3<f32>,
        uv: vec2<f32>, uv1: vec2<f32>, packed: u32, front: bool) -> SurfacePoint {
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    let texels = surface_texels(surface_normal, world, world_dx, world_dy, uv, uv1);
    var point = textured_point(color, surface_normal, world, world_dx, world_dy, texels, packed,
        front);
    if GROUNDING {
        point = grounded(point);
    }
    let result = widened(point, dpdx(point.normal), dpdy(point.normal));
    if ALPHA_MASK && !PRETESTED && masked_out(result.opacity, texels.coverage) {
        discard;
    }
    return result;
}

fn grounded(point: SurfacePoint) -> SurfacePoint {
    let ground = grounds[fragment_slot];
    let reach = bitcast<f32>(ground.z);
    if !(reach > 0.0) || ground.w == 0u {
        return point;
    }
    let pose = poses[fragment_slot];
    let origin = vec3<f32>(pose.row_0.w, pose.row_1.w, pose.row_2.w);
    let tilt = unpack2x16snorm(ground.x);
    let up = sqrt(max(1.0 - dot(tilt, tilt), 0.01));
    let above = point.world.y - origin.y - bitcast<f32>(ground.y)
        + dot(tilt, point.world.xz - origin.xz) / up;
    if above >= reach {
        return point;
    }
    let amount = 1.0 - smoothstep(0.0, reach, above);
    let texel = unpack4x8unorm(ground.w);
    let albedo = texel.rgb * texel.rgb;
    var result = point;
    result.albedo = mix(point.albedo, albedo, amount);
    result.diffuse_color = mix(point.diffuse_color, albedo, amount);
    result.f0 = mix(point.f0, vec3<f32>(0.04), amount);
    result.roughness = mix(point.roughness, max(texel.a, 0.03), amount);
    result.alpha = result.roughness * result.roughness;
    result.normal = stable_normalize(mix(point.normal, vec3<f32>(tilt.x, up, tilt.y), amount));
    result.n_v = max(dot(result.normal, result.view), 1.0e-4);
    return result;
}

fn box_projected(position: vec3<f32>, direction: vec3<f32>, center: vec3<f32>,
        extent: vec3<f32>) -> vec3<f32> {
    let safe = select(direction, vec3<f32>(1.0e-6), abs(direction) < vec3<f32>(1.0e-6));
    let exits = max((center - extent - position) / safe, (center + extent - position) / safe);
    return position + direction * min(exits.x, min(exits.y, exits.z)) - center;
}

struct EnvironmentLight {
    reflected: vec3<f32>,
    interior: vec2<f32>,
}

struct EnvironmentLobe {
    direction: vec3<f32>,
    level: f32,
    texels: f32,
}

fn environment_lobe(point: SurfacePoint) -> EnvironmentLobe {
    let levels = textureNumLevels(environment_maps);
    return EnvironmentLobe(dominant_direction(point), point.roughness * f32(levels - 1u),
        lobe_texels(point.alpha));
}

fn lobe_texels(alpha: f32) -> f32 {
    let size = f32(textureDimensions(environment_maps).x);
    let lobe = 4.0 * PI * alpha * alpha;
    let texel = 4.0 * PI / (6.0 * size * size);
    return 0.5 * log2(max(lobe / texel, 1.0e-12));
}

fn environment_light(lobe: EnvironmentLobe, world: vec3<f32>, reflected_sky: vec3<f32>)
        -> EnvironmentLight {
    var light = vec3<f32>(0.0);
    var covered = 0.0;
    var shares = vec2<f32>(1.0);
    let header = probe_words[0];
    if PROBES && header.w != 0u {
        let grid = probe_words[1];
        let cell = vec2<i32>(floor((world.xz - bitcast<vec2<f32>>(header.xy))
            / bitcast<f32>(header.z)));
        if all(cell >= vec2<i32>(0)) && all(vec2<u32>(cell) < grid.xy) {
            let list = probe_words[grid.z + u32(cell.y) * grid.x + u32(cell.x)];
            let interiors = list.z != 0u;
            for (var n = 0u; n < list.y && (covered < 0.999 || interiors); n++) {
                let entry = grid.w + (list.x + n) * PROBE_ENTRY_WORDS;
                let interior = probe_words[entry + 2u];
                let reflects = interior.w != 0u && covered < 0.999;
                let shades = interiors && interior.z != 0u;
                if !reflects && !shades {
                    continue;
                }
                let first = probe_words[entry];
                let second = probe_words[entry + 1u];
                let center = bitcast<vec3<f32>>(first.xyz);
                let blend = bitcast<f32>(first.w);
                let extent = bitcast<vec3<f32>>(second.xyz);
                let inside = extent - abs(world - center);
                let depth = min(inside.x, min(inside.y, inside.z));
                if depth <= 0.0 {
                    continue;
                }
                let fade = select(1.0, saturate(depth / blend), blend > 0.0);
                if reflects {
                    let weight = fade * (1.0 - covered);
                    let along = box_projected(world, lobe.direction, center, extent);
                    light += weight * textureSampleLevel(environment_maps, environment_sampler,
                        along, second.w, lobe.level).rgb;
                    covered += weight;
                }
                if shades {
                    shares = min(shares, mix(vec2<f32>(1.0), bitcast<vec2<f32>>(interior.xy),
                        fade));
                }
            }
        }
    }
    if covered < 0.999 {
        light += (1.0 - covered) * reflected_sky;
    }
    return EnvironmentLight(light, shares);
}

fn sky_map_light(lobe: EnvironmentLobe) -> vec3<f32> {
    return textureSampleLevel(environment_maps, environment_sampler, lobe.direction, 0,
        lobe.level).rgb;
}

fn mirrored_sky(lobe: EnvironmentLobe, world: vec3<f32>) -> vec3<f32> {
    let share = saturate(lobe.texels);
    if share >= 1.0 {
        return sky_map_light(lobe);
    }
    let exact = mirrored_along(world, lobe.direction);
    if share <= 0.0 {
        return exact;
    }
    return mix(exact, sky_map_light(lobe), share);
}

fn in_interior(world: vec3<f32>) -> bool {
    let header = probe_words[0];
    if !PROBES || header.w == 0u {
        return false;
    }
    let grid = probe_words[1];
    let cell = vec2<i32>(floor((world.xz - bitcast<vec2<f32>>(header.xy))
        / bitcast<f32>(header.z)));
    if any(cell < vec2<i32>(0)) || any(vec2<u32>(cell) >= grid.xy) {
        return false;
    }
    let list = probe_words[grid.z + u32(cell.y) * grid.x + u32(cell.x)];
    let count = select(0u, list.y, list.z != 0u);
    for (var n = 0u; n < count; n++) {
        let entry = grid.w + (list.x + n) * PROBE_ENTRY_WORDS;
        if probe_words[entry + 2u].z == 0u {
            continue;
        }
        let center = bitcast<vec3<f32>>(probe_words[entry].xyz);
        let extent = bitcast<vec3<f32>>(probe_words[entry + 1u].xyz);
        if all(abs(world - center) < extent) {
            return true;
        }
    }
    return false;
}

fn sized_highlights() -> bool {
    return lighting.highlights.y != 0.0;
}

fn dominant_direction(point: SurfacePoint) -> vec3<f32> {
    let smoothness = 1.0 - point.roughness;
    let lean = smoothness * (sqrt(smoothness) + point.roughness);
    return normalize(mix(point.normal, reflect(-point.view, point.normal), lean));
}

fn disk_energy(alpha: f32, sine: f32) -> f32 {
    let kept = alpha / saturate(alpha + 0.5 * sine);
    return kept * kept;
}

fn cap_share(light: vec3<f32>) -> f32 {
    let cap = lighting.highlights.z;
    let bright = luminance(light) * lighting.horizon.w;
    let knee = 0.5 * cap;
    if cap <= 0.0 || bright <= knee {
        return 1.0;
    }
    let over = bright - knee;
    return (knee + over / (1.0 + over / knee)) / bright;
}

fn sun_highlight(point: SurfacePoint, sun: vec3<f32>, n_l: f32) -> vec3<f32> {
    let cosine = lighting.highlights.x;
    let sine = lighting.highlights.y;
    let direction = dominant_direction(point);
    let along = dot(direction, sun);
    let across = direction - sun * along;
    let off = length(across);
    var toward = direction;
    if along < cosine {
        toward = select(sun, sun * cosine + across * (sine / off), off > 1.0e-6);
    }
    let half_vector = normalize(point.view + toward);
    let n_h = max(dot(point.normal, half_vector), 0.0);
    let lobe = distribution(n_h, point.alpha) * visibility(point.n_v, n_l, point.alpha)
        * disk_energy(point.alpha, sine);
    return fresnel(point.f0, max(dot(point.view, half_vector), 0.0)) * (lobe * n_l);
}

const REFLECTION_FADE: f32 = 0.2;

fn finish(point: SurfacePoint, local: vec3<f32>) -> Shaded {
    let lobe = environment_lobe(point);
    return finish_reflecting(point, local, lobe, mirrored_sky(lobe, point.world));
}

fn finish_ground(point: SurfacePoint, local: vec3<f32>) -> Shaded {
    let lobe = environment_lobe(point);
    return finish_reflecting(point, local, lobe, sky_map_light(lobe));
}

fn finish_reflecting(point: SurfacePoint, local: vec3<f32>, lobe: EnvironmentLobe,
        reflected_sky: vec3<f32>) -> Shaded {
    let normal = point.normal;
    let view = point.view;
    let sun = lighting.sun.xyz;
    let n_l = dot(normal, sun);
    let specular = lighting.sun_color.w;
    var facing = 1.0;
    let face = cross(point.world_dx, point.world_dy); // filter: none, the face's own plane across the pixel, a direction
    if dot(face, face) > 1.0e-20 && all(point.shadow_offset == vec3<f32>(0.0))
        && !point.on_terrain {
        let toward = select(-1.0, 1.0, dot(face, point.geometric) >= 0.0);
        facing = smoothstep(-0.05, 0.1, dot(normalize(face) * toward, sun));
    }
    let lambert = max(n_l, 0.0);
    let wrapped = lambert + (max((n_l + 0.25) / 1.25, 0.0) - lambert) * facing;
    var highlight = vec3<f32>(0.0);
    if n_l > 0.0 {
        if sized_highlights() {
            highlight = sun_highlight(point, sun, n_l);
        } else {
            let half_vector = normalize(view + sun);
            let n_h = max(dot(normal, half_vector), 0.0);
            highlight = fresnel(point.f0, max(dot(view, half_vector), 0.0))
                * (distribution(n_h, point.alpha) * visibility(point.n_v, n_l, point.alpha) * n_l);
        }
    }
    let environment = environment_light(lobe, point.world, reflected_sky);
    let brdf = environment_brdf(point.f0, point.roughness, point.n_v);
    let reflection = environment.reflected * brdf * specular;
    let grazing = 1.0 - max(dot(point.geometric, view), 0.0);
    let rim = (grazing * grazing) * (grazing * grazing) * lighting.sun.w
        * (1.0 - abs(point.geometric.y));
    let covered_sky = occupancy_visibility(point.world);
    let interior = environment.interior;
    let skyward = normalize(normal + point.sky_turn);
    var ambient = sky_average(skyward);
    var bounce = point.bounce;
    if !point.on_terrain {
        let around = bounce_light(point.world, normal, false);
        ambient = mix(ambient, sky_above(skyward) * around.sky, around.weight);
        bounce += around.light * (around.weight * point.occlusion);
    }
    let sky_light = (point.diffuse_color * (ambient * lighting.zenith.w
        + lighting.horizon.rgb * rim) * interior.x + reflection) * point.occlusion
        * covered_sky + point.diffuse_color * bounce * covered_sky * interior.x;
    let sunlit = (point.diffuse_color * wrapped + highlight * specular + transmitted(point))
        * lighting.sun_color.rgb;
    var visible = 1.0;
    if any(sunlit != vec3<f32>(0.0)) && point.sun_shade != 1.0 && interior.y != 0.0 {
        crown_light = point.crown;
        visible = sun_visibility(point.world + point.shadow_offset, point.geometric,
            point.world_dx, point.world_dy, point.distance, point.on_terrain);
        crown_light = 0.0;
    }
    let sun_light = sunlit * visible * (1.0 - point.sun_shade) * interior.y;
    let emitted = (point.albedo * point.emission + point.emissive) * lighting.effects.x;
    let lit = sky_light + sun_light + emitted + local;
    let through = atmospheric(lit, camera.position.xyz, point.world, -view);
    let fogged = through.rgb;
    let covered = through.a;
    var shaded: Shaded;
    shaded.opacity = point.opacity;
    shaded.normal = normal;
    shaded.roughness = point.roughness;
    shaded.across = point.roughness - point.smoother_across;
    shaded.body = point.body;
    shaded.linear = fogged;
    shaded.color = grade(fogged);
    shaded.ambient = saturate(luminance(sky_light) / max(luminance(lit), 1.0e-6)) * (1.0 - covered)
        * foliage_occlusion(point);
    shaded.glow = glow_weight(fogged);
    if lighting.effects.w != 0.0 {
        let smooth_enough = saturate((1.0 - shaded.across / lighting.effects.z)
            / REFLECTION_FADE);
        let faded = smooth_enough * (1.0 - covered);
        shaded.reflect = saturate(luminance(brdf) * specular) * faded;
        shaded.mirrored = saturate(luminance(reflection * (point.occlusion * covered_sky))
            / max(luminance(lit), 1.0e-6)) * faded;
    }
    return shaded;
}

struct Fragment {
    color: vec4<f32>,
    normal: vec3<f32>,
    world: vec3<f32>,
    uv: vec2<f32>,
    uv1: vec2<f32>,
    surface: u32,
    pixel: vec2<f32>,
    origin: u32,
}

fn fragment_color(material: vec4<u32>) -> vec4<f32> {
    if GROUNDING {
        fragment_slot = material.y;
        return unpack4x8unorm(material.x);
    }
    return output_color(material);
}

fn fragment(input: VertexOutput) -> Fragment {
    if FADE_LIST && fade_hidden(input.material.w, input.clip_position.xy) {
        discard;
    }
    fragment_plant = input.material.w;
    fragment_pixel = input.clip_position.xy;
    var values: Fragment;
    values.color = fragment_color(input.material);
    values.normal = input.normal_u.xyz;
    values.world = input.world_v.xyz;
    values.uv = vec2<f32>(input.normal_u.w, input.world_v.w);
    values.uv1 = values.uv;
    values.surface = input.material.z;
    values.pixel = input.clip_position.xy;
    values.origin = input.material.w >> 16u;
    return values;
}

fn extended_fragment(input: ExtendedOutput) -> Fragment {
    if FADE_LIST && fade_hidden(input.material.w, input.clip_position.xy) {
        discard;
    }
    fragment_plant = input.material.w;
    fragment_pixel = input.clip_position.xy;
    var values: Fragment;
    values.color = fragment_color(input.material) * input.color;
    values.normal = input.normal_u.xyz;
    values.world = input.world_v.xyz;
    values.uv = vec2<f32>(input.normal_u.w, input.world_v.w);
    values.uv1 = input.uv1;
    values.surface = input.material.z;
    values.pixel = input.clip_position.xy;
    values.origin = input.material.w >> 16u;
    return values;
}

fn masked_depth(input: Fragment) {
    let texels = surface_texels(input.normal, input.world, dpdx(input.world),
        dpdy(input.world), input.uv, input.uv1);
    if masked_out(surface_opacity(input.color, texels), texels.coverage) {
        discard;
    }
}

const UNLIT_ROOM: f32 = 0.12;
const COOL_ROOM: vec3<f32> = vec3<f32>(0.78, 0.88, 1.0);
const WARM_ROOM: vec3<f32> = vec3<f32>(1.0, 0.72, 0.42);

fn room_light(point: SurfacePoint, input: Fragment) -> SurfacePoint {
    let rooms = material_maps.layer.y;
    if !WINDOWS || lighting.windows.z <= 0.0 || rooms <= 0.0 {
        return point;
    }
    var base_uv = input.uv;
    if material_flag(VARIANT_MAPS, MATERIAL_MAPS) {
        base_uv = map_uv(0u, input.uv, input.uv1);
    }
    let size = vec2<f32>(textureDimensions(base_color_texture));
    let room = vec2<i32>(floor(base_uv * vec2<f32>(rooms, rooms * size.y / size.x)));
    let key = pcg(input.origin ^ pcg(bitcast<u32>(room.x) ^ pcg(bitcast<u32>(room.y))));
    let switched = lighting.windows.y + lighting.windows.z * (2.0 * unit(key) - 1.0);
    var lit = point;
    lit.emission = 0.0;
    lit.emissive = vec3<f32>(0.0);
    if lighting.windows.x < switched && unit(pcg(key + 1u)) >= lighting.windows.w {
        let tint = mix(COOL_ROOM, WARM_ROOM, unit(pcg(key + 2u)));
        let brightness = 0.6 + 0.8 * unit(pcg(key + 3u));
        lit.emissive = (point.albedo * point.emission + point.emissive) * tint
            * (brightness / luminance(tint));
    } else {
        let window = point.emission / max(unpack4x8unorm(input.surface).z * 16.0, 1e-4);
        let kept = mix(1.0, UNLIT_ROOM, saturate(window));
        lit.albedo = point.albedo * kept;
        lit.diffuse_color = point.diffuse_color * kept;
    }
    return lit;
}

fn far_rooms(point: SurfacePoint, window: f32) -> SurfacePoint {
    if !WINDOWS || lighting.windows.z <= 0.0 {
        return point;
    }
    let lit = saturate(0.5 - 0.5 * (lighting.windows.x - lighting.windows.y)
        / lighting.windows.z) * (1.0 - lighting.windows.w);
    let tint = mix(COOL_ROOM, WARM_ROOM, 0.5);
    var far = point;
    far.emission = 0.0;
    far.emissive = (point.albedo * point.emission + point.emissive) * tint
        * (lit / luminance(tint));
    let kept = mix(1.0, UNLIT_ROOM, window * (1.0 - lit));
    far.albedo = point.albedo * kept;
    far.diffuse_color = point.diffuse_color * kept;
    return far;
}

fn shade(input: Fragment, front: bool) -> Shaded {
    return finish(room_light(surface_point(input.color, input.normal, input.world, input.uv,
        input.uv1, input.surface, front), input), vec3<f32>(0.0));
}

fn tube_energy(alpha: f32, angle: f32) -> f32 {
    return alpha / saturate(alpha + 0.5 * angle);
}

fn shaped_light(point: SurfacePoint, light: Light, shaped: bool, kept: bool,
        inside: ptr<function, f32>) -> vec3<f32> {
    if kept && light.region != 0u {
        if *inside < 0.0 {
            *inside = select(0.0, 1.0, in_interior(point.world));
        }
        let in_box = *inside > 0.0;
        if (light.region == LIGHT_INTERIOR && !in_box)
            || (light.region == LIGHT_EXTERIOR && in_box) {
            return vec3<f32>(0.0);
        }
    }
    let radius = select(0.0, max(light.radius, lighting.highlights.w), shaped);
    let span = select(vec3<f32>(0.0), light.direction * light.length, shaped);
    let first = light.position - point.world;
    let last = first + span;
    let spanned = dot(span, span);
    let nearest = first + span * saturate(-dot(first, span) / max(spanned, 1.0e-8));
    let d2 = dot(nearest, nearest);
    let r2 = light.range * light.range;
    if d2 >= r2 {
        return vec3<f32>(0.0);
    }
    let ratio = d2 / r2;
    let window = saturate(1.0 - ratio * ratio);
    let first_length = max(length(first), 1.0e-4);
    let last_length = max(length(last), 1.0e-4);
    let n_l = saturate(0.5 * (dot(point.normal, first) / first_length
        + dot(point.normal, last) / last_length));
    var falloff = window * window
        / max(0.5 * (first_length * last_length + dot(first, last)), 0.01);
    if light.cos_outer > -1.5 {
        let center = first * inverseSqrt(max(dot(first, first), 1.0e-8));
        falloff *= smoothstep(light.cos_outer, light.cos_inner, dot(-center, light.direction));
    }
    if falloff == 0.0 {
        return vec3<f32>(0.0);
    }
    let reflected = dominant_direction(point);
    var toward = first;
    if spanned > 1.0e-8 {
        let across = dot(reflected, span);
        toward = first + span * saturate((dot(first, reflected) * across - dot(first, span))
            / max(spanned - across * across, 1.0e-8));
    }
    if radius > 0.0 {
        let to_ray = reflected * dot(toward, reflected) - toward;
        toward += to_ray * saturate(radius / max(length(to_ray), 1.0e-6));
    }
    let distance = max(length(first + 0.5 * span), 1.0e-4);
    let l = normalize(toward);
    let n_l_highlight = dot(point.normal, l);
    var highlight = vec3<f32>(0.0);
    if n_l_highlight > 0.0 {
        let half_vector = normalize(point.view + l);
        let n_h = max(dot(point.normal, half_vector), 0.0);
        let energy = disk_energy(point.alpha, saturate(radius / distance))
            * tube_energy(point.alpha, saturate(sqrt(spanned) / distance));
        let lobe = distribution(n_h, point.alpha)
            * visibility(point.n_v, n_l_highlight, point.alpha) * energy;
        highlight = fresnel(point.f0, max(dot(point.view, half_vector), 0.0))
            * (lobe * n_l_highlight);
        highlight *= cap_share(highlight * (lighting.sun_color.w * falloff) * light.color);
    }
    return (point.diffuse_color * n_l + highlight * lighting.sun_color.w) * light.color
        * falloff;
}

fn local_light(point: SurfacePoint, pixel: vec2<f32>) -> vec3<f32> {
    let depth = dot(clusters.depth.xyz, point.world) + clusters.depth.w;
    let end = clusters.projection.w;
    if depth >= end {
        return vec3<f32>(0.0);
    }
    let counts = clusters.counts;
    let tile = min(vec2<u32>(max(pixel, vec2<f32>(0.0)) / clusters.grid.xy),
        counts.xy - vec2<u32>(1u));
    let slice = u32(clamp(floor(log2(max(depth, 1.0e-6)) * clusters.grid.z + clusters.grid.w),
        0.0, f32(counts.z - 1u)));
    let cluster = (slice * counts.y + tile.y) * counts.x + tile.x;
    let range = cluster_ranges[cluster];
    let specular = lighting.sun_color.w;
    let shaped = (clusters.lights.y & SHAPED_LIGHTS) != 0u;
    let kept = (clusters.lights.y & KEPT_REGIONS) != 0u;
    let capped = lighting.highlights.z > 0.0;
    var inside = -1.0;
    var sum = vec3<f32>(0.0);
    for (var n = 0u; n < range.y; n++) {
        let light = lights[cluster_lights[range.x + n]];
        if shaped || capped || (kept && light.region != 0u) {
            sum += shaped_light(point, light, shaped, kept, &inside);
            continue;
        }
        let to_light = light.position - point.world;
        let d2 = dot(to_light, to_light);
        let r2 = light.range * light.range;
        if d2 >= r2 {
            continue;
        }
        let l = to_light * inverseSqrt(max(d2, 1.0e-8));
        let n_l = dot(point.normal, l);
        if n_l <= 0.0 {
            continue;
        }
        let ratio = d2 / r2;
        let window = saturate(1.0 - ratio * ratio);
        var falloff = window * window / max(d2, 0.01);
        if light.cos_outer > -1.5 {
            falloff *= smoothstep(light.cos_outer, light.cos_inner, dot(-l, light.direction));
        }
        if falloff == 0.0 {
            continue;
        }
        let half_vector = normalize(point.view + l);
        let n_h = max(dot(point.normal, half_vector), 0.0);
        let highlight = fresnel(point.f0, max(dot(point.view, half_vector), 0.0))
            * (distribution(n_h, point.alpha) * visibility(point.n_v, n_l, point.alpha));
        sum += (point.diffuse_color + highlight * specular) * light.color * (falloff * n_l);
    }
    return sum * saturate((end - depth) / (0.1 * end));
}

fn shade_lit(input: Fragment, front: bool) -> Shaded {
    let point = room_light(surface_point(input.color, input.normal, input.world, input.uv,
        input.uv1, input.surface, front), input);
    return finish(point, local_light(point, input.pixel));
}

fn temporal_output(input: TemporalOutput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = input.clip_position;
    output.normal_u = input.normal_u;
    output.world_v = input.world_v;
    output.material = input.material;
    return output;
}

fn extended_temporal_output(input: ExtendedTemporalOutput) -> ExtendedOutput {
    var output: ExtendedOutput;
    output.clip_position = input.clip_position;
    output.normal_u = input.normal_u;
    output.world_v = input.world_v;
    output.material = input.material;
    output.color = input.color;
    output.uv1 = input.uv1;
    return output;
}

fn octahedral_coordinates(normal: vec3<f32>) -> vec2<f32> {
    let n = normal / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    let sign = select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0));
    return select(n.xy, (vec2<f32>(1.0) - abs(n.yx)) * sign, n.z < 0.0);
}

fn surface_texel(shaded: Shaded) -> vec4<u32> {
    let normal = pack2x16unorm(octahedral_coordinates(shaded.normal) * 0.5 + 0.5);
    let roughness = pack4x8unorm(vec4<f32>(shaded.roughness, shaded.across, 0.0, 0.0));
    let most = f32((1u << MIRRORED_BITS) - 1u);
    let mirrored = u32(floor(saturate(shaded.mirrored) * most + 0.5));
    return vec4<u32>(normal & 0xffffu, normal >> 16u, roughness,
        mirrored | (shaded.body << MIRRORED_BITS));
}

fn scene_targets(shaded: Shaded, pixel: vec2<f32>) -> SceneTargets {
    var targets: SceneTargets;
    targets.color = vec4<f32>(stored(shaded.color, pixel), shaded.ambient);
    targets.glow = vec2<f32>(shaded.glow, shaded.reflect);
    targets.surface = surface_texel(shaded);
    return targets;
}

fn temporal_targets(shaded: Shaded, clip_position: vec4<f32>, previous: vec3<f32>)
        -> TemporalTargets {
    var targets: TemporalTargets;
    targets.color = vec4<f32>(stored(shaded.color, clip_position.xy), shaded.ambient);
    targets.glow = vec2<f32>(shaded.glow, shaded.reflect);
    targets.surface = surface_texel(shaded);
    let current = vec2<f32>(clip_position.x * camera.jitter.z - 1.0,
        1.0 - clip_position.y * camera.jitter.w) - camera.jitter.xy;
    targets.motion = (previous.xy / previous.z - current) * vec2<f32>(0.5, -0.5);
    if !(previous.z > 0.0) {
        targets.motion = vec2<f32>(2.0);
    }
    return targets;
}

fn transmitted_share() -> f32 {
    if TRANSMISSION {
        return material_maps.volume.x;
    }
    return 0.0;
}

fn transparent_point(input: Fragment, front: bool) -> SurfacePoint {
    var point = surface_point(input.color, input.normal, input.world, input.uv, input.uv1,
        input.surface, front);
    point.diffuse_color *= 1.0 - transmitted_share();
    return point;
}

const GLASS_TAPS = array<vec2<f32>, 8>(
    vec2<f32>(1.0, 0.0), vec2<f32>(0.3535534, 0.3535534),
    vec2<f32>(0.0, 1.0), vec2<f32>(-0.3535534, 0.3535534),
    vec2<f32>(-1.0, 0.0), vec2<f32>(-0.3535534, -0.3535534),
    vec2<f32>(0.0, -1.0), vec2<f32>(0.3535534, -0.3535534),
);

fn transparent_targets(shaded: Shaded, point: SurfacePoint, input: Fragment)
        -> TransparentTargets {
    let distance = length(camera.position.xyz - input.world);
    let share = transmitted_share();
    if share <= 0.0 {
        return weighted(shaded.color, shaded.opacity, distance);
    }
    let reflectance = saturate(luminance(environment_brdf(point.f0, point.roughness,
        point.n_v)));
    let through = share * (1.0 - reflectance);
    if lighting.refraction.x <= 0.0 {
        let coverage = max(shaded.opacity * (1.0 - through), 1.0e-3);
        return weighted(min(shaded.color / coverage, vec3<f32>(1.0)), coverage, distance);
    }
    let bent = refract(-point.view, point.normal, 1.0 / material_maps.volume.y);
    let exit = point.world + bent * (material_maps.volume.z * lighting.refraction.x);
    let clip = camera.view_projection * vec4<f32>(exit, 1.0);
    let size = vec2<f32>(textureDimensions(refraction_source));
    let reach = camera.region.zw - 0.5 / size;
    var uv = input.pixel / size;
    if clip.w > 1.0e-6 {
        uv = (vec2<f32>(clip.x, -clip.y) / clip.w * 0.5 + 0.5) * camera.region.zw;
    }
    uv = min(uv, reach);
    var behind = textureSampleLevel(refraction_source, refraction_sampler, uv, 0.0).rgb;
    let radius = point.roughness * lighting.refraction.y / size;
    if radius.x * size.x >= 0.5 {
        var sum = behind;
        for (var tap = 0u; tap < 8u; tap++) {
            sum += textureSampleLevel(refraction_source, refraction_sampler,
                min(uv + GLASS_TAPS[tap] * radius, reach), 0.0).rgb;
        }
        behind = sum / 9.0;
    }
    return weighted(shaded.color + behind * point.albedo * through, shaded.opacity, distance);
}
