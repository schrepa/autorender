import package::frame::encode::stored;
import package::frame::view::{camera, fade_noise, lighting, material_bias, material_detail,
    stable_normalize};
import package::lighting::surface::{BODY_THIN, MIRRORED_BITS, Shaded, SurfacePoint, finish};
import package::lights::shade::local_light;
import package::meshes::declarations::{ALPHA_MASK, DOUBLE_SIDED, EMISSIVE_MAP, FADE_LIST, FOLIAGE,
    GROUNDING, MATERIAL_MAPS, METALLIC_ROUGHNESS_MAP, NORMAL_MAP, OCCLUSION_MAP, PACKED_OCCLUSION,
    PRETESTED, SceneTargets, TOP_LAYER, TemporalOutput, TemporalTargets, VARIANT_DOUBLE_SIDED,
    VARIANT_EMISSIVE_MAP, VARIANT_FOLIAGE, VARIANT_MAPS, VARIANT_METALLIC_ROUGHNESS_MAP,
    VARIANT_NORMAL_MAP, VARIANT_OCCLUSION_MAP, VARIANT_PACKED_OCCLUSION, VARIANT_TOP_LAYER,
    VARIANT_WORLD_MAPPING, VertexOutput, WORLD_MAPPING, base_color_sampler, base_color_texture,
    emissive_sampler, emissive_texture, fade_hidden, grounds, material_flag, material_maps,
    metallic_roughness_sampler, metallic_roughness_texture, normal_sampler, normal_texture,
    occlusion_sampler, occlusion_texture, poses};
import package::meshes::vertex::{ExtendedOutput, ExtendedTemporalOutput, output_color};
import package::terrain::variety::variety_noise;

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

fn map_gradient(index: u32, d: vec2<f32>, d1: vec2<f32>) -> vec2<f32> {
    let second = ((u32(material_maps.params.w) >> index) & 1u) == 1u;
    let change = vec3<f32>(select(d, d1, second), 0.0);
    return vec2<f32>(dot(material_maps.transforms[index * 2u].xyz, change),
        dot(material_maps.transforms[index * 2u + 1u].xyz, change));
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

struct Fragment {
    color: vec4<f32>,
    normal: vec3<f32>,
    world: vec3<f32>,
    uv: vec2<f32>,
    uv1: vec2<f32>,
    surface: u32,
    pixel: vec2<f32>,
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
    return values;
}

fn masked_depth(input: Fragment) {
    let texels = surface_texels(input.normal, input.world, dpdx(input.world),
        dpdy(input.world), input.uv, input.uv1);
    if masked_out(surface_opacity(input.color, texels), texels.coverage) {
        discard;
    }
}

fn shade(input: Fragment, front: bool) -> Shaded {
    return finish(surface_point(input.color, input.normal, input.world, input.uv, input.uv1,
        input.surface, front), vec3<f32>(0.0));
}

fn shade_lit(input: Fragment, front: bool) -> Shaded {
    let point = surface_point(input.color, input.normal, input.world, input.uv, input.uv1,
        input.surface, front);
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
