import package::engine::materials::{MAP_BASE_COLOR, MAP_METALLIC_ROUGHNESS, MAP_NORMAL,
    MAP_OCCLUSION, VALUE_OCCLUSION_STRENGTH, VALUE_TOP_BLUE, VALUE_TOP_COVERAGE, VALUE_TOP_GREEN,
    VALUE_TOP_RED, VALUE_TOP_ROUGHNESS, VALUE_WORLD_TILING, base_color_map, base_color_sampler,
    map_gradient, map_present, map_uv, material_value, metallic_roughness_map,
    metallic_roughness_sampler, normal_map, normal_sampler, normal_scale, occlusion_map,
    occlusion_sampler};
import package::engine::shading::{SurfaceInput, SurfacePoint};
import package::engine::view::{material_detail, stable_normalize};
import package::scatter::grounding::{grounded_color, grounded_metalness, grounded_normal,
    grounded_roughness};

const NORMAL_ROUNDING: f32 = 1.7320508 / 255.0;

fn variety_hash(cell: vec2<i32>) -> f32 {
    var h = (u32(cell.x) * 0x8da6b343u) ^ (u32(cell.y) * 0xd8163841u);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return f32((h ^ (h >> 16u)) >> 8u) * (1.0 / 16777216.0);
}

fn variety_noise(p: vec2<f32>) -> f32 {
    let cell = vec2<i32>(floor(p));
    let f = fract(p);
    let s = f * f * (3.0 - 2.0 * f);
    let a = variety_hash(cell);
    let b = variety_hash(cell + vec2<i32>(1, 0));
    let c = variety_hash(cell + vec2<i32>(0, 1));
    let d = variety_hash(cell + vec2<i32>(1, 1));
    return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
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

struct WorldLook {
    base: vec4<f32>,
    factors: vec2<f32>,
    height: f32,
    occlusion: f32,
    normal: vec3<f32>,
    slopes: f32,
    top: f32,
}

fn world_shares(facing: vec3<f32>) -> vec3<f32> {
    let sharp = pow(abs(facing), vec3<f32>(4.0));
    let shares = max(sharp / max(sharp.x + sharp.y + sharp.z, 1.0e-6) - 0.03, vec3<f32>(0.0));
    return shares / max(shares.x + shares.y + shares.z, 1.0e-6);
}

fn map_mean(map: texture_2d<f32>, map_sampler: sampler) -> vec4<f32> {
    return textureSampleLevel(map, map_sampler, vec2<f32>(0.5), f32(textureNumLevels(map) - 1u));
}

fn world_texel(map: texture_2d<f32>, map_sampler: sampler, index: u32, input: SurfaceInput,
        axis: u32) -> vec4<f32> {
    let scale = 1.0 / max(material_value(VALUE_WORLD_TILING), 1.0e-6);
    let detail = material_detail();
    let uv = world_plane(input.world, axis) * scale;
    let across = world_plane(input.world_dx * detail, axis) * scale;
    let down = world_plane(input.world_dy * detail, axis) * scale;
    let grad_x = map_gradient(index, across, across);
    let grad_y = map_gradient(index, down, down);
    if min(dot(grad_x, grad_x), dot(grad_y, grad_y)) >= 1.0 {
        return map_mean(map, map_sampler);
    }
    return textureSampleGrad(map, map_sampler, map_uv(index, uv, uv), grad_x, grad_y);
}

fn hidden_slopes(sampled: vec3<f32>, scale: f32) -> f32 {
    let mean = max(length(sampled), 1.0e-4);
    return 2.0 * max(1.0 - mean - NORMAL_ROUNDING, 0.0) / mean * scale * scale;
}

fn plane_normal(sampled: vec3<f32>, axis: u32, facing: vec3<f32>, scale: f32) -> vec3<f32> {
    let n = vec3<f32>(sampled.xy * scale, sampled.z);
    if axis == 0u {
        return stable_normalize(vec3<f32>(abs(n.z) * facing.x, n.y + facing.y, n.x + facing.z));
    }
    if axis == 2u {
        return stable_normalize(vec3<f32>(n.x + facing.x, n.y + facing.y, abs(n.z) * facing.z));
    }
    return stable_normalize(vec3<f32>(n.x + facing.x, abs(n.z) * facing.y, facing.z - n.y));
}

fn top_amount(up: f32, input: SurfaceInput, look: WorldLook) -> f32 {
    let world = input.world;
    let breakup = variety_noise(world.xz * 2.3 + vec2<f32>(world.y * 0.7, 3.1)) * 0.6
        + variety_noise(world.xz * 0.55 - vec2<f32>(world.y * 0.4, 7.7)) * 0.4;
    var hollow = 1.0 - dot(look.base.rgb, vec3<f32>(0.299, 0.587, 0.114));
    if map_present(MAP_METALLIC_ROUGHNESS) {
        hollow = mix(hollow, 1.0 - look.height, 0.6);
    }
    let lift = up + 0.5 * (breakup - 0.5) + 0.4 * (hollow - 0.5);
    let reach = material_value(VALUE_TOP_COVERAGE);
    return smoothstep(1.0 - reach, 1.2 - reach, lift) * step(1.0e-4, reach);
}

var<private> held_look: WorldLook;

fn prepare(input: SurfaceInput, point: SurfacePoint) {
    held_look = sampled_look(input, point);
}

fn sampled_look(input: SurfaceInput, point: SurfacePoint) -> WorldLook {
    var look: WorldLook;
    look.factors = vec2<f32>(1.0);
    look.height = 1.0;
    look.occlusion = 1.0;
    look.normal = point.geometric;
    let shares = world_shares(point.geometric);
    var base = vec4<f32>(0.0);
    var metal_rough = vec4<f32>(0.0);
    var occluded = 0.0;
    var mapped = vec3<f32>(0.0);
    var hidden = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let share = shares[axis];
        if share <= 0.0 {
            continue;
        }
        base += share * world_texel(base_color_map, base_color_sampler, MAP_BASE_COLOR, input,
            axis);
        if map_present(MAP_METALLIC_ROUGHNESS) {
            metal_rough += share * world_texel(metallic_roughness_map,
                metallic_roughness_sampler, MAP_METALLIC_ROUGHNESS, input, axis);
        }
        if map_present(MAP_OCCLUSION) {
            occluded += share * world_texel(occlusion_map, occlusion_sampler, MAP_OCCLUSION,
                input, axis).r;
        }
        if map_present(MAP_NORMAL) {
            let sampled = world_texel(normal_map, normal_sampler, MAP_NORMAL, input, axis).xyz
                * 2.0 - 1.0;
            mapped += share * plane_normal(sampled, axis, point.geometric, normal_scale());
            hidden += share * hidden_slopes(sampled, normal_scale());
        }
    }
    look.base = base;
    if map_present(MAP_METALLIC_ROUGHNESS) {
        look.factors = metal_rough.gb;
        look.height = metal_rough.a;
    }
    if map_present(MAP_OCCLUSION) {
        look.occlusion = 1.0 + material_value(VALUE_OCCLUSION_STRENGTH) * (occluded - 1.0);
    }
    if dot(mapped, mapped) > 0.0 {
        look.normal = stable_normalize(mapped);
        if dot(look.normal, point.geometric) < 0.0 {
            look.normal = -look.normal;
        }
        look.slopes = hidden;
    }
    look.top = top_amount(look.normal.y, input, look);
    look.normal = stable_normalize(mix(look.normal, point.geometric, 0.5 * look.top));
    return look;
}

fn top_color() -> vec3<f32> {
    return vec3<f32>(material_value(VALUE_TOP_RED), material_value(VALUE_TOP_GREEN),
        material_value(VALUE_TOP_BLUE));
}

fn base_color(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    let look = held_look;
    return grounded_color(input, mix(look.base.rgb * input.color.rgb, top_color(), look.top));
}

fn normal(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    return grounded_normal(input, held_look.normal);
}

fn roughness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    let look = held_look;
    var rough = input.surface.x * look.factors.x;
    if look.slopes > 0.0 {
        let alpha = rough * rough;
        rough = sqrt(sqrt(min(alpha * alpha + look.slopes, 1.0)));
    }
    rough = mix(rough, material_value(VALUE_TOP_ROUGHNESS), look.top);
    return grounded_roughness(input, rough);
}

fn metalness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    let look = held_look;
    return grounded_metalness(input, mix(input.surface.y * look.factors.y, 0.0, look.top));
}

fn occlusion(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return held_look.occlusion;
}
