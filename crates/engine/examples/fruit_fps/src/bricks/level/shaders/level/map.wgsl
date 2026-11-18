import package::engine::drawn::{drawn_model, drawn_slot};
import package::engine::view::{camera, stable_normalize, view_clip};
import package::level::light::level_light;
import package::level::material::{level_baked, level_broken, level_sky_map, level_sky_sampler,
    level_surface, level_texels};
import package::level::sky::{sky_gradient, sky_pictured};

override LEVEL_BAKED: bool = false;
override LEVEL_DETAIL: bool = false;
override LEVEL_CUTOUT: bool = false;

const LEVEL_PI: f32 = 3.14159265;
const LEVEL_REFLECTANCE: f32 = 0.04;
const LEVEL_SMALLEST_ALPHA: f32 = 0.002;
const LEVEL_SMALLEST_COSINE: f32 = 1.0e-4;

struct LevelVertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec2<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) uv1: vec2<f32>,
}

struct LevelDetailVertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec2<f32>,
    @location(3) uv0: vec2<f32>,
    @location(6) color: vec4<f32>,
}

struct LevelCorner {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) fog: f32,
}

fn level_octahedral(packed: vec2<f32>) -> vec3<f32> {
    var normal = vec3<f32>(packed, 1.0 - abs(packed.x) - abs(packed.y));
    let fold = max(-normal.z, 0.0);
    normal.x += select(fold, -fold, normal.x >= 0.0);
    normal.y += select(fold, -fold, normal.y >= 0.0);
    return normalize(normal);
}

fn level_deflected(normal: vec3<f32>, world: vec3<f32>, uv: vec2<f32>, sampled: vec3<f32>)
        -> vec3<f32> {
    let dp_dx = dpdx(world);
    let dp_dy = dpdy(world);
    let uv_dx = dpdx(uv);
    let uv_dy = dpdy(uv);
    let dp_dy_perp = cross(dp_dy, normal);
    let dp_dx_perp = cross(normal, dp_dx);
    let tangent = dp_dy_perp * uv_dx.x + dp_dx_perp * uv_dy.x;
    let bitangent = dp_dy_perp * uv_dx.y + dp_dx_perp * uv_dy.y;
    let frame = inverseSqrt(max(max(dot(tangent, tangent), dot(bitangent, bitangent)), 1.0e-30));
    let deflection = sampled.xy * (frame * level_surface.baked.z);
    return stable_normalize(tangent * deflection.x - bitangent * deflection.y + normal * sampled.z);
}

fn level_fog(world: vec3<f32>) -> f32 {
    let distance = length(world - camera.position.xyz);
    let fog = level_light.fog;
    return level_light.horizon.w * (1.0 - exp(-max(distance - fog.x, 0.0) * fog.y));
}

fn level_pow5(x: f32) -> f32 {
    let squared = x * x;
    return squared * squared * x;
}

fn level_corner(index: u32, position: vec3<f32>, normal: vec2<f32>, uv0: vec2<f32>) -> LevelCorner {
    let model = drawn_model(drawn_slot(index));
    let world = (model * vec4<f32>(position, 1.0)).xyz;
    var out: LevelCorner;
    out.clip_position = view_clip(world);
    out.world = world;
    out.normal = (model * vec4<f32>(level_octahedral(normal), 0.0)).xyz;
    out.uv0 = uv0;
    out.uv1 = uv0;
    out.color = vec4<f32>(1.0);
    out.fog = level_fog(world);
    return out;
}

@vertex fn vs_level(vertex: LevelVertex, @builtin(instance_index) index: u32) -> LevelCorner {
    var out = level_corner(index, vertex.position, vertex.normal, vertex.uv0);
    out.uv1 = vertex.uv1;
    return out;
}

@vertex fn vs_level_detail(vertex: LevelDetailVertex, @builtin(instance_index) index: u32)
        -> LevelCorner {
    var out = level_corner(index, vertex.position, vertex.normal, vertex.uv0);
    out.color = vertex.color;
    return out;
}

@fragment fn fs_level(corner: LevelCorner, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    let sampled = level_texels(corner.uv0);
    if LEVEL_CUTOUT && sampled.alpha < level_surface.cut.x {
        discard;
    }
    let facing = stable_normalize(select(-corner.normal, corner.normal, front));
    let texels = level_broken(sampled, corner.world, facing);
    let scale = level_surface.baked.x;
    let baked = LEVEL_DETAIL || LEVEL_BAKED;
    var indirect = vec3<f32>(0.0);
    var sun_seen = 1.0;
    if LEVEL_DETAIL {
        indirect = corner.color.rgb * corner.color.rgb * scale;
        sun_seen = corner.color.a;
    } else if LEVEL_BAKED {
        let stored = level_baked(corner.uv1);
        indirect = select(stored.rgb, stored.rgb * stored.rgb, level_surface.baked.y > 1.5) * scale;
        sun_seen = stored.a;
    }
    var normal = facing;
    if level_surface.cut.z == 0.0 {
        normal = level_deflected(facing, corner.world, corner.uv0, texels.normal);
    }
    if !baked {
        let ambient = level_light.ambient;
        let sky = level_light.sky_light.rgb * ambient.x;
        let ground = level_light.ground.rgb * ambient.y;
        let lifted = saturate((corner.world.y - level_light.sky_light.w) / max(ambient.w, 1.0e-3));
        let base = mix(1.0, mix(ambient.z, 1.0, lifted), 1.0 - abs(normal.y));
        indirect = mix(ground, sky, normal.y * 0.5 + 0.5) * base;
    }
    let view = stable_normalize(camera.position.xyz - corner.world);
    let to_sun = level_light.to_sun.xyz;
    let wrap = level_light.to_sun.w;
    let n_l = dot(normal, to_sun);
    let wrapped = saturate((n_l + wrap) / (1.0 + wrap));
    let half_way = stable_normalize(to_sun + view);
    let n_h = max(dot(normal, half_way), 0.0);
    let n_v = max(dot(normal, view), LEVEL_SMALLEST_COSINE);
    let lit_side = max(n_l, 0.0);
    let alpha = max(texels.roughness * texels.roughness, LEVEL_SMALLEST_ALPHA);
    let alpha_2 = alpha * alpha;
    let lobe = n_h * n_h * (alpha_2 - 1.0) + 1.0;
    let distribution = alpha_2 / (LEVEL_PI * lobe * lobe);
    let visibility = 0.5 / max(mix(2.0 * lit_side * n_v, lit_side + n_v, alpha), LEVEL_SMALLEST_COSINE);
    let reflectance = mix(vec3<f32>(LEVEL_REFLECTANCE), texels.base_color, texels.metallic);
    let grazing = level_pow5(1.0 - max(dot(view, half_way), 0.0));
    let fresnel = reflectance + (1.0 - reflectance) * grazing;
    let gloss = fresnel * distribution * visibility * lit_side;
    let mirrored = reflect(-view, normal);
    var sky_seen = mix(level_light.ground.rgb, level_light.sky_light.rgb, mirrored.y * 0.5 + 0.5);
    if level_surface.cut.y > 0.0 {
        var mirrored_sky = sky_gradient(mirrored);
        if level_light.picture.w > 0.0 {
            mirrored_sky = sky_pictured(level_sky_map, level_sky_sampler, mirrored);
        }
        let limit = max(level_light.reflect.x, 1.0e-3);
        let sharp = saturate(1.0 - texels.roughness / limit) * level_surface.cut.y;
        sky_seen = mix(sky_seen, mirrored_sky, sharp);
    }
    let sky_luma = max(dot(level_light.sky_light.rgb, vec3<f32>(0.333)), 1.0e-3);
    let dimmed = select(1.0, saturate(dot(indirect, vec3<f32>(0.333)) / sky_luma), baked);
    let rim_fresnel = reflectance + (1.0 - reflectance) * level_pow5(1.0 - n_v) * (1.0 - texels.roughness);
    let sheen = sky_seen * rim_fresnel * (1.0 - texels.roughness) * texels.occlusion * dimmed;
    let albedo = max(texels.base_color, vec3<f32>(level_light.ground.w));
    let diffuse = albedo * (1.0 - texels.metallic);
    let sun = level_light.sun.rgb * sun_seen;
    let lit = diffuse * (sun * wrapped + indirect * texels.occlusion) + sun * gloss + sheen;
    let glow = level_surface.glow.rgb * texels.glow;
    let shown = select(texels.base_color, lit, level_light.sun.w != 0.0) + glow;
    var fog = corner.fog;
    if !LEVEL_DETAIL {
        fog = level_fog(corner.world);
    }
    return vec4<f32>(mix(shown, level_light.horizon.rgb, fog), 1.0);
}
