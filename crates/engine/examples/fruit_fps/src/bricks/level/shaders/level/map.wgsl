import package::engine::drawn::{drawn_model, drawn_slot};
import package::engine::view::{camera, stable_normalize, view_clip};
import package::level::light::level_light;
import package::level::material::{level_baked, level_texels};

override LEVEL_BAKED: bool = false;
override LEVEL_LIGHTMAP_SCALE: f32 = 1.0;
override LEVEL_LIGHTMAP_CURVE: f32 = 1.0;

const LEVEL_PI: f32 = 3.14159265;
const LEVEL_REFLECTANCE: f32 = 0.04;
const LEVEL_SMALLEST_ALPHA: f32 = 0.002;

struct LevelVertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec2<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) uv1: vec2<f32>,
}

struct LevelCorner {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
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
    let deflection = sampled.xy * frame;
    return stable_normalize(tangent * deflection.x - bitangent * deflection.y + normal * sampled.z);
}

fn level_fogged(color: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let distance = length(world - camera.position.xyz);
    let fog = level_light.fog;
    let amount = level_light.horizon.w * (1.0 - exp(-max(distance - fog.x, 0.0) * fog.y));
    return mix(color, level_light.horizon.rgb, amount);
}

@vertex fn vs_level(vertex: LevelVertex, @builtin(instance_index) index: u32) -> LevelCorner {
    let model = drawn_model(drawn_slot(index));
    let world = (model * vec4<f32>(vertex.position, 1.0)).xyz;
    var out: LevelCorner;
    out.clip_position = view_clip(world);
    out.world = world;
    out.normal = (model * vec4<f32>(level_octahedral(vertex.normal), 0.0)).xyz;
    out.uv0 = vertex.uv0;
    out.uv1 = vertex.uv1;
    return out;
}

@fragment fn fs_level(corner: LevelCorner, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    let texels = level_texels(corner.uv0);
    var indirect = vec3<f32>(0.0);
    var sun_seen = 1.0;
    if LEVEL_BAKED {
        let baked = level_baked(corner.uv1);
        let stored = select(baked.rgb, baked.rgb * baked.rgb, LEVEL_LIGHTMAP_CURVE > 1.5);
        indirect = stored * LEVEL_LIGHTMAP_SCALE;
        sun_seen = baked.a;
    }
    let facing = stable_normalize(select(-corner.normal, corner.normal, front));
    let normal = level_deflected(facing, corner.world, corner.uv0, texels.normal);
    if !LEVEL_BAKED {
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
    let alpha = max(texels.roughness * texels.roughness, LEVEL_SMALLEST_ALPHA);
    let alpha_2 = alpha * alpha;
    let lobe = n_h * n_h * (alpha_2 - 1.0) + 1.0;
    let reflectance = mix(vec3<f32>(LEVEL_REFLECTANCE), texels.base_color, texels.metallic);
    let gloss = reflectance * alpha_2 / (4.0 * LEVEL_PI * lobe * lobe) * max(n_l, 0.0);
    let albedo = max(texels.base_color, vec3<f32>(level_light.ground.w));
    let diffuse = albedo * (1.0 - texels.metallic);
    let sun = level_light.sun.rgb * sun_seen;
    let lit = diffuse * (sun * wrapped + indirect * texels.occlusion) + sun * gloss;
    let shown = select(texels.base_color, lit, level_light.sun.w != 0.0);
    return vec4<f32>(level_fogged(shown, corner.world), 1.0);
}
