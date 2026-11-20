import package::engine::drawn::{drawn_model, drawn_slot};
import package::engine::view::{camera, view_clip};
import package::generated::level::{MeshBase, MeshColor, MeshLightmap, mesh_normal};
import package::level::light::level_light;
import package::level::material::{level_baked, level_broken, level_normal_sample, level_sky_map,
    level_sky_sampler, level_surface, level_texels};
import package::level::sky::{sky_gradient, sky_pictured};

override LEVEL_BAKED: bool = false;
override LEVEL_DETAIL: bool = false;
override LEVEL_CUTOUT: bool = false;
override LEVEL_BROKEN: bool = false;
override LEVEL_FLAT: bool = false;
override LEVEL_MIRROR: bool = false;
override LEVEL_BAKED_SQUARED: bool = false;
override LEVEL_GLOW: bool = false;

const LEVEL_PI: f32 = 3.14159265;
const LEVEL_REFLECTANCE: f32 = 0.04;
const LEVEL_SMALLEST_ALPHA: f32 = 0.002;
const LEVEL_SMALLEST_COSINE: f32 = 1.0e-4;

struct LevelCorner {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) fog: f32,
}

fn level_unit(value: vec3<f32>) -> vec3<f32> {
    let length_sq = dot(value, value);
    return select(vec3<f32>(0.0, 1.0, 0.0), value * inverseSqrt(length_sq), length_sq > 1.0e-20);
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
    return level_unit(tangent * deflection.x - bitangent * deflection.y + normal * sampled.z);
}

fn level_fog_at(distance: f32) -> f32 {
    let fog = level_light.fog;
    let reach = distance - fog.x;
    if level_light.horizon.w <= 0.0 || reach <= 0.0 {
        return 0.0;
    }
    return level_light.horizon.w * (1.0 - exp(-reach * fog.y));
}

fn level_fog(world: vec3<f32>) -> f32 {
    return level_fog_at(length(world - camera.position.xyz));
}

fn level_pow5(x: f32) -> f32 {
    let squared = x * x;
    return squared * squared * x;
}

fn level_corner(index: u32, base: MeshBase) -> LevelCorner {
    let model = drawn_model(drawn_slot(index));
    let world = (model * vec4<f32>(base.position, 1.0)).xyz;
    var out: LevelCorner;
    out.clip_position = view_clip(world);
    out.world = world;
    out.normal = (model * vec4<f32>(mesh_normal(base.normal), 0.0)).xyz;
    out.uv0 = base.uv0;
    out.uv1 = base.uv0;
    out.color = vec4<f32>(1.0);
    return out;
}

@vertex fn vs_level_depth(base: MeshBase, @builtin(instance_index) index: u32)
        -> @invariant @builtin(position) vec4<f32> {
    let model = drawn_model(drawn_slot(index));
    let world = (model * vec4<f32>(base.position, 1.0)).xyz;
    return view_clip(world);
}

@fragment fn fs_level_depth() {}

@vertex fn vs_level(base: MeshBase, lightmap: MeshLightmap, @builtin(instance_index) index: u32)
        -> LevelCorner {
    var out = level_corner(index, base);
    out.uv1 = lightmap.uv1;
    return out;
}

@vertex fn vs_level_detail(base: MeshBase, color: MeshColor, @builtin(instance_index) index: u32)
        -> LevelCorner {
    var out = level_corner(index, base);
    out.color = color.color;
    out.fog = level_fog(out.world);
    return out;
}

@fragment fn fs_level(corner: LevelCorner, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    let sampled = level_texels(corner.uv0);
    if LEVEL_CUTOUT && sampled.alpha < level_surface.cut.x {
        discard;
    }
    let facing = level_unit(select(-corner.normal, corner.normal, front));
    var texels = sampled;
    if LEVEL_BROKEN {
        texels = level_broken(sampled, corner.world, facing);
    }
    let scale = level_surface.baked.x;
    let baked = LEVEL_DETAIL || LEVEL_BAKED;
    var indirect = vec3<f32>(0.0);
    var sun_seen = 1.0;
    if LEVEL_DETAIL {
        indirect = corner.color.rgb * corner.color.rgb * scale;
        sun_seen = corner.color.a;
    } else if LEVEL_BAKED {
        let stored = level_baked(corner.uv1);
        if LEVEL_BAKED_SQUARED {
            indirect = stored.rgb * stored.rgb * scale;
        } else {
            indirect = stored.rgb * scale;
        }
        sun_seen = stored.a;
    }
    var normal = facing;
    if !LEVEL_FLAT {
        normal = level_deflected(facing, corner.world, corner.uv0, level_normal_sample(corner.uv0));
    }
    let to_eye = camera.position.xyz - corner.world;
    let eye_sq = dot(to_eye, to_eye);
    let inv_eye = inverseSqrt(max(eye_sq, 1.0e-20));
    var shown = texels.base_color;
    if level_light.sun.w != 0.0 {
        if !baked {
            let ambient = level_light.ambient;
            let sky = level_light.sky_light.rgb * ambient.x;
            let ground = level_light.ground.rgb * ambient.y;
            let lifted = saturate((corner.world.y - level_light.sky_light.w) / max(ambient.w, 1.0e-3));
            let base = mix(1.0, mix(ambient.z, 1.0, lifted), 1.0 - abs(normal.y));
            indirect = mix(ground, sky, normal.y * 0.5 + 0.5) * base;
        }
        let view = select(vec3<f32>(0.0, 1.0, 0.0), to_eye * inv_eye, eye_sq > 1.0e-20);
        let n_v = max(dot(normal, view), LEVEL_SMALLEST_COSINE);
        let reflectance = mix(vec3<f32>(LEVEL_REFLECTANCE), texels.base_color, texels.metallic);
        let albedo = max(texels.base_color, vec3<f32>(level_light.ground.w));
        let diffuse = albedo * (1.0 - texels.metallic);
        shown = diffuse * (indirect * texels.occlusion);
        if sun_seen > 0.0 {
            let to_sun = level_light.to_sun.xyz;
            let wrap = level_light.to_sun.w;
            let n_l = dot(normal, to_sun);
            let wrapped = saturate((n_l + wrap) * level_light.reflect.z);
            let sun = level_light.sun.rgb * sun_seen;
            shown += diffuse * (sun * wrapped);
            if n_l > 0.0 {
                let half_way = level_unit(to_sun + view);
                let n_h = max(dot(normal, half_way), 0.0);
                let alpha = max(texels.roughness * texels.roughness, LEVEL_SMALLEST_ALPHA);
                let alpha_2 = alpha * alpha;
                let lobe = n_h * n_h * (alpha_2 - 1.0) + 1.0;
                let distribution = alpha_2 / (LEVEL_PI * lobe * lobe);
                let visibility = 0.5 / max(mix(2.0 * n_l * n_v, n_l + n_v, alpha), LEVEL_SMALLEST_COSINE);
                let grazing = level_pow5(1.0 - max(dot(view, half_way), 0.0));
                let fresnel = reflectance + (1.0 - reflectance) * grazing;
                shown += sun * (fresnel * (distribution * visibility * n_l));
            }
        }
        let smoothness = 1.0 - texels.roughness;
        if smoothness > 0.0 && texels.occlusion > 0.0 {
            let mirrored = reflect(-view, normal);
            var sky_seen = mix(level_light.ground.rgb, level_light.sky_light.rgb, mirrored.y * 0.5 + 0.5);
            if LEVEL_MIRROR && level_surface.cut.y > 0.0 {
                var mirrored_sky = sky_gradient(mirrored);
                if level_light.picture.w > 0.0 {
                    mirrored_sky = sky_pictured(level_sky_map, level_sky_sampler, mirrored);
                }
                let limit = max(level_light.reflect.x, 1.0e-3);
                let sharp = saturate(1.0 - texels.roughness / limit) * level_surface.cut.y;
                sky_seen = mix(sky_seen, mirrored_sky, sharp);
            }
            let dimmed = select(1.0, saturate(dot(indirect, vec3<f32>(level_light.reflect.w))), baked);
            let rim_fresnel = reflectance + (1.0 - reflectance) * (level_pow5(1.0 - n_v) * smoothness);
            shown += sky_seen * rim_fresnel * (smoothness * texels.occlusion * dimmed);
        }
    }
    if LEVEL_GLOW {
        shown += level_surface.glow.rgb * texels.glow;
    }
    var fog = corner.fog;
    if !LEVEL_DETAIL {
        fog = level_fog_at(eye_sq * inv_eye);
    }
    return vec4<f32>(mix(shown, level_light.horizon.rgb, fog), 1.0);
}
