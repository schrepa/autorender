import package::engine::color::luminance;
import package::face::bloom::glow_weight;
import package::face::horizons::bounce_light;
import package::face::occupancy::{foliage, occupancy_visibility};
import package::face::probe_light::environment_light;
import package::face::reflections::{max_roughness, reflects};
import package::face::sky::{ambient as sky_ambient, atmospheric, environment_levels,
    environment_sample, environment_size, horizon, mirrored_along, sky_above, sky_average};
import package::face::sun::{sun_color, sun_direction};
import package::face::tonemap::{exposure, tonemap};
import package::frame::view::{PI, camera};
import package::generated::pieces::pieces;
import package::lighting::faces::lighting::{distribution, environment_brdf, fresnel,
    sun_visibility, visibility};
import package::meshes::surface::{Shaded, SurfacePoint};

const TRANSMISSION_BEND: f32 = 0.3;
const TRANSMISSION_POWER: f32 = 4.0;

fn transmitted(point: SurfacePoint) -> vec3<f32> {
    let strength = foliage().x * point.leaf;
    if strength <= 0.0 {
        return vec3<f32>(0.0);
    }
    let bent = sun_direction() + point.normal * TRANSMISSION_BEND;
    let through = saturate(dot(point.view, -normalize(bent)));
    return point.diffuse_color * (strength * pow(through, TRANSMISSION_POWER));
}

fn foliage_occlusion(point: SurfacePoint) -> f32 {
    return 1.0 - 0.5 * saturate(foliage().x) * point.leaf;
}

struct EnvironmentLobe {
    direction: vec3<f32>,
    level: f32,
    texels: f32,
}

fn environment_lobe(point: SurfacePoint) -> EnvironmentLobe {
    let levels = environment_levels();
    return EnvironmentLobe(dominant_direction(point), point.roughness * f32(levels - 1u),
        lobe_texels(point.alpha));
}

fn lobe_texels(alpha: f32) -> f32 {
    let size = f32(environment_size());
    let lobe = 4.0 * PI * alpha * alpha;
    let texel = 4.0 * PI / (6.0 * size * size);
    return 0.5 * log2(max(lobe / texel, 1.0e-12));
}

fn sky_map_light(lobe: EnvironmentLobe) -> vec3<f32> {
    return environment_sample(lobe.direction, 0u, lobe.level);
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

fn sized_highlights() -> bool {
    return pieces.lighting.highlights.y != 0.0;
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
    let cap = pieces.lighting.highlights.z;
    let bright = luminance(light) * exposure();
    let knee = 0.5 * cap;
    if cap <= 0.0 || bright <= knee {
        return 1.0;
    }
    let over = bright - knee;
    return (knee + over / (1.0 + over / knee)) / bright;
}

fn sun_highlight(point: SurfacePoint, sun: vec3<f32>, n_l: f32) -> vec3<f32> {
    let cosine = pieces.lighting.highlights.x;
    let sine = pieces.lighting.highlights.y;
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

fn finish_reflecting(point: SurfacePoint, local: vec3<f32>, lobe: EnvironmentLobe,
        reflected_sky: vec3<f32>) -> Shaded {
    let normal = point.normal;
    let view = point.view;
    let sun = sun_direction();
    let n_l = dot(normal, sun);
    let specular = pieces.lighting.model.z;
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
    let environment = environment_light(lobe.direction, lobe.level, point.world, reflected_sky);
    let brdf = environment_brdf(point.f0, point.roughness, point.n_v);
    let reflection = environment.reflected * brdf * specular;
    let grazing = 1.0 - max(dot(point.geometric, view), 0.0);
    let rim = (grazing * grazing) * (grazing * grazing) * pieces.lighting.model.x
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
    let sky_light = (point.diffuse_color * (ambient * sky_ambient()
        + horizon() * rim) * interior.x + reflection) * point.occlusion
        * covered_sky + point.diffuse_color * bounce * covered_sky * interior.x;
    let sunlit = (point.diffuse_color * wrapped + highlight * specular + transmitted(point))
        * sun_color();
    var visible = 1.0;
    if any(sunlit != vec3<f32>(0.0)) && point.sun_shade != 1.0 && interior.y != 0.0 {
        visible = sun_visibility(point.world + point.shadow_offset, point.geometric,
            point.world_dx, point.world_dy, point.distance, point.on_terrain, point.crown);
    }
    let sun_light = sunlit * visible * (1.0 - point.sun_shade) * interior.y;
    let emitted = (point.albedo * point.emission + point.emissive) * pieces.lighting.model.y;
    let lit = sky_light + sun_light + emitted + local;
    let through = atmospheric(lit, camera.position.xyz, point.world, -view);
    let fogged = through.rgb;
    let covered = through.a;
    var shaded: Shaded;
    shaded.normal = normal;
    shaded.roughness = point.roughness;
    shaded.across = point.roughness - point.smoother_across;
    shaded.body = point.body;
    shaded.linear = fogged;
    shaded.color = tonemap(fogged);
    shaded.ambient = saturate(luminance(sky_light) / max(luminance(lit), 1.0e-6)) * (1.0 - covered)
        * foliage_occlusion(point);
    shaded.glow = glow_weight(fogged);
    if reflects() {
        let smooth_enough = saturate((1.0 - shaded.across / max_roughness())
            / REFLECTION_FADE);
        let faded = smooth_enough * (1.0 - covered);
        shaded.reflect = saturate(luminance(brdf) * specular) * faded;
        shaded.mirrored = saturate(luminance(reflection * (point.occlusion * covered_sky))
            / max(luminance(lit), 1.0e-6)) * faded;
    }
    return shaded;
}
