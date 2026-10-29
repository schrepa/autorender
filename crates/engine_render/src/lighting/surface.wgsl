import package::engine::color::luminance;
import package::frame::encode::{glow_weight, grade};
import package::frame::view::{PI, camera, lighting};
import package::lighting::brdf::{distribution, environment_brdf, fresnel, visibility};
import package::lighting::sun::sun_visibility;
import package::probes::sample::{PROBES, PROBE_ENTRY_WORDS, probe_words};
import package::shadows::sample::crown_light;
import package::sky::air::{atmospheric, sky_above, sky_average};
import package::sky::environment::{environment_maps, environment_sampler};
import package::sky::light::mirrored_along;
import package::terrain::bounce::bounce_light;
import package::terrain::occupancy::occupancy_visibility;

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
