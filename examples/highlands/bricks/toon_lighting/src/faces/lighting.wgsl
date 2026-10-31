import package::engine::color::luminance;
import package::face::bloom::glow_weight;
import package::face::lights::{cluster_light, light_cluster, lights_on};
import package::face::shadows::{shadow_lookup, sun_shadow, sun_shadowed};
import package::face::sky::{ambient, atmospheric, sky_average};
import package::face::sun::{sun_color, sun_direction};
import package::face::tonemap::tonemap;
import package::frame::view::{PI, camera};
import package::meshes::debug::{DEBUG_SHADOWED, DEBUG_SUN_BEYOND, DEBUG_SUN_LEVEL,
    DEBUG_SUN_NONE};
import package::meshes::surface::{Shaded, SurfacePoint};
import package::toon::bands::{TOON_DARK, TOON_HIGHLIGHTS, TOON_NEAREST, TOON_RIM,
    TOON_SHADOW_EDGE, TOON_SHARPEST, TOON_SPECULAR, TOON_UP, TOON_VISIBILITY, TOON_WIDEST,
    banded};

fn highlights() -> vec4<f32> {
    return TOON_HIGHLIGHTS;
}

fn specular() -> f32 {
    return TOON_SPECULAR;
}

fn widened(point: SurfacePoint, turn_dx: vec3<f32>, turn_dy: vec3<f32>) -> SurfacePoint {
    var result = point;
    result.alpha = max(point.alpha, min(length(turn_dx) + length(turn_dy), TOON_WIDEST));
    return result;
}

fn distribution(n_h: f32, alpha: f32) -> f32 {
    let cap = max(alpha * alpha, TOON_SHARPEST);
    return select(0.0, 1.0 / (PI * cap), n_h * n_h >= 1.0 - cap);
}

fn visibility(n_v: f32, n_l: f32, alpha: f32) -> f32 {
    return TOON_VISIBILITY;
}

fn fresnel(f0: vec3<f32>, cosine: f32) -> vec3<f32> {
    return select(f0, vec3<f32>(1.0), cosine < TOON_RIM);
}

fn environment_brdf(f0: vec3<f32>, roughness: f32, n_v: f32) -> vec3<f32> {
    return fresnel(f0, n_v) * (1.0 - roughness);
}

fn sun_visibility(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>,
        distance: f32, on_terrain: bool, crown: f32) -> f32 {
    if !sun_shadowed() {
        return 1.0;
    }
    return step(TOON_SHADOW_EDGE, sun_shadow(world, normal, dx, dy, crown));
}

fn sun_debug(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>,
        on_terrain: bool) -> u32 {
    if !sun_shadowed() {
        return DEBUG_SUN_NONE;
    }
    let found = shadow_lookup(world, normal, dx, dy);
    if found.y < 0.0 {
        return DEBUG_SUN_BEYOND;
    }
    return (DEBUG_SUN_LEVEL + u32(found.y))
        | select(0u, DEBUG_SHADOWED, found.x < TOON_SHADOW_EDGE);
}

fn local_light(point: SurfacePoint, pixel: vec2<f32>) -> vec3<f32> {
    if !lights_on() {
        return vec3<f32>(0.0);
    }
    let cluster = light_cluster(point.world, pixel);
    if cluster.depth >= cluster.end {
        return vec3<f32>(0.0);
    }
    var sum = vec3<f32>(0.0);
    for (var n = 0u; n < cluster.range.y; n++) {
        let light = cluster_light(cluster, n);
        let to_light = light.position - point.world;
        let d2 = dot(to_light, to_light);
        let reach = 1.0 - d2 / max(light.range * light.range, TOON_DARK);
        let toward = to_light * inverseSqrt(max(d2, TOON_DARK));
        let inside = light.cos_outer <= -1.5
            || dot(-toward, light.direction) >= 0.5 * (light.cos_outer + light.cos_inner);
        if reach > 0.0 && inside {
            sum += point.diffuse_color * light.color
                * (banded(dot(point.normal, toward)) * banded(reach) / max(d2, TOON_NEAREST));
        }
    }
    return sum;
}

fn finish(point: SurfacePoint, local: vec3<f32>) -> Shaded {
    let sun = sun_direction();
    let n_l = dot(point.normal, sun);
    let half_vector = normalize(point.view + sun);
    let n_h = max(dot(point.normal, half_vector), 0.0);
    let spot = fresnel(point.f0, max(dot(point.view, half_vector), 0.0))
        * (distribution(n_h, point.alpha) * visibility(point.n_v, n_l, point.alpha));
    let shine = select(0.0, specular(), n_l > 0.0);
    let sunlit = (point.diffuse_color * banded(n_l) + spot * shine) * sun_color()
        * (1.0 - point.sun_shade);
    var shown = 1.0;
    if any(sunlit != vec3<f32>(0.0)) {
        shown = sun_visibility(point.world + point.shadow_offset, point.geometric,
            point.world_dx, point.world_dy, point.distance, point.on_terrain, point.crown);
    }
    let skylight = point.diffuse_color * sky_average(TOON_UP) * (ambient() * point.occlusion);
    let lit = skylight + sunlit * shown + point.albedo * point.emission + point.emissive
        + local;
    let through = atmospheric(lit, camera.position.xyz, point.world, -point.view);
    var shaded: Shaded;
    shaded.color = tonemap(through.rgb);
    shaded.linear = through.rgb;
    shaded.ambient = saturate(luminance(skylight) / max(luminance(lit), TOON_DARK))
        * (1.0 - through.a);
    shaded.glow = glow_weight(through.rgb);
    shaded.normal = point.normal;
    shaded.roughness = point.roughness;
    shaded.body = point.body;
    return shaded;
}

fn finish_ground(point: SurfacePoint, local: vec3<f32>) -> Shaded {
    return finish(point, local);
}
