import package::engine::color::luminance;
import package::engine::shading::{Shaded, SurfacePoint};
import package::engine::view::{PI, camera};
import package::face::ambient::{EnvironmentAt, EnvironmentLight, environment};
import package::face::lights::{cluster_light, light_cluster, lights_on};
import package::face::sky::atmospheric;
import package::face::sun::{sun_color, sun_direction};
import package::face::sun_blockers::{DEBUG_SUN_NONE, SunAt, SunOpen, sun_flags, sun_open};
import package::toon_lighting::bands::{TOON_DARK, TOON_HIGHLIGHTS, TOON_NEAREST, TOON_RIM,
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

fn surface_sun(point: SurfacePoint) -> SunAt {
    return SunAt(point.world + point.shadow_offset, point.geometric, point.world_dx,
        point.world_dy, point.shadow_softness, point.ground, point.shadow_lift_first,
        point.shadow_lifts);
}

fn sun_visibility(at: SunAt) -> f32 {
    let open = sun_open(SunOpen(1.0, 1.0), at);
    return step(TOON_SHADOW_EDGE, open.blocked * open.passing);
}

fn sun_debug(at: SunAt) -> u32 {
    return sun_flags(DEBUG_SUN_NONE, at);
}

fn toon_environment() -> EnvironmentLight {
    var light: EnvironmentLight;
    light.near_open = 1.0;
    light.open = 1.0;
    light.inside = vec2<f32>(1.0);
    return light;
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
        shown = sun_visibility(surface_sun(point));
    }
    let around = environment(toon_environment(), EnvironmentAt(point.world, point.normal,
        TOON_UP, point.normal, point.roughness, point.alpha, point.ground));
    let skylight = point.diffuse_color * around.sky * point.occlusion;
    let lit = skylight + sunlit * shown + point.albedo * point.emission + point.emissive
        + local;
    let through = atmospheric(lit, camera.position.xyz, point.world, -point.view);
    var shaded: Shaded;
    shaded.linear = through.rgb;
    shaded.ambient = saturate(luminance(skylight) / max(luminance(lit), TOON_DARK))
        * (1.0 - through.a);
    shaded.normal = point.normal;
    shaded.roughness = point.roughness;
    return shaded;
}
