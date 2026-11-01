import package::engine::shading::{SPECULAR_LIMIT, SPECULAR_VARIANCE, Shaded, SurfacePoint};
import package::engine::view::PI;
import package::face::lights::{cluster_light, kept_regions, light_cluster, lights_on,
    shaped_lights};
import package::face::sun_blockers::{DEBUG_SUN_NONE, SunAt, SunOpen, sun_flags, sun_open};
import package::generated::pieces::lighting_model::piece;
import package::lighting_model::inputs::sun_seen;
import package::lighting_model::local::shaped_light;
import package::lighting_model::surface::shade_surface;

fn highlights() -> vec4<f32> {
    return piece().highlights;
}

fn specular() -> f32 {
    return piece().model.z;
}

fn distribution(n_h: f32, alpha: f32) -> f32 {
    let alpha2 = alpha * alpha;
    let d = n_h * n_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * d * d);
}

fn visibility(n_v: f32, n_l: f32, alpha: f32) -> f32 {
    return 0.5 / mix(2.0 * n_l * n_v, n_l + n_v, alpha);
}

fn fresnel(f0: vec3<f32>, cosine: f32) -> vec3<f32> {
    let m = 1.0 - cosine;
    let m2 = m * m;
    return f0 + (vec3<f32>(1.0) - f0) * (m2 * m2 * m);
}

fn environment_brdf(f0: vec3<f32>, roughness: f32, n_v: f32) -> vec3<f32> {
    let r = roughness * vec4<f32>(-1.0, -0.0275, -0.572, 0.022)
        + vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let a004 = min(r.x * r.x, exp2(-9.28 * n_v)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

fn surface_sun(point: SurfacePoint) -> SunAt {
    return SunAt(point.world + point.shadow_offset, point.geometric, point.world_dx,
        point.world_dy, point.shadow_softness, point.ground, point.shadow_lift_first,
        point.shadow_lifts);
}

fn sun_visibility(at: SunAt) -> f32 {
    return sun_seen(sun_open(SunOpen(1.0, 1.0), at));
}

fn sun_debug(at: SunAt) -> u32 {
    return sun_flags(DEBUG_SUN_NONE, at);
}

fn local_light(point: SurfacePoint, pixel: vec2<f32>) -> vec3<f32> {
    if !lights_on() {
        return vec3<f32>(0.0);
    }
    let cluster = light_cluster(point.world, pixel);
    let depth = cluster.depth;
    let end = cluster.end;
    if depth >= end {
        return vec3<f32>(0.0);
    }
    let range = cluster.range;
    let specular = piece().model.z;
    let shaped = shaped_lights();
    let kept = kept_regions();
    let capped = piece().highlights.z > 0.0;
    var inside = -1.0;
    var sum = vec3<f32>(0.0);
    for (var n = 0u; n < range.y; n++) {
        let light = cluster_light(cluster, n);
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

fn widened(point: SurfacePoint, turn_dx: vec3<f32>, turn_dy: vec3<f32>) -> SurfacePoint {
    var result = point;
    let widening = min(2.0 * SPECULAR_VARIANCE * (dot(turn_dx, turn_dx) + dot(turn_dy, turn_dy)),
        SPECULAR_LIMIT);
    result.alpha = sqrt(saturate(point.alpha * point.alpha + widening));
    result.roughness = sqrt(result.alpha);
    return result;
}

fn finish(point: SurfacePoint, local: vec3<f32>) -> Shaded {
    return shade_surface(point, local);
}
