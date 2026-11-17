import package::engine::math::PI;
import package::engine::shading::{Shaded, SurfacePoint};
import package::engine::surfaces::SURFACE_THIN as THIN_KIND;
import package::face::lights::{cluster_light, kept_regions, light_cluster, lights_on,
    shaped_lights};
import package::face::sun::sun_direction;
import package::face::sun_blockers::{DEBUG_SUN_NONE, SunAt, sun_flags, sun_open};
import package::generated::pieces::lighting_model::piece;
import package::lighting_model::local::shaped_light;
import package::lighting_model::surface::{Crossing, crossing_back, crossing_of, reflected,
    shade_surface};
import package::place::lighting::Reflecting;

const SURFACE_THIN: u32 = THIN_KIND;

fn highlights() -> vec4<f32> {
    return piece().highlights;
}

fn specular() -> f32 {
    return piece().model.z;
}

fn specular_widening() -> vec2<f32> {
    return piece().widening.xy;
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
    return SunAt(point.world + sun_direction() * point.thickness, point.geometric,
        point.world_dx, point.world_dy, point.direct_softness, point.kind,
        point.receiver_lift_first, point.receiver_lifts);
}

fn sun_visibility(at: SunAt) -> f32 {
    return sun_open(1.0, at);
}

fn sun_debug(at: SunAt) -> u32 {
    return sun_flags(DEBUG_SUN_NONE, at);
}

fn local_light(point: SurfacePoint, pixel: vec2<f32>, crossing: Crossing) -> vec3<f32> {
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
            sum += shaped_light(point, crossing, light, shaped, kept, &inside);
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
        var back = 0.0;
        if crossing.passes {
            back = crossing_back(crossing, max(-dot(crossing.face, l), 0.0), dot(point.view, l));
        }
        if n_l <= 0.0 && back <= 0.0 {
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
        if n_l > 0.0 {
            let half_vector = normalize(point.view + l);
            let n_h = max(dot(point.normal, half_vector), 0.0);
            let highlight = fresnel(point.f0, max(dot(point.view, half_vector), 0.0))
                * (distribution(n_h, point.alpha) * visibility(point.n_v, n_l, point.alpha));
            sum += (point.diffuse_color + highlight * specular) * light.color * (falloff * n_l);
        }
        if back > 0.0 {
            sum += crossing.light * light.color * (falloff * back);
        }
    }
    return sum * saturate((end - depth) / (0.1 * end));
}

fn widened(point: SurfacePoint) -> SurfacePoint {
    var result = point;
    let spread = specular_widening();
    let turned = dot(point.normal_dx, point.normal_dx) + dot(point.normal_dy, point.normal_dy);
    let widening = min(2.0 * spread.x * turned, spread.y);
    result.alpha = sqrt(saturate(point.alpha * point.alpha + widening));
    result.roughness = sqrt(result.alpha);
    return result;
}

struct ReflectingKey {
    world: vec3<u32>,
    normal: vec3<u32>,
    normal_dx: vec3<u32>,
    normal_dy: vec3<u32>,
    view: vec3<u32>,
    bend: vec3<u32>,
    f0: vec3<u32>,
    terms: vec4<u32>,
    kind: u32,
}

var<private> held_reflecting: Reflecting;
var<private> held_reflecting_key: ReflectingKey;
var<private> reflecting_held: bool;

fn reflecting_key(point: SurfacePoint) -> ReflectingKey {
    return ReflectingKey(bitcast<vec3<u32>>(point.world), bitcast<vec3<u32>>(point.normal),
        bitcast<vec3<u32>>(point.normal_dx), bitcast<vec3<u32>>(point.normal_dy),
        bitcast<vec3<u32>>(point.view), bitcast<vec3<u32>>(point.ambient_bend),
        bitcast<vec3<u32>>(point.f0), bitcast<vec4<u32>>(vec4<f32>(point.alpha,
        point.smoother_across, point.occlusion, point.n_v)), point.kind);
}

fn same_reflecting_key(a: ReflectingKey, b: ReflectingKey) -> bool {
    return all(a.world == b.world) && all(a.normal == b.normal)
        && all(a.normal_dx == b.normal_dx) && all(a.normal_dy == b.normal_dy)
        && all(a.view == b.view) && all(a.bend == b.bend) && all(a.f0 == b.f0)
        && all(a.terms == b.terms) && a.kind == b.kind;
}

fn reflecting(point: SurfacePoint) -> Reflecting {
    if reflecting_held && same_reflecting_key(reflecting_key(point), held_reflecting_key) {
        return held_reflecting;
    }
    return reflected(widened(point));
}

fn lit(surface: SurfacePoint, pixel: vec2<f32>) -> Shaded {
    let point = widened(surface);
    let crossing = crossing_of(point);
    let lighted = shade_surface(point, pixel, crossing,
        local_light(point, pixel, crossing) + point.added_light);
    held_reflecting = lighted.reflecting;
    held_reflecting_key = reflecting_key(surface);
    reflecting_held = true;
    return lighted.shaded;
}

fn given_reflectance(point: SurfacePoint, reflectance: vec3<f32>, diffuse: vec3<f32>)
        -> SurfacePoint {
    var result = point;
    result.f0 = reflectance;
    result.diffuse_color = diffuse;
    return result;
}
