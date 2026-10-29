import package::frame::view::lighting;
import package::lighting::brdf::{distribution, fresnel, visibility};
import package::lighting::surface::{SurfacePoint, cap_share, disk_energy, dominant_direction,
    in_interior};

struct Clusters {
    view_projection: mat4x4<f32>,
    depth: vec4<f32>,
    projection: vec4<f32>,
    grid: vec4<f32>,
    counts: vec4<u32>,
    lights: vec4<u32>,
}
struct Light {
    position: vec3<f32>,
    range: f32,
    color: vec3<f32>,
    cos_outer: f32,
    direction: vec3<f32>,
    cos_inner: f32,
    radius: f32,
    length: f32,
    region: u32,
    pad: u32,
}
const SHAPED_LIGHTS: u32 = 1u;
const KEPT_REGIONS: u32 = 2u;
const LIGHT_INTERIOR: u32 = 1u;
const LIGHT_EXTERIOR: u32 = 2u;
@group(3) @binding(0) var<uniform> clusters: Clusters;
@group(3) @binding(1) var<storage, read> lights: array<Light>;
@group(3) @binding(2) var<storage, read> cluster_ranges: array<vec2<u32>>;
@group(3) @binding(3) var<storage, read> cluster_lights: array<u32>;

fn tube_energy(alpha: f32, angle: f32) -> f32 {
    return alpha / saturate(alpha + 0.5 * angle);
}

fn shaped_light(point: SurfacePoint, light: Light, shaped: bool, kept: bool,
        inside: ptr<function, f32>) -> vec3<f32> {
    if kept && light.region != 0u {
        if *inside < 0.0 {
            *inside = select(0.0, 1.0, in_interior(point.world));
        }
        let in_box = *inside > 0.0;
        if (light.region == LIGHT_INTERIOR && !in_box)
            || (light.region == LIGHT_EXTERIOR && in_box) {
            return vec3<f32>(0.0);
        }
    }
    let radius = select(0.0, max(light.radius, lighting.highlights.w), shaped);
    let span = select(vec3<f32>(0.0), light.direction * light.length, shaped);
    let first = light.position - point.world;
    let last = first + span;
    let spanned = dot(span, span);
    let nearest = first + span * saturate(-dot(first, span) / max(spanned, 1.0e-8));
    let d2 = dot(nearest, nearest);
    let r2 = light.range * light.range;
    if d2 >= r2 {
        return vec3<f32>(0.0);
    }
    let ratio = d2 / r2;
    let window = saturate(1.0 - ratio * ratio);
    let first_length = max(length(first), 1.0e-4);
    let last_length = max(length(last), 1.0e-4);
    let n_l = saturate(0.5 * (dot(point.normal, first) / first_length
        + dot(point.normal, last) / last_length));
    var falloff = window * window
        / max(0.5 * (first_length * last_length + dot(first, last)), 0.01);
    if light.cos_outer > -1.5 {
        let center = first * inverseSqrt(max(dot(first, first), 1.0e-8));
        falloff *= smoothstep(light.cos_outer, light.cos_inner, dot(-center, light.direction));
    }
    if falloff == 0.0 {
        return vec3<f32>(0.0);
    }
    let reflected = dominant_direction(point);
    var toward = first;
    if spanned > 1.0e-8 {
        let across = dot(reflected, span);
        toward = first + span * saturate((dot(first, reflected) * across - dot(first, span))
            / max(spanned - across * across, 1.0e-8));
    }
    if radius > 0.0 {
        let to_ray = reflected * dot(toward, reflected) - toward;
        toward += to_ray * saturate(radius / max(length(to_ray), 1.0e-6));
    }
    let distance = max(length(first + 0.5 * span), 1.0e-4);
    let l = normalize(toward);
    let n_l_highlight = dot(point.normal, l);
    var highlight = vec3<f32>(0.0);
    if n_l_highlight > 0.0 {
        let half_vector = normalize(point.view + l);
        let n_h = max(dot(point.normal, half_vector), 0.0);
        let energy = disk_energy(point.alpha, saturate(radius / distance))
            * tube_energy(point.alpha, saturate(sqrt(spanned) / distance));
        let lobe = distribution(n_h, point.alpha)
            * visibility(point.n_v, n_l_highlight, point.alpha) * energy;
        highlight = fresnel(point.f0, max(dot(point.view, half_vector), 0.0))
            * (lobe * n_l_highlight);
        highlight *= cap_share(highlight * (lighting.sun_color.w * falloff) * light.color);
    }
    return (point.diffuse_color * n_l + highlight * lighting.sun_color.w) * light.color
        * falloff;
}

fn local_light(point: SurfacePoint, pixel: vec2<f32>) -> vec3<f32> {
    let depth = dot(clusters.depth.xyz, point.world) + clusters.depth.w;
    let end = clusters.projection.w;
    if depth >= end {
        return vec3<f32>(0.0);
    }
    let counts = clusters.counts;
    let tile = min(vec2<u32>(max(pixel, vec2<f32>(0.0)) / clusters.grid.xy),
        counts.xy - vec2<u32>(1u));
    let slice = u32(clamp(floor(log2(max(depth, 1.0e-6)) * clusters.grid.z + clusters.grid.w),
        0.0, f32(counts.z - 1u)));
    let cluster = (slice * counts.y + tile.y) * counts.x + tile.x;
    let range = cluster_ranges[cluster];
    let specular = lighting.sun_color.w;
    let shaped = (clusters.lights.y & SHAPED_LIGHTS) != 0u;
    let kept = (clusters.lights.y & KEPT_REGIONS) != 0u;
    let capped = lighting.highlights.z > 0.0;
    var inside = -1.0;
    var sum = vec3<f32>(0.0);
    for (var n = 0u; n < range.y; n++) {
        let light = lights[cluster_lights[range.x + n]];
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
