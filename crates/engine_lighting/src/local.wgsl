import package::face::lighting::{distribution, fresnel, visibility};
import package::face::lights::{LIGHT_EXTERIOR, LIGHT_INTERIOR, Light};
import package::face::probes::in_interior;
import package::face::surface_point::SurfacePoint;
import package::generated::pieces::pieces;
import package::lighting::surface::{cap_share, disk_energy, dominant_direction};

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
    let radius = select(0.0, max(light.radius, pieces.lighting.highlights.w), shaped);
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
        highlight *= cap_share(highlight * (pieces.lighting.model.z * falloff) * light.color);
    }
    return (point.diffuse_color * n_l + highlight * pieces.lighting.model.z) * light.color
        * falloff;
}
