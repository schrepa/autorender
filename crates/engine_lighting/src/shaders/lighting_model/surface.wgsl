import package::engine::color::luminance;
import package::engine::math::PI;
import package::engine::shading::{Shaded, SurfacePoint};
import package::face::ambient::{EnvironmentAt, EnvironmentLight, environment};
import package::face::ambient_occlusion::ambient_occlusion;
import package::face::sky_visibility::sky_visibility;
import package::face::sun::{sun_color, sun_direction};
import package::face::tonemap::exposure;
import package::generated::pieces::lighting_model::piece;
import package::lighting_model::faces::lighting::{distribution, environment_brdf, fresnel,
    sun_visibility, surface_sun, visibility};
import package::lighting_model::inputs::{ambient_light, environment_specular, no_environment};
import package::place::lighting::Reflecting;

struct Crossing {
    face: vec3<f32>,
    light: vec3<f32>,
    passes: bool,
    tangled: f32,
}

fn transmitted(point: SurfacePoint) -> vec3<f32> {
    let strength = point.transmission * piece().model.w;
    if strength <= 0.0 {
        return vec3<f32>(0.0);
    }
    let lobe = piece().thin.zw;
    let bent = sun_direction() + point.normal * lobe.x;
    let through = saturate(dot(point.view, -normalize(bent)));
    return point.diffuse_color * (strength * pow(through, lobe.y));
}

fn crossed(albedo: vec3<f32>) -> vec3<f32> {
    let peak = max(max(albedo.r, albedo.g), max(albedo.b, 1.0e-4));
    let kept = max(albedo / peak, vec3<f32>(1.0e-4));
    return albedo * pow(kept, vec3<f32>(piece().thin.y));
}

fn crossing_of(point: SurfacePoint) -> Crossing {
    var crossing: Crossing;
    let plane = cross(point.world_dx, point.world_dy); // filter: none, the face's own plane across the pixel, a direction
    let faced = dot(plane, plane) > 1.0e-20 && point.thickness == 0.0 && !point.ignore_face;
    if faced {
        crossing.face = normalize(plane * select(-1.0, 1.0, dot(plane, point.geometric) >= 0.0));
    }
    crossing.tangled = saturate(point.tangled);
    if faced || crossing.tangled > 0.0 {
        let share = piece().thin.x * saturate(point.transmission);
        if share > 0.0 {
            crossing.light = crossed(point.diffuse_color) * share;
            crossing.passes = true;
        }
    }
    return crossing;
}

fn crossing_back(crossing: Crossing, back: f32, along: f32) -> f32 {
    return mix(back, 0.25 * (1.0 - along), crossing.tangled);
}

fn transmission_occlusion(point: SurfacePoint) -> f32 {
    return 1.0 - 0.5 * saturate(point.transmission);
}

fn interreflected(open: f32, albedo: vec3<f32>) -> vec3<f32> {
    let a = 2.0404 * albedo - 0.3324;
    let b = -4.7951 * albedo + 0.6417;
    let c = 2.7552 * albedo + 0.6903;
    return max(vec3<f32>(open), ((open * a + b) * open + c) * open);
}

fn sized_highlights() -> bool {
    return piece().highlights.y != 0.0;
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
    let cap = piece().highlights.z;
    let bright = luminance(light) * exposure();
    let knee = 0.5 * cap;
    if cap <= 0.0 || bright <= knee {
        return 1.0;
    }
    let over = bright - knee;
    return (knee + over / (1.0 + over / knee)) / bright;
}

fn sun_highlight(point: SurfacePoint, sun: vec3<f32>, n_l: f32) -> vec3<f32> {
    let cosine = piece().highlights.x;
    let sine = piece().highlights.y;
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

fn environment_of(point: SurfacePoint) -> EnvironmentLight {
    let skyward = normalize(point.normal + point.ambient_bend);
    return environment(no_environment(), EnvironmentAt(point.world, point.normal, skyward,
        dominant_direction(point), point.roughness, point.alpha, point.kind));
}

fn environment_reflectance(point: SurfacePoint) -> vec3<f32> {
    return environment_brdf(point.f0, point.roughness, point.n_v) * piece().model.z;
}

fn reflected(point: SurfacePoint) -> Reflecting {
    var result: Reflecting;
    result.across = point.roughness - point.smoother_across;
    let light = environment_of(point);
    let reflectance = environment_reflectance(point);
    result.light = environment_specular(light) * reflectance * (point.occlusion * light.open);
    result.share = saturate(luminance(reflectance));
    return result;
}

fn shade_surface(point: SurfacePoint, pixel: vec2<f32>, crossing: Crossing, local: vec3<f32>)
        -> Shaded {
    let normal = point.normal;
    let view = point.view;
    let sun = sun_direction();
    let n_l = dot(normal, sun);
    let specular = piece().model.z;
    var facing = 1.0;
    var back = 0.0;
    if any(crossing.face != vec3<f32>(0.0)) {
        let toward = dot(crossing.face, sun);
        facing = smoothstep(-0.05, 0.1, toward);
        back = max(-toward, 0.0);
    }
    if crossing.tangled > 0.0 {
        let along = clamp(dot(view, sun), -1.0, 1.0);
        facing = mix(facing, 1.0 - acos(along) / PI, crossing.tangled);
        back = crossing_back(crossing, back, along);
    }
    let lambert = max(n_l, 0.0);
    let around = saturate(point.transmission);
    let wrapped = lambert + (max((n_l + around) / (1.0 + around), 0.0) - lambert) * facing;
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
    let light = environment_of(point);
    let reflection = environment_specular(light) * environment_reflectance(point);
    let grazing = 1.0 - max(dot(point.geometric, view), 0.0);
    let rim = (grazing * grazing) * (grazing * grazing) * piece().model.x
        * (1.0 - abs(point.geometric.y));
    let inside = light.inside;
    let open = light.open;
    let bounce = point.added_indirect + light.bounce * point.occlusion;
    let contact = ambient_occlusion(pixel, point.world);
    let closed = transmission_occlusion(point);
    let seen = mix(1.0, contact * sky_visibility(pixel, point.world), closed);
    let sky_seen = interreflected(seen, point.diffuse_color);
    let bounce_seen = interreflected(mix(1.0, contact, closed), point.diffuse_color);
    let sky_light = (point.diffuse_color * (ambient_light(light) + light.horizon * rim)
        * inside.x * sky_seen + reflection * seen) * point.occlusion * open
        + point.diffuse_color * bounce * open * inside.x * bounce_seen;
    var sunlit = (point.diffuse_color * (wrapped * point.direct_share.x)
        + highlight * (specular * PI * point.direct_share.y) + transmitted(point)) * sun_color();
    if crossing.passes && back > 0.0 {
        sunlit += crossing.light * sun_color() * back;
    }
    var visible = 1.0;
    if any(sunlit != vec3<f32>(0.0)) && point.direct_blocked != 1.0 && !point.direct_known
        && inside.y != 0.0 {
        visible = sun_visibility(surface_sun(point));
    }
    let sun_light = sunlit * visible * (1.0 - point.direct_blocked) * inside.y;
    let emitted = point.emissive * piece().model.y;
    var shaded: Shaded;
    shaded.linear = sky_light + sun_light + emitted + local;
    return shaded;
}
