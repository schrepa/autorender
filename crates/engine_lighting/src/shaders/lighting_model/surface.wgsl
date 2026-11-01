import package::engine::color::luminance;
import package::engine::shading::{Shaded, SurfacePoint};
import package::engine::view::camera;
import package::face::ambient::{EnvironmentAt, environment};
import package::face::sky::atmospheric;
import package::face::sun::{sun_color, sun_direction};
import package::face::tonemap::exposure;
import package::generated::pieces::lighting_model::piece;
import package::lighting_model::faces::lighting::{distribution, environment_brdf, fresnel,
    sun_visibility, surface_sun, visibility};
import package::lighting_model::inputs::{ambient_light, environment_specular, no_environment};

const TRANSMISSION_BEND: f32 = 0.3;
const TRANSMISSION_POWER: f32 = 4.0;

fn transmitted(point: SurfacePoint) -> vec3<f32> {
    let strength = point.transmission;
    if strength <= 0.0 {
        return vec3<f32>(0.0);
    }
    let bent = sun_direction() + point.normal * TRANSMISSION_BEND;
    let through = saturate(dot(point.view, -normalize(bent)));
    return point.diffuse_color * (strength * pow(through, TRANSMISSION_POWER));
}

fn transmission_occlusion(point: SurfacePoint) -> f32 {
    return 1.0 - 0.5 * saturate(point.transmission);
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

fn shade_surface(point: SurfacePoint, local: vec3<f32>) -> Shaded {
    let normal = point.normal;
    let view = point.view;
    let sun = sun_direction();
    let n_l = dot(normal, sun);
    let specular = piece().model.z;
    var facing = 1.0;
    let face = cross(point.world_dx, point.world_dy); // filter: none, the face's own plane across the pixel, a direction
    if dot(face, face) > 1.0e-20 && all(point.shadow_offset == vec3<f32>(0.0))
        && !point.ground {
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
    let skyward = normalize(normal + point.sky_turn);
    let light = environment(no_environment(), EnvironmentAt(point.world, normal, skyward,
        dominant_direction(point), point.roughness, point.alpha, point.ground));
    let brdf = environment_brdf(point.f0, point.roughness, point.n_v);
    let reflection = environment_specular(light) * brdf * specular;
    let grazing = 1.0 - max(dot(point.geometric, view), 0.0);
    let rim = (grazing * grazing) * (grazing * grazing) * piece().model.x
        * (1.0 - abs(point.geometric.y));
    let inside = light.inside;
    let open = light.open;
    let bounce = point.indirect + light.bounce * point.occlusion;
    let sky_light = (point.diffuse_color * (ambient_light(light) + light.horizon * rim)
        * inside.x + reflection) * point.occlusion * open
        + point.diffuse_color * bounce * open * inside.x;
    let sunlit = (point.diffuse_color * wrapped + highlight * specular + transmitted(point))
        * sun_color();
    var visible = 1.0;
    if any(sunlit != vec3<f32>(0.0)) && point.sun_shade != 1.0 && inside.y != 0.0 {
        visible = sun_visibility(surface_sun(point));
    }
    let sun_light = sunlit * visible * (1.0 - point.sun_shade) * inside.y;
    let emitted = (point.albedo * point.emission + point.emissive) * piece().model.y;
    let lit = sky_light + sun_light + emitted + local;
    let through = atmospheric(lit, camera.position.xyz, point.world, -view);
    let fogged = through.rgb;
    let covered = through.a;
    var shaded: Shaded;
    shaded.normal = normal;
    shaded.roughness = point.roughness;
    shaded.across = point.roughness - point.smoother_across;
    shaded.linear = fogged;
    shaded.ambient = saturate(luminance(sky_light) / max(luminance(lit), 1.0e-6)) * (1.0 - covered)
        * transmission_occlusion(point);
    shaded.reflect = saturate(luminance(brdf) * specular) * (1.0 - covered);
    shaded.mirrored = saturate(luminance(reflection * (point.occlusion * open))
        / max(luminance(sky_light), 1.0e-6));
    return shaded;
}
