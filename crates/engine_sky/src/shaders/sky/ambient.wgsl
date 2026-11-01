import package::engine::view::PI;
import package::generated::sky::MIRROR_LAYER;
import package::place::ambient::{EnvironmentAt, EnvironmentLight};
import package::sky::air::behind_clouds;
import package::sky::environment::environment_maps;
import package::sky::faces::sky::{ambient, environment_levels, environment_sample, horizon,
    sky_above, sky_average};
import package::sky::light::{sky_air_light, sky_cloud};

struct EnvironmentLobe {
    direction: vec3<f32>,
    level: f32,
    texels: f32,
}

fn environment_size() -> u32 {
    return textureDimensions(environment_maps).x;
}

fn mirrored_along(start: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    return behind_clouds(sky_air_light(direction), sky_cloud(MIRROR_LAYER, start, direction),
        direction);
}

fn environment_lobe(at: EnvironmentAt) -> EnvironmentLobe {
    let levels = environment_levels();
    return EnvironmentLobe(at.lobe, at.roughness * f32(levels - 1u), lobe_texels(at.alpha));
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

fn environment(light: EnvironmentLight, at: EnvironmentAt) -> EnvironmentLight {
    var result = light;
    result.sky = sky_average(at.skyward) * ambient();
    result.upper = sky_above(at.skyward) * ambient();
    result.horizon = horizon();
    result.sky_specular = mirrored_sky(environment_lobe(at), at.world);
    return result;
}
