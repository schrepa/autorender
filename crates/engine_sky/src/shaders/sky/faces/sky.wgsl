import package::face::sky_cloud_shadows::cloud_shadow;
import package::generated::sky::MIRROR_LAYER;
import package::sky::air::{air_light, behind_clouds, fog, height_depth, marched_reach};
import package::sky::environment::{environment_maps, environment_sampler};
import package::sky::light::{sky_air_light, sky_cloud};
import package::sky::piece::sky_piece;

fn sky_average(normal: vec3<f32>) -> vec3<f32> {
    return mix(sky_above(normal), sky_piece.ground.rgb, max(-normal.y, 0.0));
}

fn sky_above(normal: vec3<f32>) -> vec3<f32> {
    let side = mix(sky_piece.horizon.rgb, sky_piece.zenith.rgb, 0.5);
    return mix(side, sky_piece.zenith.rgb, max(normal.y, 0.0));
}

fn atmospheric(light: vec3<f32>, eye: vec3<f32>, point: vec3<f32>, direction: vec3<f32>)
        -> vec4<f32> {
    let air_lit = air_light(direction);
    let ray = point - eye;
    let distance = length(ray);
    let reached = eye + ray * (min(marched_reach(), distance) / max(distance, 1.0e-6));
    let fogged = fog(eye, point);
    var color = light;
    if fogged != 0.0 {
        color = mix(light, air_lit.fog, fogged) + air_lit.fog_sun * (fogged - fog(eye, reached));
    }
    var covered = fogged;
    if sky_piece.aerial.y > 0.0 {
        let depth = height_depth(eye, point, sky_piece.aerial.x, sky_piece.aerial.y,
            sky_piece.aerial.z);
        if depth != 0.0 {
            let air = 1.0 - exp(-depth);
            let marched = 1.0 - exp(-height_depth(eye, reached, sky_piece.aerial.x,
                sky_piece.aerial.y, sky_piece.aerial.z));
            color = mix(color, air_lit.air, air) + air_lit.air_sun * (air - marched);
            covered = fogged + (1.0 - fogged) * air;
        }
    }
    return vec4<f32>(color, covered);
}

fn atmospheric_cover(eye: vec3<f32>, point: vec3<f32>) -> f32 {
    let fogged = fog(eye, point);
    if sky_piece.aerial.y <= 0.0 {
        return fogged;
    }
    let depth = height_depth(eye, point, sky_piece.aerial.x, sky_piece.aerial.y,
        sky_piece.aerial.z);
    return fogged + (1.0 - fogged) * (1.0 - exp(-depth));
}

fn cloud_passing(world: vec3<f32>) -> f32 {
    return cloud_shadow(world);
}

fn mirrored_along(start: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    return behind_clouds(sky_air_light(direction), sky_cloud(MIRROR_LAYER, start, direction),
        direction);
}

fn environment_levels() -> u32 {
    return textureNumLevels(environment_maps);
}

fn environment_size() -> u32 {
    return textureDimensions(environment_maps).x;
}

fn environment_sample(direction: vec3<f32>, layer: u32, level: f32) -> vec3<f32> {
    return textureSampleLevel(environment_maps, environment_sampler, direction, layer, level).rgb;
}

fn zenith() -> vec3<f32> {
    return sky_piece.zenith.rgb;
}

fn horizon() -> vec3<f32> {
    return sky_piece.horizon.rgb;
}

fn ambient() -> f32 {
    return sky_piece.zenith.w;
}

fn sunlight(authored: vec3<f32>) -> vec3<f32> {
    return select(authored, sky_piece.sun_light.rgb, sky_piece.state.y != 0.0);
}

fn adapted_exposure(authored: f32) -> f32 {
    return select(authored, sky_piece.horizon.w, sky_piece.state.y != 0.0);
}
