import package::face::sky_cloud_shadows::cloud_shadow;
import package::sky::air::{air_light, height_depth, marched_reach};
import package::sky::environment::{environment_maps, environment_sampler};
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
    let depth = height_depth(eye, point, sky_piece.aerial.y, sky_piece.aerial.z);
    if !(depth > 0.0) {
        return vec4<f32>(light, 0.0);
    }
    let air_lit = air_light(direction);
    let ray = point - eye;
    let distance = length(ray);
    let reached = eye + ray * (min(marched_reach(), distance) / max(distance, 1.0e-6));
    let air = 1.0 - exp(-depth);
    let marched = 1.0 - exp(-height_depth(eye, reached, sky_piece.aerial.y, sky_piece.aerial.z));
    return vec4<f32>(mix(light, air_lit.air, air) + air_lit.air_sun * (air - marched), air);
}

fn atmospheric_cover(eye: vec3<f32>, point: vec3<f32>) -> f32 {
    return 1.0 - exp(-height_depth(eye, point, sky_piece.aerial.y, sky_piece.aerial.z));
}

fn cloud_passing(world: vec3<f32>) -> f32 {
    return cloud_shadow(world);
}

fn environment_levels() -> u32 {
    return textureNumLevels(environment_maps);
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

fn sunlight() -> vec3<f32> {
    return sky_piece.sun_light.rgb;
}

fn adapted_exposure(authored: f32) -> f32 {
    return sky_piece.horizon.w;
}
