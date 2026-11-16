import package::sky::environment::{environment_maps, environment_sampler};
import package::sky::piece::sky_piece;

fn sky_toward(normal: vec3<f32>) -> vec3<f32> {
    let rise = select(sky_piece.sky_flat.rgb, sky_piece.sky_y.rgb, normal.y >= 0.0);
    return max(sky_piece.sky_flat.rgb + sky_piece.sky_x.rgb * normal.x + rise * normal.y
        + sky_piece.sky_z.rgb * normal.z, vec3<f32>(0.0));
}

fn sky_average(normal: vec3<f32>) -> vec3<f32> {
    return sky_toward(normal) + sky_piece.ground.rgb * (0.5 * (1.0 - normal.y));
}

fn sky_above(normal: vec3<f32>) -> vec3<f32> {
    let lifted = vec3<f32>(normal.x, max(normal.y, 0.0), normal.z);
    return sky_toward(lifted) / (0.5 * (1.0 + lifted.y));
}

fn environment_levels() -> u32 {
    return textureNumLevels(environment_maps);
}

fn environment_sample(direction: vec3<f32>, level: f32) -> vec3<f32> {
    return textureSampleLevel(environment_maps, environment_sampler, direction, 0u, level).rgb;
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
