import package::face::sun::sun_direction;
import package::generated::pieces::pieces;
import package::sky::air::{
    CLOUD_SHADOWS, CLOUD_SHADOW_CELLS, air_light, behind_clouds, cloud_shadow_cell, fog,
    height_depth, marched_reach
};
import package::sky::environment::{environment_maps, environment_sampler};
import package::sky::light::{MIRROR_LAYER, sky_air_light, sky_cloud};

fn sky_average(normal: vec3<f32>) -> vec3<f32> {
    return mix(sky_above(normal), pieces.sky.ground.rgb, max(-normal.y, 0.0));
}

fn sky_above(normal: vec3<f32>) -> vec3<f32> {
    let side = mix(pieces.sky.horizon.rgb, pieces.sky.zenith.rgb, 0.5);
    return mix(side, pieces.sky.zenith.rgb, max(normal.y, 0.0));
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
    if pieces.sky.aerial.y > 0.0 {
        let depth = height_depth(eye, point, pieces.sky.aerial.x, pieces.sky.aerial.y,
            pieces.sky.aerial.z);
        if depth != 0.0 {
            let air = 1.0 - exp(-depth);
            let marched = 1.0 - exp(-height_depth(eye, reached, pieces.sky.aerial.x,
                pieces.sky.aerial.y, pieces.sky.aerial.z));
            color = mix(color, air_lit.air, air) + air_lit.air_sun * (air - marched);
            covered = fogged + (1.0 - fogged) * air;
        }
    }
    return vec4<f32>(color, covered);
}

fn atmospheric_cover(eye: vec3<f32>, point: vec3<f32>) -> f32 {
    let fogged = fog(eye, point);
    if pieces.sky.aerial.y <= 0.0 {
        return fogged;
    }
    let depth = height_depth(eye, point, pieces.sky.aerial.x, pieces.sky.aerial.y,
        pieces.sky.aerial.z);
    return fogged + (1.0 - fogged) * (1.0 - exp(-depth));
}

fn cloud_passing(world: vec3<f32>) -> f32 {
    let strength = pieces.sky.cloud_shadow.w;
    if !CLOUD_SHADOWS || strength <= 0.0 || sun_direction().y <= 0.01 {
        return 1.0;
    }
    let rise = max(pieces.sky.cloud_layer.x - world.y, 0.0);
    let base = world.xz + sun_direction().xz * (rise / sun_direction().y);
    let size = f32(CLOUD_SHADOW_CELLS);
    let cells = base * pieces.sky.cloud_shadow.z - 0.5;
    let local = cells - pieces.sky.cloud_shadow.xy;
    if any(local < vec2<f32>(0.0)) || any(local > vec2<f32>(size - 1.0)) {
        return 1.0;
    }
    let count = i32(CLOUD_SHADOW_CELLS);
    let first = vec2<i32>(floor(cells));
    let a = vec2<u32>(first & vec2<i32>(count - 1));
    let b = vec2<u32>((first + vec2<i32>(1)) & vec2<i32>(count - 1));
    let f = cells - floor(cells);
    let passing = mix(
        mix(cloud_shadow_cell(a.x, a.y), cloud_shadow_cell(b.x, a.y), f.x),
        mix(cloud_shadow_cell(a.x, b.y), cloud_shadow_cell(b.x, b.y), f.x), f.y);
    let edge = min(local, vec2<f32>(size - 1.0) - local) / (0.1 * size);
    let fade = smoothstep(0.0, 1.0, saturate(min(edge.x, edge.y)));
    return mix(1.0, passing, strength * fade);
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
    return pieces.sky.zenith.rgb;
}

fn horizon() -> vec3<f32> {
    return pieces.sky.horizon.rgb;
}

fn ambient() -> f32 {
    return pieces.sky.zenith.w;
}

fn sunlight(authored: vec3<f32>) -> vec3<f32> {
    return select(authored, pieces.sky.sun_light.rgb, pieces.sky.state.y != 0.0);
}

fn adapted_exposure(authored: f32) -> f32 {
    return select(authored, pieces.sky.horizon.w, pieces.sky.state.y != 0.0);
}
