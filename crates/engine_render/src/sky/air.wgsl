import package::frame::view::{camera, lighting};
import package::shadows::sample::{level_point, level_slots, light_space, shadow, shadow_atlas,
    shadow_level, shadow_sampler, slot_texel};
import package::sky::environment::{environment_maps, environment_sampler};

override CLOUD_SHADOWS: bool = true;

fn haze(direction: vec3<f32>) -> vec3<f32> {
    return haze_sky(direction) + sun_halo(direction);
}

fn haze_sky(direction: vec3<f32>) -> vec3<f32> {
    let up = direction.y;
    let color = mix(lighting.horizon.rgb, lighting.zenith.rgb, sqrt(max(up, 0.0)));
    return mix(color, lighting.ground.rgb, min(max(-up, 0.0) * 1.5, 0.35));
}

fn sun_halo(direction: vec3<f32>) -> vec3<f32> {
    let toward = max(dot(direction, lighting.sun.xyz), 0.0);
    let square = toward * toward;
    return lighting.sun_color.rgb * (0.12 * square * square * square);
}

fn sun_disc(direction: vec3<f32>) -> vec3<f32> {
    let toward = max(dot(direction, lighting.sun.xyz), 0.0);
    return lighting.sun_color.rgb * (6.0 * pow(toward, 900.0));
}

const CLOUD_HORIZON: f32 = 0.05;

fn cloud_dome_uv(direction: vec3<f32>) -> vec2<f32> {
    let around = atan2(direction.z, direction.x) * (1.0 / 6.2831853);
    let low = -asin(CLOUD_HORIZON);
    let elevation = asin(clamp(direction.y, -1.0, 1.0));
    return vec2<f32>(fract(around), saturate((elevation - low) / (1.5707963 - low)));
}

struct CloudDomes {
    older: vec4<f32>,
    newer: vec4<f32>,
    older_mirror: vec4<f32>,
    newer_mirror: vec4<f32>,
    eye: vec4<f32>,
    view: vec4<f32>,
}

const CLOUD_PARALLAX_REACH: f32 = 60000.0;

fn cloud_dome_direction(domes: CloudDomes, origin: vec3<f32>, start: vec3<f32>,
        direction: vec3<f32>) -> vec3<f32> {
    let rise = domes.eye.w - start.y;
    if direction.y * rise <= 0.0 || abs(direction.y) < 1.0e-5 {
        return direction;
    }
    let along = min(rise / direction.y, CLOUD_PARALLAX_REACH);
    let toward = start + direction * along - origin;
    let reach = length(toward);
    if reach < 1.0e-3 {
        return direction;
    }
    return toward / reach;
}

fn behind_clouds(light: vec3<f32>, cloud: vec4<f32>, direction: vec3<f32>) -> vec3<f32> {
    let shown = smoothstep(-CLOUD_HORIZON, -0.4 * CLOUD_HORIZON, direction.y);
    return mix(light, light * cloud.a + cloud.rgb, shown);
}

fn sky_average(normal: vec3<f32>) -> vec3<f32> {
    return mix(sky_above(normal), lighting.ground.rgb, max(-normal.y, 0.0));
}

fn sky_above(normal: vec3<f32>) -> vec3<f32> {
    let side = mix(lighting.horizon.rgb, lighting.zenith.rgb, 0.5);
    return mix(side, lighting.zenith.rgb, max(normal.y, 0.0));
}

fn height_depth(eye: vec3<f32>, point: vec3<f32>, start: f32, density: f32, falloff: f32)
        -> f32 {
    let ray = point - eye;
    let distance = length(ray);
    if distance <= start {
        return 0.0;
    }
    let from_height = max(eye.y + ray.y * (start / distance), -1000.0);
    let to_height = max(point.y, -1000.0);
    let rise = falloff * (to_height - from_height);
    let near = exp(-falloff * from_height);
    let far = exp(-falloff * to_height);
    let mean = select(near, (near - far) / rise, abs(rise) > 1.0e-4);
    return density * (distance - start) * mean;
}

fn fog(eye: vec3<f32>, point: vec3<f32>) -> f32 {
    let depth = height_depth(eye, point, lighting.fog.x, lighting.fog.y, lighting.fog.z);
    return min(1.0 - exp(-depth), lighting.fog.w);
}

struct AirLight {
    fog: vec3<f32>,
    fog_sun: vec3<f32>,
    air: vec3<f32>,
    air_sun: vec3<f32>,
}

fn air_light(direction: vec3<f32>) -> AirLight {
    var light: AirLight;
    if hazy() {
        let behind = sky_behind(direction);
        let glow = min(sky_glow(direction), behind);
        light.fog = behind - glow;
        light.fog_sun = glow;
        light.air = light.fog;
        light.air_sun = glow;
        return light;
    }
    light.fog = haze_sky(direction);
    light.fog_sun = sun_halo(direction);
    light.air = lighting.aerial_color.rgb;
    light.air_sun = air_glow(direction);
    return light;
}

fn air_glow(direction: vec3<f32>) -> vec3<f32> {
    let toward = max(dot(direction, lighting.sun.xyz), 0.0);
    let broad = toward * toward;
    let lobe = broad * broad;
    return lighting.sun_color.rgb * (lighting.aerial.w * lobe * (0.25 + 0.75 * lobe));
}

fn sky_glow(direction: vec3<f32>) -> vec3<f32> {
    if lighting.grade.z > 0.5 {
        return air_glow(direction);
    }
    return sun_halo(direction);
}

fn marched_reach() -> f32 {
    return select(lighting.haze_march.x, 0.0, camera.options.x != 0.0);
}

const HAZE_BLUR: f32 = 0.3;
const HAZE_LIFT: f32 = 0.02;

fn hazy() -> bool {
    return lighting.aerial_color.w != 0.0;
}

fn sky_behind(direction: vec3<f32>) -> vec3<f32> {
    let lifted = normalize(vec3<f32>(direction.x, max(direction.y, HAZE_LIFT), direction.z));
    let level = HAZE_BLUR * f32(textureNumLevels(environment_maps) - 1u);
    return textureSampleLevel(environment_maps, environment_sampler, lifted, 0, level).rgb;
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
    if lighting.aerial.y > 0.0 {
        let depth = height_depth(eye, point, lighting.aerial.x, lighting.aerial.y,
            lighting.aerial.z);
        if depth != 0.0 {
            let air = 1.0 - exp(-depth);
            let marched = 1.0 - exp(-height_depth(eye, reached, lighting.aerial.x,
                lighting.aerial.y, lighting.aerial.z));
            color = mix(color, air_lit.air, air) + air_lit.air_sun * (air - marched);
            covered = fogged + (1.0 - fogged) * air;
        }
    }
    return vec4<f32>(color, covered);
}

fn atmospheric_cover(eye: vec3<f32>, point: vec3<f32>) -> f32 {
    let fogged = fog(eye, point);
    if lighting.aerial.y <= 0.0 {
        return fogged;
    }
    let depth = height_depth(eye, point, lighting.aerial.x, lighting.aerial.y,
        lighting.aerial.z);
    return fogged + (1.0 - fogged) * (1.0 - exp(-depth));
}

const CLOUD_SHADOW_CELLS: u32 = 128u;
@group(0) @binding(19) var<storage, read> cloud_shadows: array<u32, CLOUD_SHADOW_CELLS * CLOUD_SHADOW_CELLS / 2u>;

fn cloud_shadow_cell(x: u32, z: u32) -> f32 {
    let index = z * CLOUD_SHADOW_CELLS + x;
    return unpack2x16float(cloud_shadows[index / 2u])[index % 2u];
}

fn cloud_passing(world: vec3<f32>) -> f32 {
    let strength = lighting.cloud_shadow.w;
    if !CLOUD_SHADOWS || strength <= 0.0 || lighting.sun.y <= 0.01 {
        return 1.0;
    }
    let rise = max(lighting.cloud_layer.x - world.y, 0.0);
    let base = world.xz + lighting.sun.xz * (rise / lighting.sun.y);
    let size = f32(CLOUD_SHADOW_CELLS);
    let cells = base * lighting.cloud_shadow.z - 0.5;
    let local = cells - lighting.cloud_shadow.xy;
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

fn air_visibility(world: vec3<f32>) -> f32 {
    let pick = shadow_level(world);
    let light = light_space(world);
    for (var level = pick.level; level < u32(shadow.level_state.x); level++) {
        let at = level_point(level, light);
        if !at.inside {
            continue;
        }
        let slots = level_slots(level, at);
        if slots.x == 0u {
            continue;
        }
        let still = slot_texel(slots.x - 1u, at.across) * shadow.atlas.zw;
        var visible = textureSampleCompareLevel(shadow_atlas, shadow_sampler, still, at.depth);
        if slots.y != 0u {
            let moving = slot_texel(slots.y - 1u, at.across) * shadow.atlas.zw;
            visible = min(visible,
                textureSampleCompareLevel(shadow_atlas, shadow_sampler, moving, at.depth));
        }
        return visible;
    }
    return 1.0;
}
