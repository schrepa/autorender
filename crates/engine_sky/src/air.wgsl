import package::face::sun::{sun_color, sun_direction};
import package::frame::view::camera;
import package::generated::pieces::pieces;
import package::sky::environment::{environment_maps, environment_sampler};

override CLOUD_SHADOWS: bool = true;

fn haze(direction: vec3<f32>) -> vec3<f32> {
    return haze_sky(direction) + sun_halo(direction);
}

fn haze_sky(direction: vec3<f32>) -> vec3<f32> {
    let up = direction.y;
    let color = mix(pieces.sky.horizon.rgb, pieces.sky.zenith.rgb, sqrt(max(up, 0.0)));
    return mix(color, pieces.sky.ground.rgb, min(max(-up, 0.0) * 1.5, 0.35));
}

fn sun_halo(direction: vec3<f32>) -> vec3<f32> {
    let toward = max(dot(direction, sun_direction()), 0.0);
    let square = toward * toward;
    return sun_color() * (0.12 * square * square * square);
}

fn sun_disc(direction: vec3<f32>) -> vec3<f32> {
    let toward = max(dot(direction, sun_direction()), 0.0);
    return sun_color() * (6.0 * pow(toward, 900.0));
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
    let depth = height_depth(eye, point, pieces.sky.fog.x, pieces.sky.fog.y, pieces.sky.fog.z);
    return min(1.0 - exp(-depth), pieces.sky.fog.w);
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
    light.air = pieces.sky.aerial_color.rgb;
    light.air_sun = air_glow(direction);
    return light;
}

fn air_glow(direction: vec3<f32>) -> vec3<f32> {
    let toward = max(dot(direction, sun_direction()), 0.0);
    let broad = toward * toward;
    let lobe = broad * broad;
    return sun_color() * (pieces.sky.aerial.w * lobe * (0.25 + 0.75 * lobe));
}

fn sky_glow(direction: vec3<f32>) -> vec3<f32> {
    if pieces.sky.state.x > 0.5 {
        return air_glow(direction);
    }
    return sun_halo(direction);
}

fn marched_reach() -> f32 {
    return select(pieces.sky.haze_march.x, 0.0, camera.options.x != 0.0);
}

const HAZE_BLUR: f32 = 0.3;
const HAZE_LIFT: f32 = 0.02;

fn hazy() -> bool {
    return pieces.sky.aerial_color.w != 0.0;
}

fn sky_behind(direction: vec3<f32>) -> vec3<f32> {
    let lifted = normalize(vec3<f32>(direction.x, max(direction.y, HAZE_LIFT), direction.z));
    let level = HAZE_BLUR * f32(textureNumLevels(environment_maps) - 1u);
    return textureSampleLevel(environment_maps, environment_sampler, lifted, 0, level).rgb;
}

const CLOUD_SHADOW_CELLS: u32 = 128u;
@group(0) @binding(19) var<storage, read> cloud_shadows: array<u32, CLOUD_SHADOW_CELLS * CLOUD_SHADOW_CELLS / 2u>;

fn cloud_shadow_cell(x: u32, z: u32) -> f32 {
    let index = z * CLOUD_SHADOW_CELLS + x;
    return unpack2x16float(cloud_shadows[index / 2u])[index % 2u];
}
