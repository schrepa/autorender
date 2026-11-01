import package::engine::view::camera;
import package::face::sun::{sun_color, sun_direction};
import package::generated::sky::{CLOUD_HORIZON, CLOUD_REACH};
import package::sky::environment::{environment_maps, environment_sampler};
import package::sky::piece::sky_piece;

fn haze(direction: vec3<f32>) -> vec3<f32> {
    return haze_sky(direction) + sun_halo(direction);
}

fn haze_sky(direction: vec3<f32>) -> vec3<f32> {
    let up = direction.y;
    let color = mix(sky_piece.horizon.rgb, sky_piece.zenith.rgb, sqrt(max(up, 0.0)));
    return mix(color, sky_piece.ground.rgb, min(max(-up, 0.0) * 1.5, 0.35));
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

fn cloud_dome_uv(direction: vec3<f32>) -> vec2<f32> {
    let around = atan2(direction.z, direction.x) * (1.0 / 6.2831853);
    let low = -asin(CLOUD_HORIZON);
    let elevation = asin(clamp(direction.y, -1.0, 1.0));
    return vec2<f32>(fract(around), saturate((elevation - low) / (1.5707963 - low)));
}

fn cloud_dome_direction(origin: vec3<f32>, start: vec3<f32>, direction: vec3<f32>)
        -> vec3<f32> {
    let rise = sky_piece.domes_eye.w - start.y;
    if direction.y * rise <= 0.0 || abs(direction.y) < 1.0e-5 {
        return direction;
    }
    let along = min(rise / direction.y, CLOUD_REACH);
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
    let depth = height_depth(eye, point, sky_piece.fog.x, sky_piece.fog.y, sky_piece.fog.z);
    return min(1.0 - exp(-depth), sky_piece.fog.w);
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
    light.air = sky_piece.aerial_color.rgb;
    light.air_sun = air_glow(direction);
    return light;
}

fn air_glow(direction: vec3<f32>) -> vec3<f32> {
    let toward = max(dot(direction, sun_direction()), 0.0);
    let broad = toward * toward;
    let lobe = broad * broad;
    return sun_color() * (sky_piece.aerial.w * lobe * (0.25 + 0.75 * lobe));
}

fn sky_glow(direction: vec3<f32>) -> vec3<f32> {
    if sky_piece.state.x > 0.5 {
        return air_glow(direction);
    }
    return sun_halo(direction);
}

fn marched_reach() -> f32 {
    return select(sky_piece.haze_march.x, 0.0, camera.options.x != 0.0);
}

const HAZE_BLUR: f32 = 0.3;
const HAZE_LIFT: f32 = 0.02;

fn hazy() -> bool {
    return sky_piece.aerial_color.w != 0.0;
}

fn sky_behind(direction: vec3<f32>) -> vec3<f32> {
    let lifted = normalize(vec3<f32>(direction.x, max(direction.y, HAZE_LIFT), direction.z));
    let level = HAZE_BLUR * f32(textureNumLevels(environment_maps) - 1u);
    return textureSampleLevel(environment_maps, environment_sampler, lifted, 0, level).rgb;
}
