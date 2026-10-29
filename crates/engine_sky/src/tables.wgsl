import package::engine::color::luminance;
import package::sky::common::{
    SKY_MULTIPLE_SIZE, SKY_PI, SkyUniform, air_radius, air_sky, air_transmittance,
    sky_view_parameters, sphere_hit, to_sub_uv, transmittance_uv
};

@group(0) @binding(0) var<uniform> air: SkyUniform;
@group(0) @binding(1) var transmittance_table: texture_2d<f32>;
@group(0) @binding(2) var multiple_table: texture_2d<f32>;
@group(0) @binding(3) var view_table: texture_2d<f32>;
@group(0) @binding(4) var table_sampler: sampler;
@group(0) @binding(5) var table_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var<storage, read_write> derived: array<vec4<f32>, 7>;

const VIEW_STEPS: u32 = 30u;
const NIGHT: vec3<f32> = vec3<f32>(0.004, 0.006, 0.012);
const MOON_NIGHT: vec3<f32> = vec3<f32>(0.006, 0.008, 0.014);
const NOON_LIGHT: f32 = 1.1;
const ADAPTATION_POWER: f32 = 0.75;
const ADAPTATION_MOST: f32 = 16.0;

struct Medium {
    scattering: vec3<f32>,
    extinction: vec3<f32>,
    rayleigh: vec3<f32>,
    mie: vec3<f32>,
}

fn medium_at(height: f32) -> Medium {
    let above = max(height, 0.0);
    let rayleigh_density = exp(-above / air.rayleigh.w);
    let mie_density = exp(-above / air.mie.z);
    let ozone_density = max(0.0, 1.0 - abs(height - 25.0) / 15.0);
    var medium: Medium;
    medium.rayleigh = air.rayleigh.xyz * rayleigh_density;
    medium.mie = vec3<f32>(air.mie.x * mie_density);
    medium.scattering = medium.rayleigh + medium.mie;
    medium.extinction = medium.scattering
        + vec3<f32>(air.mie.y * mie_density)
        + air.ozone.xyz * ozone_density;
    return medium;
}

fn rayleigh_phase(cosine: f32) -> f32 {
    return 3.0 / (16.0 * SKY_PI) * (1.0 + cosine * cosine);
}

fn mie_phase(cosine: f32) -> f32 {
    let g = air.mie.w;
    let k = 3.0 / (8.0 * SKY_PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cosine * cosine) / pow(max(1.0 + g * g - 2.0 * g * cosine, 1e-4), 1.5);
}

fn sun_transmittance(r: f32, mu: f32) -> vec3<f32> {
    return air_transmittance(transmittance_table, table_sampler, air, r, mu);
}

fn multiple_light(r: f32, mu: f32) -> vec3<f32> {
    let height = (r - air.radii.x) / max(air.radii.y - air.radii.x, 1e-6);
    let uv = vec2<f32>(
        to_sub_uv(clamp(mu * 0.5 + 0.5, 0.0, 1.0), SKY_MULTIPLE_SIZE),
        to_sub_uv(clamp(height, 0.0, 1.0), SKY_MULTIPLE_SIZE),
    );
    return textureSampleLevel(multiple_table, table_sampler, uv, 0.0).rgb;
}

fn scattered(origin: vec3<f32>, direction: vec3<f32>, sun: vec3<f32>, steps: u32) -> vec3<f32> {
    var light = vec3<f32>(0.0);
    let bottom = air.radii.x;
    let top_hit = sphere_hit(origin, direction, air.radii.y);
    if top_hit < 0.0 {
        return light;
    }
    let ground_hit = sphere_hit(origin, direction, bottom);
    var span = top_hit;
    if ground_hit >= 0.0 {
        span = ground_hit;
    }
    let cosine = dot(direction, sun);
    let rayleigh = rayleigh_phase(cosine);
    let mie = mie_phase(cosine);
    var throughput = vec3<f32>(1.0);
    var travelled = 0.0;
    for (var step = 0u; step < steps; step++) {
        let t = span * (f32(step) + 0.3) / f32(steps);
        let dt = t - travelled;
        travelled = t;
        let point = origin + direction * t;
        let r = length(point);
        let up = point / r;
        let medium = medium_at(r - bottom);
        let sample_transmittance = exp(-medium.extinction * dt);
        let sun_cos = dot(sun, up);
        let toward_sun = sun_transmittance(r, sun_cos);
        let shadowed = sphere_hit(point, sun, bottom) >= 0.0;
        let phased = medium.rayleigh * rayleigh + medium.mie * mie;
        let source = select(toward_sun, vec3<f32>(0.0), shadowed) * phased
            + multiple_light(r, sun_cos) * medium.scattering;
        let extinction = max(medium.extinction, vec3<f32>(1e-6));
        let integrated = (source - source * sample_transmittance) / extinction;
        light += throughput * integrated;
        throughput *= sample_transmittance;
    }
    if ground_hit >= 0.0 {
        let up = normalize(origin + direction * span);
        let sun_cos = dot(sun, up);
        if sun_cos > 0.0 {
            let through = textureSampleLevel(transmittance_table, table_sampler,
                transmittance_uv(air, bottom, sun_cos), 0.0).rgb;
            light += throughput * through * sun_cos * air.ground.rgb / SKY_PI;
        }
    }
    return light;
}

@compute @workgroup_size(8, 8)
fn build_sky_view(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(table_out);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let r = air_radius(air);
    let parameters = sky_view_parameters(air, uv, r);
    let view_cos = parameters.x;
    let light_cos = parameters.y;
    let sun_cos = clamp(normalize(air.sun.xyz).y, -1.0, 1.0);
    let sun = vec3<f32>(sqrt(max(1.0 - sun_cos * sun_cos, 0.0)), sun_cos, 0.0);
    let view_sin = sqrt(max(1.0 - view_cos * view_cos, 0.0));
    let direction = vec3<f32>(
        view_sin * light_cos,
        view_cos,
        view_sin * sqrt(max(1.0 - light_cos * light_cos, 0.0)),
    );
    let light = scattered(vec3<f32>(0.0, r, 0.0), direction, sun, VIEW_STEPS);
    textureStore(table_out, id.xy, vec4<f32>((light + NIGHT) * air.ozone.w, 1.0));
}

fn view_light(direction: vec3<f32>) -> vec3<f32> {
    return air_sky(view_table, table_sampler, air, direction);
}

fn around(azimuth: f32, elevation: f32) -> vec3<f32> {
    return vec3<f32>(cos(azimuth) * cos(elevation), sin(elevation), sin(azimuth) * cos(elevation));
}

@compute @workgroup_size(1)
fn derive_light() {
    let r = air_radius(air);
    let sun = normalize(air.sun.xyz);
    let sun_color = air.ozone.w * sun_transmittance(r, sun.y);
    let zenith = view_light(vec3<f32>(0.0, 1.0, 0.0));
    var horizon = vec3<f32>(0.0);
    for (var index = 0u; index < 8u; index++) {
        horizon += view_light(around(f32(index) * SKY_PI * 0.25, 0.1));
    }
    horizon /= 8.0;
    let azimuth = atan2(sun.z, sun.x);
    let toward = view_light(around(azimuth, 0.035));
    let away = view_light(around(azimuth + SKY_PI, 0.035));
    let night = (NIGHT + MOON_NIGHT * (air.moon.w * max(air.moon.y, 0.0))) * air.ozone.w;
    let ground = air.ground.rgb
        * (sun_color * max(sun.y, 0.0) / SKY_PI + 0.5 * (zenith + horizon)) + night * 0.3;
    let glow = max(luminance(toward - away), 0.0) / max(luminance(sun_color), 1e-4);
    let falling = luminance(sun_color) * max(sun.y, 0.0)
        + SKY_PI * luminance(0.5 * (zenith + horizon));
    let noon = air.ozone.w * NOON_LIGHT;
    let adapted = clamp(pow(noon / max(falling, 1e-5), air.radii.z * ADAPTATION_POWER), 1.0,
        ADAPTATION_MOST);
    let wanted = air.sun.w * adapted;
    let previous = derived[6].x;
    var exposure = wanted;
    if air.frame.x > 0.0 && previous > 0.0 {
        exposure = mix(previous, wanted, 1.0 - exp(-air.frame.x / 0.8));
    }
    derived[0] = vec4<f32>(sun_color, 0.0);
    derived[1] = vec4<f32>(max(zenith, night), 0.0);
    derived[2] = vec4<f32>(max(horizon, night), exposure);
    derived[6] = vec4<f32>(exposure, 0.0, 0.0, 0.0);
    derived[3] = vec4<f32>(ground, 0.0);
    derived[4] = vec4<f32>(max(away, night), 0.0);
    derived[5] = vec4<f32>(min(glow, 4.0), 0.0, 0.0, 0.0);
}
