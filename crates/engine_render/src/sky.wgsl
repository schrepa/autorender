// The physically based sky's tables and light (`sky.rs`),
// after `sky_common.wgsl`. Every pass binds one group: the sky's uniform,
// the tables to read (a blank texel in place of the one a pass writes), the
// table it writes and the light it derives.

@group(0) @binding(0) var<uniform> air: SkyUniform;
@group(0) @binding(1) var transmittance_table: texture_2d<f32>;
@group(0) @binding(2) var multiple_table: texture_2d<f32>;
@group(0) @binding(3) var view_table: texture_2d<f32>;
@group(0) @binding(4) var table_sampler: sampler;
@group(0) @binding(5) var table_out: texture_storage_2d<rgba16float, write>;
// The scene's light the sky gives (`sky.rs` `DERIVED`): the sun's color,
// the zenith's, the horizon's, the ground's, the air's away from the sun,
// and the sun's glow in it (x).
@group(0) @binding(6) var<storage, read_write> derived: array<vec4<f32>, 7>;

const TRANSMITTANCE_STEPS: u32 = 40u;
const MULTIPLE_STEPS: u32 = 20u;
const MULTIPLE_DIRECTIONS: u32 = 64u;
const VIEW_STEPS: u32 = 30u;
// Starlight and the glow of the night sky (at unit intensity), which keep
// nights from black.
const NIGHT: vec3<f32> = vec3<f32>(0.004, 0.006, 0.012);
// Kilometers above the ground scattering starts, avoiding the sphere's
// own surface.
const RADIUS_OFFSET: f32 = 0.01;

struct Medium {
    scattering: vec3<f32>,
    extinction: vec3<f32>,
    rayleigh: vec3<f32>,
    mie: vec3<f32>,
}

/// The air at `height` km above the ground.
fn medium_at(height: f32) -> Medium {
    let above = max(height, 0.0);
    let rayleigh_density = exp(-above / air.rayleigh.w);
    let mie_density = exp(-above / air.mie.z);
    // Ozone: a tent 30 km wide, densest 25 km up.
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

/// Cornette-Shanks.
fn mie_phase(cosine: f32) -> f32 {
    let g = air.mie.w;
    let k = 3.0 / (8.0 * SKY_PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cosine * cosine) / pow(max(1.0 + g * g - 2.0 * g * cosine, 1e-4), 1.5);
}

/// The share of light through the air from radius `r` along zenith cosine
/// `mu`, from the transmittance table.
fn sun_transmittance(r: f32, mu: f32) -> vec3<f32> {
    return air_transmittance(transmittance_table, table_sampler, air, r, mu);
}

/// Multiple scattering's isotropic light at radius `r` with the sun at
/// zenith cosine `mu`.
fn multiple_light(r: f32, mu: f32) -> vec3<f32> {
    let height = (r - air.radii.x) / max(air.radii.y - air.radii.x, 1e-6);
    let uv = vec2<f32>(
        to_sub_uv(clamp(mu * 0.5 + 0.5, 0.0, 1.0), SKY_MULTIPLE_SIZE),
        to_sub_uv(clamp(height, 0.0, 1.0), SKY_MULTIPLE_SIZE),
    );
    return textureSampleLevel(multiple_table, table_sampler, uv, 0.0).rgb;
}

struct Scattered {
    light: vec3<f32>,
    // The share of light scattered once more, for multiple scattering's
    // series (Hillaire's f_ms).
    again: vec3<f32>,
}

/// The light the air scatters toward `origin` along `direction` (unit)
/// from the sun along `sun` with unit illuminance, `steps` samples to the
/// top of the atmosphere or the ground; `isotropic` for multiple
/// scattering's table (one more bounce, without that table and with the
/// ground's reflection), otherwise single scattering plus the table's.
fn scattered(origin: vec3<f32>, direction: vec3<f32>, sun: vec3<f32>, steps: u32, isotropic: bool)
        -> Scattered {
    var result: Scattered;
    result.light = vec3<f32>(0.0);
    result.again = vec3<f32>(0.0);
    let bottom = air.radii.x;
    let top_hit = sphere_hit(origin, direction, air.radii.y);
    if top_hit < 0.0 {
        return result;
    }
    let ground_hit = sphere_hit(origin, direction, bottom);
    var span = top_hit;
    if ground_hit >= 0.0 {
        span = ground_hit;
    }
    let cosine = dot(direction, sun);
    let rayleigh = rayleigh_phase(cosine);
    let mie = mie_phase(cosine);
    let uniform_phase = 1.0 / (4.0 * SKY_PI);
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
        // The planet's shadow.
        let shadowed = sphere_hit(point, sun, bottom) >= 0.0;
        var phased: vec3<f32>;
        if isotropic {
            phased = medium.scattering * uniform_phase;
        } else {
            phased = medium.rayleigh * rayleigh + medium.mie * mie;
        }
        var source = select(toward_sun, vec3<f32>(0.0), shadowed) * phased;
        if !isotropic {
            source += multiple_light(r, sun_cos) * medium.scattering;
        }
        let extinction = max(medium.extinction, vec3<f32>(1e-6));
        // The light scattered over the step, integrated exactly for a
        // constant medium across it.
        let integrated = (source - source * sample_transmittance) / extinction;
        result.light += throughput * integrated;
        let again = (medium.scattering - medium.scattering * sample_transmittance) / extinction;
        result.again += throughput * again;
        throughput *= sample_transmittance;
    }
    // The ground's reflection of the sun: for multiple scattering, and
    // where views meet the planet beyond the world. The point lies on the
    // ground exactly: its radius in 32-bit floats rounds decimeters above
    // or below it, which turned the planet's own shadow on and off from
    // texel to texel. The sun reaches the ground wherever it stands above
    // its horizon, through the air the table's bottom row holds.
    if ground_hit >= 0.0 {
        let up = normalize(origin + direction * span);
        let sun_cos = dot(sun, up);
        if sun_cos > 0.0 {
            let through = textureSampleLevel(transmittance_table, table_sampler,
                transmittance_uv(air, bottom, sun_cos), 0.0).rgb;
            result.light += throughput * through * sun_cos * air.ground.rgb / SKY_PI;
        }
    }
    return result;
}

@compute @workgroup_size(8, 8)
fn build_transmittance(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(table_out);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let parameters = transmittance_parameters(air, uv);
    let r = parameters.x;
    let mu = parameters.y;
    let origin = vec3<f32>(0.0, r, 0.0);
    let direction = vec3<f32>(sqrt(max(1.0 - mu * mu, 0.0)), mu, 0.0);
    let reach = sphere_hit(origin, direction, air.radii.y);
    var depth = vec3<f32>(0.0);
    if reach > 0.0 {
        let dt = reach / f32(TRANSMITTANCE_STEPS);
        for (var step = 0u; step < TRANSMITTANCE_STEPS; step++) {
            let point = origin + direction * ((f32(step) + 0.5) * dt);
            depth += medium_at(length(point) - air.radii.x).extinction * dt;
        }
    }
    textureStore(table_out, id.xy, vec4<f32>(exp(-depth), 1.0));
}

/// A direction of `count` spread evenly over the sphere (Fibonacci).
fn sphere_direction(index: u32, count: u32) -> vec3<f32> {
    let golden = 2.39996323;
    let y = 1.0 - (f32(index) + 0.5) / f32(count) * 2.0;
    let radius = sqrt(max(1.0 - y * y, 0.0));
    let angle = golden * f32(index);
    return vec3<f32>(cos(angle) * radius, y, sin(angle) * radius);
}

@compute @workgroup_size(8, 8)
fn build_multiple(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(table_out);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let uv = vec2<f32>(
        from_sub_uv((f32(id.x) + 0.5) / f32(size.x), f32(size.x)),
        from_sub_uv((f32(id.y) + 0.5) / f32(size.y), f32(size.y)),
    );
    let sun_cos = clamp(uv.x * 2.0 - 1.0, -1.0, 1.0);
    let sun = vec3<f32>(0.0, sun_cos, sqrt(max(1.0 - sun_cos * sun_cos, 0.0)));
    let r = air.radii.x + clamp(uv.y + RADIUS_OFFSET, 0.0, 1.0)
        * (air.radii.y - air.radii.x - RADIUS_OFFSET);
    let origin = vec3<f32>(0.0, r, 0.0);
    var light = vec3<f32>(0.0);
    var again = vec3<f32>(0.0);
    for (var index = 0u; index < MULTIPLE_DIRECTIONS; index++) {
        let result = scattered(origin, sphere_direction(index, MULTIPLE_DIRECTIONS), sun,
            MULTIPLE_STEPS, true);
        light += result.light;
        again += result.again;
    }
    // Averaged over the sphere with an isotropic phase: the second order's
    // light and the share scattered again, summed as a geometric series.
    light /= f32(MULTIPLE_DIRECTIONS);
    again /= f32(MULTIPLE_DIRECTIONS);
    let series = 1.0 / max(vec3<f32>(1.0) - again, vec3<f32>(1e-3));
    textureStore(table_out, id.xy, vec4<f32>(light * series, 1.0));
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
    // The sun in the view's plane of azimuth zero, at its zenith angle.
    let sun_cos = clamp(normalize(air.sun.xyz).y, -1.0, 1.0);
    let sun = vec3<f32>(sqrt(max(1.0 - sun_cos * sun_cos, 0.0)), sun_cos, 0.0);
    let view_sin = sqrt(max(1.0 - view_cos * view_cos, 0.0));
    let direction = vec3<f32>(
        view_sin * light_cos,
        view_cos,
        view_sin * sqrt(max(1.0 - light_cos * light_cos, 0.0)),
    );
    let result = scattered(vec3<f32>(0.0, r, 0.0), direction, sun, VIEW_STEPS, false);
    textureStore(table_out, id.xy, vec4<f32>((result.light + NIGHT) * air.ozone.w, 1.0));
}

/// The sky's light looking along `direction` from the camera, without the
/// sun's disc.
fn view_light(direction: vec3<f32>) -> vec3<f32> {
    return air_sky(view_table, transmittance_table, table_sampler, air, direction, false);
}

/// A direction `elevation` radians above the horizon, `azimuth` around it.
fn around(azimuth: f32, elevation: f32) -> vec3<f32> {
    return vec3<f32>(cos(azimuth) * cos(elevation), sin(elevation), sin(azimuth) * cos(elevation));
}

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
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
    let night = NIGHT * air.ozone.w;
    let ground = air.ground.rgb
        * (sun_color * max(sun.y, 0.0) / SKY_PI + 0.5 * (zenith + horizon)) + night * 0.3;
    let glow = max(luminance(toward - away), 0.0) / max(luminance(sun_color), 1e-4);
    // Exposure adapts toward the light falling on level ground at noon,
    // part of the way, by at most 16 times.
    let falling = luminance(sun_color) * max(sun.y, 0.0)
        + SKY_PI * luminance(0.5 * (zenith + horizon));
    let noon = air.ozone.w * 1.1;
    let adapted = clamp(pow(noon / max(falling, 1e-5), air.radii.z * 0.75), 1.0, 16.0);
    // Exposure eases toward its target over about a second,
    // from where it stood last frame, as eyes adapt.
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
