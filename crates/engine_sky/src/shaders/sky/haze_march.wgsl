import package::engine::math::{pcg, unit};
import package::engine::view::{PI, camera, eye_world, pixel_noise};
import package::face::horizons::open_sky;
import package::face::shadows::air_shadow;
import package::face::sun::{sun_color, sun_direction};
import package::sky::air::air_light;
import package::sky::faces::sky::cloud_passing;
import package::sky::haze_view::{depth_texture, inverse_distance_at, post};
import package::sky::piece::sky_piece;

const HAZE_STEPS: u32 = 24u;
const HAZE_OCTAVES: u32 = 3u;
const HAZE_PRIMES: vec3<u32> = vec3<u32>(73856093u, 19349663u, 83492791u);
const HAZE_SALT: u32 = 0x9e3779b9u;
const HAZE_OPEN_SKY: f32 = 0.5;

fn haze_value(p: vec3<f32>, salt: u32) -> f32 {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * (3.0 - 2.0 * f);
    let low = vec3<u32>(vec3<i32>(cell)) * HAZE_PRIMES;
    let high = low + HAZE_PRIMES;
    let z0 = low.z ^ salt;
    let z1 = high.z ^ salt;
    let y0z0 = low.y ^ z0;
    let y1z0 = high.y ^ z0;
    let y0z1 = low.y ^ z1;
    let y1z1 = high.y ^ z1;
    let nearer = mix(
        mix(unit(pcg(low.x ^ y0z0)), unit(pcg(high.x ^ y0z0)), u.x),
        mix(unit(pcg(low.x ^ y1z0)), unit(pcg(high.x ^ y1z0)), u.x), u.y);
    let farther = mix(
        mix(unit(pcg(low.x ^ y0z1)), unit(pcg(high.x ^ y0z1)), u.x),
        mix(unit(pcg(low.x ^ y1z1)), unit(pcg(high.x ^ y1z1)), u.x), u.y);
    return mix(nearer, farther, u.z);
}

fn haze_density(world: vec3<f32>) -> f32 {
    let spread = sky_piece.haze_march.z;
    if spread <= 0.0 {
        return 1.0;
    }
    let drift = vec3<f32>(sky_piece.wind.x, 0.0, sky_piece.wind.y) * camera.position.w;
    var p = (world - drift) * sky_piece.haze_march.w;
    var sum = 0.0;
    var strength = 1.0;
    var total = 0.0;
    for (var octave = 0u; octave < HAZE_OCTAVES; octave++) {
        sum += strength * haze_value(p, octave * HAZE_SALT);
        total += strength;
        p *= 2.0;
        strength *= 0.5;
    }
    return 1.0 + spread * (2.0 * sum / total - 1.0);
}

fn haze_covered(depth: f32) -> f32 {
    if depth < 1.0e-3 {
        return depth * (1.0 - depth * (0.5 - depth * (1.0 / 6.0)));
    }
    return 1.0 - exp(-depth);
}

fn haze_seen(world: vec3<f32>) -> f32 {
    let open = open_sky(world);
    if !open.held {
        return 1.0;
    }
    return mix(saturate(open.seen / HAZE_OPEN_SKY), 1.0, open.beyond);
}

@fragment
fn fs_haze(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let inverse = inverse_distance_at(depth_texture, pixel, post);
    let depth = textureLoad(depth_texture, pixel, 0);
    let uv = (vec2<f32>(pixel) + 0.5) * post.size.zw;
    let far_point = post.inverse_view_projection
        * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let ray = far_point.xyz / far_point.w - post.eye.xyz;
    let ray_length = length(ray);
    let reach = min(ray_length, sky_piece.haze_march.x);
    if !(reach > 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, inverse);
    }
    let direction = ray / ray_length;
    let jitter = fract(pixel_noise(position.xy) + sky_piece.haze.x);
    var through = 1.0;
    var sunlit = 0.0;
    var air_sky = 0.0;
    var mean_through = 1.0;
    var mean_air_sky = 0.0;
    for (var index = 0u; index < HAZE_STEPS; index++) {
        let first = f32(index) / f32(HAZE_STEPS);
        let next = f32(index + 1u) / f32(HAZE_STEPS);
        let near = reach * first * first;
        let far = reach * next * next;
        let point = eye_world(direction * mix(near, far, jitter));
        let height = max(point.y, -1000.0);
        let air_mean = sky_piece.aerial.y * exp(-sky_piece.aerial.z * height) * (far - near);
        if !(air_mean > 0.0) {
            continue;
        }
        let air_step = air_mean * haze_density(point);
        if air_step > 0.0 {
            let covered = haze_covered(air_step);
            let scattered = through * covered;
            var visible = cloud_passing(point);
            if visible > 0.0 {
                visible *= air_shadow(point);
            }
            sunlit += scattered * visible;
            air_sky += scattered * haze_seen(point);
            through *= 1.0 - covered;
        }
        let mean_covered = haze_covered(air_mean);
        mean_air_sky += mean_through * mean_covered;
        mean_through *= 1.0 - mean_covered;
    }
    let forward = sky_piece.haze_march.y;
    let cosine = dot(direction, sun_direction());
    let denominator = 1.0 + forward * forward - 2.0 * forward * cosine;
    let phase = (1.0 - forward * forward) / (4.0 * PI * denominator * sqrt(denominator));
    let sun = sun_color() * (PI * phase * sunlit);
    let sky_light = air_light(direction).air * (air_sky - mean_air_sky);
    return vec4<f32>(sun + sky_light, inverse);
}
