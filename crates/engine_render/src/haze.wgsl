@group(1) @binding(0) var<uniform> post: Post;
@group(1) @binding(1) var depth_texture: texture_depth_2d;

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
    let spread = lighting.haze_march.z;
    if spread <= 0.0 {
        return 1.0;
    }
    let drift = vec3<f32>(camera.wind.x, 0.0, camera.wind.y) * camera.position.w;
    var p = (world - drift) * lighting.haze_march.w;
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
    let at = bounce_at(world, false);
    if at.texel.x < 0.0 {
        return 1.0;
    }
    return mix(saturate(bounce_seen(at) / HAZE_OPEN_SKY), 1.0, saturate(at.layer - 1.0));
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
    let reach = min(ray_length, lighting.haze_march.x);
    if lighting.underwater.w > 0.0 || !(reach > 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, inverse);
    }
    let direction = ray / ray_length;
    let jitter = fract(pixel_noise(position.xy) + post.haze.x);
    let fog_most = -log(max(1.0 - lighting.fog.w, 1.0e-6));
    let shadowed = SHADOW_MAPS && shadow.state.x != 0.0;
    var through = 1.0;
    var fogged = 0.0;
    var sunlit = 0.0;
    var fog_sky = 0.0;
    var air_sky = 0.0;
    var mean_through = 1.0;
    var mean_fogged = 0.0;
    var mean_fog_sky = 0.0;
    var mean_air_sky = 0.0;
    for (var index = 0u; index < HAZE_STEPS; index++) {
        let first = f32(index) / f32(HAZE_STEPS);
        let next = f32(index + 1u) / f32(HAZE_STEPS);
        let near = reach * first * first;
        let far = reach * next * next;
        let point = eye_world(direction * mix(near, far, jitter));
        let height = max(point.y, -1000.0);
        let fog_mean = lighting.fog.y * exp(-lighting.fog.z * height)
            * max(far - max(near, lighting.fog.x), 0.0);
        let air_mean = lighting.aerial.y * exp(-lighting.aerial.z * height)
            * max(far - max(near, lighting.aerial.x), 0.0);
        if !(fog_mean + air_mean > 0.0) {
            continue;
        }
        let drifted = haze_density(point);
        let fog_step = min(fog_mean * drifted, max(fog_most - fogged, 0.0));
        let air_step = air_mean * drifted;
        fogged += fog_step;
        let step_depth = fog_step + air_step;
        if step_depth > 0.0 {
            let covered = haze_covered(step_depth);
            let scattered = through * covered / step_depth;
            var visible = cloud_passing(point);
            if shadowed && visible > 0.0 {
                visible *= air_visibility(point);
            }
            sunlit += scattered * step_depth * visible;
            let sky_seen = scattered * haze_seen(point);
            fog_sky += sky_seen * fog_step;
            air_sky += sky_seen * air_step;
            through *= 1.0 - covered;
        }
        let mean_fog_step = min(fog_mean, max(fog_most - mean_fogged, 0.0));
        mean_fogged += mean_fog_step;
        let mean_depth = mean_fog_step + air_mean;
        if mean_depth > 0.0 {
            let covered = haze_covered(mean_depth);
            let scattered = mean_through * covered / mean_depth;
            mean_fog_sky += scattered * mean_fog_step;
            mean_air_sky += scattered * air_mean;
            mean_through *= 1.0 - covered;
        }
    }
    let forward = lighting.haze_march.y;
    let cosine = dot(direction, lighting.sun.xyz);
    let denominator = 1.0 + forward * forward - 2.0 * forward * cosine;
    let phase = (1.0 - forward * forward) / (4.0 * PI * denominator * sqrt(denominator));
    let sun = lighting.sun_color.rgb * (PI * phase * sunlit);
    let lights = air_light(direction);
    let sky_light = lights.fog * (fog_sky - mean_fog_sky) + lights.air * (air_sky - mean_air_sky);
    return vec4<f32>((sun + sky_light) * lighting.horizon.w, inverse);
}
