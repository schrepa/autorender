// The haze's march, prepended with scene.wgsl and effects.wgsl: at half
// resolution, each view ray marches the air `atmospheric` integrates, the
// fog and the haze's height air, to the depth buffer's surface or the
// haze's reach, in HAZE_STEPS steps from a start jittered per pixel and per
// frame. Each step adds the sunlight its air scatters toward the eye, by
// Henyey and Greenstein's phase of the haze's forward scattering (Mie
// scattering's), where the shadow clipmap and the clouds' shadows let the
// sun through, dimmed by the air in front of it. The air's density is its
// mean's times a noise of mean 1, anchored to the world and drifting with
// the wind, and the air scatters the sky's light only from the share of the
// sky it sees, which terrains' bounce probes hold (`haze_seen`); the march
// adds how far that light departs from the sky's light `atmospheric` gives
// at the mean, which counts the whole sky.
// Within the reach the march is the only source of the sunlight the air
// scatters (`atmospheric` keeps its own beyond it), and it takes back the
// sky's light `atmospheric` gives the air there by the share of the sky the
// air does not see; beyond the reach the air counts as seeing the whole
// sky, and nothing seams there. The march runs with haze only, where fog
// and the air take the sky's one light (`air_light`), so what `atmospheric`
// gives of it within the reach is that light times what the air there
// covers at the mean, which the march's mean gathers again from the same
// media: the split is exact but for the march's sampling, each step's air
// taken at its jittered point, which the filter over time averages over
// the step. post.wgsl filters the result over time and the composite lays
// it over the scene.

@group(1) @binding(0) var<uniform> post: Post;
@group(1) @binding(1) var depth_texture: texture_depth_2d;

// Steps of each view's march, their lengths growing with the square of the
// distance from the eye: finest near it, where shadows span the most
// pixels.
const HAZE_STEPS: u32 = 24u;
// Octaves of the haze's noise, each half the size and half the strength of
// the one before it.
const HAZE_OCTAVES: u32 = 3u;
// What a lattice point's coordinates are multiplied by before they are
// mixed (Teschner et al., "Optimized Spatial Hashing for Collision
// Detection of Deformable Objects", 2003).
const HAZE_PRIMES: vec3<u32> = vec3<u32>(73856093u, 19349663u, 83492791u);
// What salts each octave's lattice apart from the one before: 2^32 over the
// golden ratio, whose multiples spread evenly.
const HAZE_SALT: u32 = 0x9e3779b9u;
// The share of all ways from a point over level ground that reach the sky:
// the half above the horizon, the open sky `atmospheric`'s light counts as
// the whole sky.
const HAZE_OPEN_SKY: f32 = 0.5;

// Smooth value noise in [0, 1] at `p` (lattice units), the lattice salted by
// `salt`: the hashed values at the corners of the lattice cell holding `p`,
// blended across it by smoothstep. Its mean is a half.
fn haze_value(p: vec3<f32>, salt: u32) -> f32 {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * (3.0 - 2.0 * f);
    // Each corner's coordinates times the primes, mixed by PCG.
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

// The haze's density at world point `world` as a multiple of its mean: 1
// plus its spread (`lighting.haze_march.z`) times HAZE_OCTAVES octaves of
// value noise centered on their mean, the first as large as the haze's
// largest patches (`lighting.haze_march.w` of them a meter), so its mean is
// 1. The noise is anchored to the world, and the wind (`camera.wind`)
// carries it along unchanged, as it carries the eddies it is made of
// (Taylor's frozen turbulence).
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

// The share of light a step of optical depth `depth` takes away and
// scatters, 1 - e^-depth, by its series where the exponential would round
// it away.
fn haze_covered(depth: f32) -> f32 {
    if depth < 1.0e-3 {
        return depth * (1.0 - depth * (0.5 - depth * (1.0 / 6.0)));
    }
    return 1.0 - exp(-depth);
}

// The share of the sky the air at world point `world` sees, of the sky
// `atmospheric`'s light counts, the open sky over level ground. The air
// scatters the sky's light alike from every way it comes, not by their
// cosine as a surface takes it, so the share is that of all ways from
// terrains' bounce probes there that reach the sky (`bounce_seen`) over the
// open sky's (HAZE_OPEN_SKY), at most all of it. Above the probes' upper
// layer it turns toward the open sky, which stands as a third layer as far
// above the upper as the upper stands above the lower; where no terrain's
// probes hold the point (no terrain, or bounce light off), the air sees
// all of it. Ghost of Tsushima's haze dims its sky's light alike, by the
// sky its lighting's probes see (Patry, "Real-Time Samurai Cinema", 2021).
fn haze_seen(world: vec3<f32>) -> f32 {
    let at = bounce_at(world, false);
    if at.texel.x < 0.0 {
        return 1.0;
    }
    return mix(saturate(bounce_seen(at) / HAZE_OPEN_SKY), 1.0, saturate(at.layer - 1.0));
}

// The haze at half-resolution texel `position`: the light the near air
// scatters toward the eye along the ray through the texel's first pixel,
// exposed (rgb), and the inverse distance of the surface the ray ends at
// (w; 0 for sky), which the filter and the composite compare. Each step's
// light is its air's share of what it scatters (Hillaire, "Physically
// Based and Unified Volumetric Rendering in Frostbite", 2015): what the air
// in front leaves of it times what the step covers, its density and light
// those at its jittered point.
@fragment
fn fs_haze(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = min(vec2<i32>(position.xy) * 2, vec2<i32>(post.size.xy) - 1);
    let inverse = inverse_distance_at(depth_texture, pixel, post);
    let depth = textureLoad(depth_texture, pixel, 0);
    let uv = (vec2<f32>(pixel) + 0.5) * post.size.zw;
    let far_point = post.inverse_view_projection
        * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let ray = far_point.xyz / far_point.w - post.eye.xyz;
    let ray_length = length(ray);
    let reach = min(ray_length, lighting.haze_march.x);
    // Under the sea, its water stands in for the air.
    if lighting.underwater.w > 0.0 || !(reach > 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, inverse);
    }
    let direction = ray / ray_length;
    let jitter = fract(pixel_noise(position.xy) + post.haze.x);
    // The optical depth at which fog covers its most (`lighting.fog.w`),
    // past which it covers no more.
    let fog_most = -log(max(1.0 - lighting.fog.w, 1.0e-6));
    let shadowed = SHADOW_MAPS && shadow.state.x != 0.0;
    // Through the drifting air: what it leaves of light from the eye to
    // the step, its fog's optical depth so far, the sunlight it scatters
    // (before the sun's light and the phase) and the shares of the sky's
    // light its fog and its air scatter from the sky they see; then the same
    // at the air's mean, seeing the whole sky, as `atmospheric` lights it.
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
        // The step's optical depths at the mean, each medium from where it
        // begins along the view.
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
    // Henyey and Greenstein's phase of the haze's forward scattering, from
    // the sun toward the eye.
    let forward = lighting.haze_march.y;
    let cosine = dot(direction, lighting.sun.xyz);
    let denominator = 1.0 + forward * forward - 2.0 * forward * cosine;
    let phase = (1.0 - forward * forward) / (4.0 * PI * denominator * sqrt(denominator));
    // The sun's irradiance is pi times its color, which a white surface
    // facing it shows.
    let sun = lighting.sun_color.rgb * (PI * phase * sunlit);
    // The sky's light the air scatters from the sky it sees, less what
    // `atmospheric` gives at the mean: the noise's departure from its mean,
    // more where the air is thicker and less where it is clearer, as far as
    // the air sees the sky; less the light the share of the sky it does not
    // see would have scattered at the mean, through the air in front.
    let lights = air_light(direction);
    let sky_light = lights.fog * (fog_sky - mean_fog_sky) + lights.air * (air_sky - mean_air_sky);
    return vec4<f32>((sun + sky_light) * lighting.horizon.w, inverse);
}
