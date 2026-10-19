// Particles, prepended with scene.wgsl. Every particle is a
// function of its emitter, its index and presentation time: particle k of an
// emitter is born at k / rate seconds, and the emitter's instances draw the
// latest births, one lane each. Billboards face the camera, or streak along
// their motion. Alpha particles are lit by the sun and sky and accumulate as
// weighted blended transparency, as mist does, each particle of it a ball
// soft through its depth; additive ones add their light.

struct Particles {
    // Camera right (xyz) and up (xyz), unit and world space.
    right: vec4<f32>,
    up: vec4<f32>,
    // Viewport size in pixels, and its inverse.
    size: vec4<f32>,
}

struct Emitter {
    origin: vec3<f32>,
    rate: f32,
    velocity: vec3<f32>,
    lifetime: f32,
    direction: vec3<f32>,
    speed: f32,
    acceleration: vec3<f32>,
    // Cosine of the spread half-angle.
    spread: f32,
    start_color: vec4<f32>,
    end_color: vec4<f32>,
    size: vec2<f32>,
    emission: f32,
    stretch: f32,
    radius: f32,
    seed: u32,
    // First instance and instances of this emitter.
    first: u32,
    count: u32,
}

@group(1) @binding(0) var<uniform> particles: Particles;
@group(1) @binding(1) var<storage, read> emitters: array<Emitter>;
// The emitter of each instance.
@group(1) @binding(2) var<storage, read> owners: array<u32>;
@group(1) @binding(3) var depth_texture: texture_depth_2d;

const TAU: f32 = 6.2831853;

struct ParticleOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Linear color times emission, and alpha.
    @location(0) color: vec4<f32>,
    // Position within the billboard, -1 to 1 on each axis.
    @location(1) corner: vec2<f32>,
    @location(2) world: vec3<f32>,
    // Distance over which the particle fades into opaque surfaces.
    @location(3) @interpolate(flat) softness: f32,
    // Its middle, and the meters over which mist thins from it (w); and
    // where its noise lies, its own.
    @location(4) @interpolate(flat) center: vec4<f32>,
    @location(5) @interpolate(flat) key: f32,
}

// PCG hash (Jarzynski and Olano).
fn pcg(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Two unit vectors perpendicular to unit `axis` and to each other.
fn basis(axis: vec3<f32>) -> mat2x3<f32> {
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(axis.x) > 0.9);
    let u = normalize(cross(axis, helper));
    return mat2x3<f32>(u, cross(axis, u));
}

fn hidden() -> ParticleOutput {
    var output: ParticleOutput;
    // Outside the clip volume: the triangle is clipped away.
    output.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    return output;
}

@vertex
fn vs_particle(@builtin(vertex_index) vertex: u32,
        @builtin(instance_index) instance: u32) -> ParticleOutput {
    let emitter = emitters[owners[instance]];
    let lane = instance - emitter.first;
    let time = camera.position.w;
    // The latest birth in this lane: k with k = lane (mod count) and
    // k / rate at or before now.
    let births = floor(time * emitter.rate);
    if births < f32(lane) {
        return hidden();
    }
    let latest = u32(births);
    let k = latest - (latest - lane) % emitter.count;
    let age = time - f32(k) / emitter.rate;
    if age < 0.0 || age > emitter.lifetime {
        return hidden();
    }
    let life = age / emitter.lifetime;
    let seed = pcg(emitter.seed ^ pcg(k));
    let r0 = unit(seed);
    let r1 = unit(pcg(seed + 1u));
    let r2 = unit(pcg(seed + 2u));
    let r3 = unit(pcg(seed + 3u));
    let r4 = unit(pcg(seed + 4u));
    // A direction within the cone and a point within the ball.
    let cos_theta = mix(1.0, emitter.spread, r0);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = r1 * TAU;
    let across = basis(emitter.direction);
    let direction = emitter.direction * cos_theta
        + (across[0] * cos(phi) + across[1] * sin(phi)) * sin_theta;
    let ball = normalize(vec3<f32>(r2, r3, r4) * 2.0 - 1.0 + vec3<f32>(1.0e-4))
        * (emitter.radius * pow(r4, 1.0 / 3.0));
    let launch = direction * (emitter.speed * (0.75 + 0.5 * r2));
    // The emitter moved on since the birth: the particle stays behind.
    let position = emitter.origin + ball - emitter.velocity * age + launch * age
        + emitter.acceleration * (0.5 * age * age);
    let velocity = launch + emitter.acceleration * age;
    let size = mix(emitter.size.x, emitter.size.y, life) * (0.8 + 0.4 * r3);
    // One strip: the triangles (0, 1, 2) and (2, 1, 3).
    let corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0),
    );
    let corner = corners[vertex % 4u];
    var along_x = particles.right.xyz * (size * 0.5);
    var along_y = particles.up.xyz * (size * 0.5);
    if emitter.stretch > 0.0 {
        // Streaks lie along the motion as the view sees it.
        let view = normalize(position - camera.position.xyz);
        let seen = velocity - view * dot(velocity, view);
        let length_seen = length(seen);
        if length_seen > 1.0e-4 {
            let axis = seen / length_seen;
            along_y = axis * (size * 0.5 + length_seen * emitter.stretch * 0.5);
            along_x = normalize(cross(view, axis)) * (size * 0.5);
        }
    }
    let world = position + along_x * corner.x + along_y * corner.y;
    var output: ParticleOutput;
    output.clip_position = camera.view_projection * vec4<f32>(world, 1.0);
    let color = mix(emitter.start_color, emitter.end_color, life);
    output.color = vec4<f32>(color.rgb * emitter.emission, color.a);
    output.corner = corner;
    output.world = world;
    output.softness = max(size, 0.05);
    output.center = vec4<f32>(position, size * MIST_SPREAD);
    output.key = unit(pcg(seed + 5u)) * 256.0;
    return output;
}

// Mist: the share of its diameter over which a ball of it thins from its
// middle (a Gaussian's width: its billboard's edge lies 2.5 of them out,
// where it has thinned to nothing), how strongly its droplets scatter the
// sun forward (Henyey and Greenstein's g) and how much of the sun's light
// they scatter toward the eye beside the sky's.
const MIST_SPREAD: f32 = 0.2;
const MIST_PHASE_G: f32 = 0.6;
const MIST_SUN: f32 = 0.35;

// The error function (Abramowitz and Stegun 7.1.26, within 1.5e-7).
fn error_function(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let tail = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592) * t * exp(-a * a);
    return sign(x) * (1.0 - tail);
}

// Value noise at `p` in three dimensions: hashed lattice values blended by
// smooth steps.
fn mist_noise(p: vec3<f32>) -> f32 {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * (3.0 - 2.0 * f);
    let q = vec3<u32>(vec3<i32>(cell));
    let x0 = q.x * 0x8da6b343u;
    let x1 = x0 + 0x8da6b343u;
    let y0 = q.y * 0xd8163841u;
    let y1 = y0 + 0xd8163841u;
    let z0 = q.z * 0xcb1ab31fu;
    let z1 = z0 + 0xcb1ab31fu;
    let low = mix(mix(unit(pcg(x0 ^ y0 ^ z0)), unit(pcg(x1 ^ y0 ^ z0)), u.x),
        mix(unit(pcg(x0 ^ y1 ^ z0)), unit(pcg(x1 ^ y1 ^ z0)), u.x), u.y);
    let high = mix(mix(unit(pcg(x0 ^ y0 ^ z1)), unit(pcg(x1 ^ y0 ^ z1)), u.x),
        mix(unit(pcg(x0 ^ y1 ^ z1)), unit(pcg(x1 ^ y1 ^ z1)), u.x), u.y);
    return mix(low, high, u.z);
}

// Whether this fragment lies outside the round billboard, which covers it
// with nothing: fragment shaders discard it before reading depth.
fn outside(input: ParticleOutput) -> bool {
    return dot(input.corner, input.corner) >= 1.0;
}

// Coverage of the round billboard at this fragment, fading where it meets
// opaque surfaces (soft particles).
fn coverage(input: ParticleOutput) -> f32 {
    let d2 = dot(input.corner, input.corner);
    let disc = (1.0 - d2) * (1.0 - d2) * select(0.0, 1.0, d2 < 1.0);
    let pixel = vec2<i32>(input.clip_position.xy);
    let depth = textureLoad(depth_texture, pixel, 0);
    let uv = input.clip_position.xy * particles.size.zw;
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let behind = camera.inverse_view_projection * clip;
    let eye = camera.position.xyz;
    let gap = length(behind.xyz / behind.w - eye) - length(input.world - eye);
    return disc * saturate(gap / input.softness);
}

@fragment
fn fs_particle(input: ParticleOutput) -> TransparentTargets {
    if outside(input) {
        discard;
    }
    let alpha = input.color.a * coverage(input);
    if alpha <= 1.0e-3 {
        discard;
    }
    // Smoke and spray scatter the sky's light and half the sun's.
    let eye = camera.position.xyz;
    let light = input.color.rgb * (sky_average(vec3<f32>(0.0, 1.0, 0.0)) * lighting.zenith.w
        + lighting.sun_color.rgb * 0.5);
    let view = normalize(input.world - eye);
    let fogged = atmospheric(light, eye, input.world, view).rgb;
    return weighted(grade(fogged), alpha, length(input.world - eye));
}

// Mist: a ball of droplets thinning from its middle as a Gaussian, which
// stops as much light as the view crosses of it before the opaque surface
// behind (so it meets surfaces softly however near they stand, and its
// edge shows round from any side), alpha being what a view through its
// middle stops; torn by noise drifting through it. It scatters the sky's
// light, and the sun's forward, the more the nearer the eye looks toward
// the sun, as much as reaches its middle.
@fragment
fn fs_particle_mist(input: ParticleOutput) -> TransparentTargets {
    if outside(input) {
        discard;
    }
    let eye = camera.position.xyz;
    let direction = normalize(input.world - eye);
    let middle = input.center.xyz;
    let spread = max(input.center.w, 1.0e-3);
    let to_middle = middle - eye;
    let along = dot(to_middle, direction);
    let missed = max(dot(to_middle, to_middle) - along * along, 0.0);
    // How far the view runs before the opaque surface behind stops it.
    let pixel = vec2<i32>(input.clip_position.xy);
    let depth = textureLoad(depth_texture, pixel, 0);
    let uv = input.clip_position.xy * particles.size.zw;
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let behind = camera.inverse_view_projection * clip;
    let reach = length(behind.xyz / behind.w - eye);
    // Of the ball's density along a line through its middle, the share the
    // view crosses: as it misses the middle, and up to the surface behind.
    let crossed = 0.5 * (error_function((reach - along) / spread)
        - error_function(-along / spread)) * exp(-missed / (spread * spread));
    let closest = eye + direction * clamp(along, 0.0, reach);
    let drift = vec3<f32>(0.0, -0.25, 0.0) * camera.position.w;
    let torn = mist_noise((closest - middle + drift) / spread * 0.7
        + vec3<f32>(input.key, 17.0, 41.0));
    let thickest = -log(max(1.0 - input.color.a, 1.0e-3));
    let alpha = 1.0 - exp(-thickest * crossed * mix(0.3, 1.7, torn));
    if alpha <= 1.0e-3 {
        discard;
    }
    let sun = lighting.sun.xyz;
    var sunlight = vec3<f32>(0.0);
    if sun.y > 0.0 {
        let g = MIST_PHASE_G;
        let denominator = 1.0 + g * g - 2.0 * g * dot(sun, direction);
        let phase = (1.0 - g * g) / (denominator * sqrt(denominator));
        sunlight = lighting.sun_color.rgb * (MIST_SUN * phase)
            * sun_visibility(middle, sun, vec3<f32>(0.0), vec3<f32>(0.0), length(to_middle),
                false);
    }
    // The light around it: the sky's, and the sunlit and skylit ground's
    // it stands over, as water's foam takes it.
    let sky = sky_average(vec3<f32>(0.0, 1.0, 0.0)) * lighting.zenith.w;
    let around = lighting.sun_color.rgb * (0.35 * saturate(sun.y)) + sky * 0.5;
    let light = input.color.rgb * (mix(around, sky, 0.5) + sunlight);
    let fogged = atmospheric(light, eye, closest, direction).rgb;
    return weighted(grade(fogged), alpha, length(closest - eye));
}

// Additive particles: their light through the fog, graded.
fn added(input: ParticleOutput) -> vec3<f32> {
    let eye = camera.position.xyz;
    let light = input.color.rgb * (input.color.a * coverage(input));
    return light * (1.0 - atmospheric_cover(eye, input.world));
}

@fragment
fn fs_particle_add(input: ParticleOutput) -> @location(0) vec4<f32> {
    if outside(input) {
        discard;
    }
    return vec4<f32>(encoded(grade(added(input))), 0.0);
}

struct AddedTargets {
    @location(0) color: vec4<f32>,
    @location(1) glow: vec2<f32>,
}

// Screen-effect frames: the light also glows.
@fragment
fn fs_particle_add_post(input: ParticleOutput) -> AddedTargets {
    if outside(input) {
        discard;
    }
    let light = added(input);
    var targets: AddedTargets;
    targets.color = vec4<f32>(grade(light), 0.0);
    targets.glow = vec2<f32>(glow_weight(light), 0.0);
    return targets;
}
