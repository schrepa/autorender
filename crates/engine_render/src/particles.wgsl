// Particles, prepended with scene.wgsl. Every particle is a
// function of its emitter, its index and presentation time: particle k of an
// emitter is born at k / rate seconds, and the emitter's instances draw the
// latest births, one lane each. Billboards face the camera, or streak along
// their motion. Alpha particles are lit by the sun and sky and accumulate as
// weighted blended transparency; additive ones add their light.

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
    return output;
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
