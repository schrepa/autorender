import package::engine::color::encode_srgb;
import package::engine::math::{pcg, unit};
import package::frame::encode::{glow_weight, grade};
import package::frame::view::{camera, lighting};
import package::lighting::sun::sun_visibility;
import package::shadows::sample::{SHADOW_MAPS, map_visibility, shadow};
import package::sky::air::{atmospheric, atmospheric_cover, cloud_passing, sky_average};
import package::terrain::horizons::terrain_visibility;
import package::transparent::weighted::{TransparentTargets, weighted_particle};

struct Particles {
    right: vec4<f32>,
    up: vec4<f32>,
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
    spread: f32,
    start_color: vec4<f32>,
    end_color: vec4<f32>,
    size: vec2<f32>,
    emission: f32,
    stretch: f32,
    radius: f32,
    seed: u32,
    first: u32,
    count: u32,
}

@group(1) @binding(0) var<uniform> particles: Particles;
@group(1) @binding(1) var<storage, read> emitters: array<Emitter>;
@group(1) @binding(2) var<storage, read> owners: array<u32>;
@group(1) @binding(3) var depth_texture: texture_depth_2d;

const TAU: f32 = 6.2831853;

struct ParticleOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) corner: vec2<f32>,
    @location(2) world: vec3<f32>,
    @location(3) @interpolate(flat) softness: f32,
    @location(4) @interpolate(flat) center: vec4<f32>,
    @location(5) @interpolate(flat) key: f32,
    @location(6) @interpolate(flat) sunlit: f32,
}

fn basis(axis: vec3<f32>) -> mat2x3<f32> {
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(axis.x) > 0.9);
    let u = normalize(cross(axis, helper));
    return mat2x3<f32>(u, cross(axis, u));
}

fn hidden() -> ParticleOutput {
    var output: ParticleOutput;
    output.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    return output;
}

@vertex
fn vs_particle(@builtin(vertex_index) vertex: u32,
        @builtin(instance_index) instance: u32) -> ParticleOutput {
    let emitter = emitters[owners[instance]];
    let lane = instance - emitter.first;
    let time = camera.position.w;
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
    let cos_theta = mix(1.0, emitter.spread, r0);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = r1 * TAU;
    let across = basis(emitter.direction);
    let direction = emitter.direction * cos_theta
        + (across[0] * cos(phi) + across[1] * sin(phi)) * sin_theta;
    let ball = normalize(vec3<f32>(r2, r3, r4) * 2.0 - 1.0 + vec3<f32>(1.0e-4))
        * (emitter.radius * pow(r4, 1.0 / 3.0));
    let launch = direction * (emitter.speed * (0.75 + 0.5 * r2));
    let position = emitter.origin + ball - emitter.velocity * age + launch * age
        + emitter.acceleration * (0.5 * age * age);
    let velocity = launch + emitter.acceleration * age;
    let size = mix(emitter.size.x, emitter.size.y, life) * (0.8 + 0.4 * r3);
    let corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0),
    );
    let corner = corners[vertex % 4u];
    var along_x = particles.right.xyz * (size * 0.5);
    var along_y = particles.up.xyz * (size * 0.5);
    if emitter.stretch > 0.0 {
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
    output.sunlit = 1.0;
    if SHADOW_MAPS && shadow.state.x != 0.0 {
        output.sunlit = map_visibility(position, lighting.sun.xyz, vec3<f32>(0.0),
            vec3<f32>(0.0));
    }
    return output;
}

const MIST_SPREAD: f32 = 0.2;
const MIST_PHASE_G: f32 = 0.6;
const MIST_SUN: f32 = 0.35;

fn error_function(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let tail = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592) * t * exp(-a * a);
    return sign(x) * (1.0 - tail);
}

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

fn outside(input: ParticleOutput) -> bool {
    return dot(input.corner, input.corner) >= 1.0;
}

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
    let eye = camera.position.xyz;
    let distance = length(input.world - eye);
    var sunlit = input.sunlit;
    if lighting.sun.y > 0.0 {
        if SHADOW_MAPS && shadow.state.x != 0.0 {
            sunlit = min(sunlit, terrain_visibility(input.world, false));
        }
        sunlit *= cloud_passing(input.world);
    }
    let light = input.color.rgb * (sky_average(vec3<f32>(0.0, 1.0, 0.0)) * lighting.zenith.w
        + lighting.sun_color.rgb * (0.5 * sunlit));
    let view = normalize(input.world - eye);
    let fogged = atmospheric(light, eye, input.world, view).rgb;
    return weighted_particle(grade(fogged), alpha, distance);
}

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
    let pixel = vec2<i32>(input.clip_position.xy);
    let depth = textureLoad(depth_texture, pixel, 0);
    let uv = input.clip_position.xy * particles.size.zw;
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let behind = camera.inverse_view_projection * clip;
    let reach = length(behind.xyz / behind.w - eye);
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
    let sky = sky_average(vec3<f32>(0.0, 1.0, 0.0)) * lighting.zenith.w;
    let around = lighting.sun_color.rgb * (0.35 * saturate(sun.y)) + sky * 0.5;
    let light = input.color.rgb * (mix(around, sky, 0.5) + sunlight);
    let fogged = atmospheric(light, eye, closest, direction).rgb;
    return weighted_particle(grade(fogged), alpha, length(closest - eye));
}

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
    return vec4<f32>(encode_srgb(grade(added(input))), 0.0);
}

struct AddedTargets {
    @location(0) color: vec4<f32>,
    @location(1) glow: vec2<f32>,
}

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
