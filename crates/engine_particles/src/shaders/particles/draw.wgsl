import package::engine::math::{pcg, unit};
import package::engine::particles::{Particle, ParticleIn, standard};
import package::engine::view::camera;
import package::face::lighting::sun_visibility;
import package::face::particle_formula::custom_particle;
import package::face::sky::{ambient, atmospheric, atmospheric_cover, sky_average};
import package::face::sun::{sun_color, sun_direction};
import package::face::sun_blockers::SunAt;
import package::generated::particles::{Emitter, Particles};
import package::particles::weighted::{TransparentTargets, weighted_particle};

@group(1) @binding(0) var<uniform> particles: Particles;
@group(1) @binding(1) var<storage, read> emitters: array<Emitter>;
@group(1) @binding(2) var depth_texture: texture_depth_2d;

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

fn hidden() -> ParticleOutput {
    var output: ParticleOutput;
    output.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    return output;
}

fn owner_of(instance: u32) -> u32 {
    var low = 0u;
    var high = particles.listed.x;
    while high - low > 1u {
        let middle = (low + high) / 2u;
        if emitters[middle].first <= instance {
            low = middle;
        } else {
            high = middle;
        }
    }
    return low;
}

@vertex
fn vs_particle(@builtin(vertex_index) vertex: u32,
        @builtin(instance_index) instance: u32) -> ParticleOutput {
    let emitter = emitters[owner_of(instance)];
    let lane = instance - emitter.first;
    let time = camera.position.w;
    var k = lane;
    var age = time - emitter.start;
    if emitter.rate > 0.0 {
        let births = floor(time * emitter.rate);
        if births < f32(lane) {
            return hidden();
        }
        let latest = u32(births);
        k = latest - (latest - lane) % emitter.count;
        let born = f32(k) / emitter.rate;
        if born > emitter.stop && born < emitter.start {
            return hidden();
        }
        age = time - born;
    }
    let seed = pcg(emitter.seed ^ pcg(k));
    let own = emitter.lifetime * (1.0 - unit(pcg(seed + 6u)) * emitter.lifetime_spread);
    if age < 0.0 || age > own {
        return hidden();
    }
    let life = age / own;
    let input = ParticleIn(seed, age, life, own, emitter.origin, emitter.velocity,
        emitter.direction, emitter.speed, emitter.speed_spread, emitter.acceleration,
        emitter.spread, emitter.ball != 0u, emitter.extent, emitter.rotation, emitter.size,
        emitter.size_spread, emitter.start_color, emitter.end_color, emitter.fade_in,
        emitter.brightness, emitter.params);
    var particle: Particle;
    if emitter.formula == 0u {
        particle = standard(input);
    } else {
        particle = custom_particle(emitter.formula, input);
    }
    if particle.hidden {
        return hidden();
    }
    let position = particle.position;
    let velocity = particle.velocity;
    let size = particle.size;
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
    output.color = particle.color;
    output.corner = corner;
    output.world = world;
    output.softness = max(size, 0.05);
    output.center = vec4<f32>(position, size * MIST_SPREAD);
    output.key = unit(pcg(seed + 5u)) * 256.0;
    output.sunlit = sun_visibility(SunAt(position, sun_direction(), vec3<f32>(0.0),
        vec3<f32>(0.0), 0.0, false, 0u, vec4<f32>(0.0)));
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
    let light = input.color.rgb * (sky_average(vec3<f32>(0.0, 1.0, 0.0)) * ambient()
        + sun_color() * (0.5 * input.sunlit));
    let view = normalize(input.world - eye);
    let fogged = atmospheric(light, eye, input.world, view).rgb;
    return weighted_particle(fogged, alpha, distance);
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
    let sun = sun_direction();
    var sunlight = vec3<f32>(0.0);
    if sun.y > 0.0 {
        let g = MIST_PHASE_G;
        let denominator = 1.0 + g * g - 2.0 * g * dot(sun, direction);
        let phase = (1.0 - g * g) / (denominator * sqrt(denominator));
        sunlight = sun_color() * (MIST_SUN * phase) * input.sunlit;
    }
    let sky = sky_average(vec3<f32>(0.0, 1.0, 0.0)) * ambient();
    let around = sun_color() * (0.35 * saturate(sun.y)) + sky * 0.5;
    let light = input.color.rgb * (mix(around, sky, 0.5) + sunlight);
    let fogged = atmospheric(light, eye, closest, direction).rgb;
    return weighted_particle(fogged, alpha, length(closest - eye));
}

fn added(input: ParticleOutput) -> vec3<f32> {
    let eye = camera.position.xyz;
    let light = input.color.rgb * (input.color.a * coverage(input));
    return light * (1.0 - atmospheric_cover(eye, input.world));
}

@fragment
fn fs_particle_add_post(input: ParticleOutput) -> @location(0) vec4<f32> {
    if outside(input) {
        discard;
    }
    return vec4<f32>(added(input), 0.0);
}
