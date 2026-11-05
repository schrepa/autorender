import package::engine::math::{TAU, pcg, unit};

struct ParticleIn {
    seed: u32,
    age: f32,
    life: f32,
    lifetime: f32,
    origin: vec3<f32>,
    emitter_velocity: vec3<f32>,
    direction: vec3<f32>,
    speed: f32,
    speed_spread: f32,
    acceleration: vec3<f32>,
    spread: f32,
    ball: bool,
    extent: vec3<f32>,
    rotation: vec4<f32>,
    size: vec2<f32>,
    size_spread: f32,
    start_color: vec4<f32>,
    end_color: vec4<f32>,
    fade_in: f32,
    brightness: f32,
    params: array<f32, 8>,
}

struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    size: f32,
    color: vec4<f32>,
    hidden: bool,
}

fn basis(axis: vec3<f32>) -> mat2x3<f32> {
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(axis.x) > 0.9);
    let u = normalize(cross(axis, helper));
    return mat2x3<f32>(u, cross(axis, u));
}

fn turned(rotation: vec4<f32>, point: vec3<f32>) -> vec3<f32> {
    let lever = cross(rotation.xyz, point) + rotation.w * point;
    return point + 2.0 * cross(rotation.xyz, lever);
}

fn standard(input: ParticleIn) -> Particle {
    let r0 = unit(input.seed);
    let r1 = unit(pcg(input.seed + 1u));
    let r2 = unit(pcg(input.seed + 2u));
    let r3 = unit(pcg(input.seed + 3u));
    let r4 = unit(pcg(input.seed + 4u));
    let cos_theta = mix(1.0, input.spread, r0);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = r1 * TAU;
    let across = basis(input.direction);
    let direction = input.direction * cos_theta
        + (across[0] * cos(phi) + across[1] * sin(phi)) * sin_theta;
    let cube = vec3<f32>(r2, r3, r4) * 2.0 - 1.0;
    var shaped = cube * input.extent;
    if input.ball {
        shaped = normalize(cube + vec3<f32>(1.0e-4)) * (input.extent * pow(r4, 1.0 / 3.0));
    }
    let launch = direction
        * (input.speed * (1.0 - input.speed_spread + 2.0 * input.speed_spread * r2));
    let position = input.origin + turned(input.rotation, shaped)
        - input.emitter_velocity * input.age + launch * input.age
        + input.acceleration * (0.5 * input.age * input.age);
    let velocity = launch + input.acceleration * input.age;
    let size = mix(input.size.x, input.size.y, input.life)
        * (1.0 - input.size_spread + 2.0 * input.size_spread * r3);
    let color = mix(input.start_color, input.end_color, input.life);
    var alpha = color.a;
    if input.fade_in > 0.0 {
        alpha *= min(input.life / input.fade_in, 1.0);
    }
    return Particle(position, velocity, size, vec4<f32>(color.rgb * input.brightness, alpha),
        false);
}
