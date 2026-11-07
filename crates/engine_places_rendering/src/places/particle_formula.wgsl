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

fn custom_particle(id: u32, input: ParticleIn) -> Particle {
    var particle: Particle;
    particle.hidden = true;
    return particle;
}
