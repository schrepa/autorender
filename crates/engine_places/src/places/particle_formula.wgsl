import package::engine::particles::{Particle, ParticleIn};

fn custom_particle(id: u32, input: ParticleIn) -> Particle {
    var particle: Particle;
    particle.hidden = true;
    return particle;
}
