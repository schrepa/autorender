use engine_core::{Brick, Bricks};
use engine_faces::slots::ParticlesList;

pub struct ParticlesBrick;

impl Brick for ParticlesBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.list::<ParticlesList>();
    }
}
