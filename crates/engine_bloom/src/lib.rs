use engine_core::{Brick, Bricks};
use engine_faces::slots::PostEffectsList;

pub struct BloomBrick;

impl Brick for BloomBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.list::<PostEffectsList>();
    }
}
