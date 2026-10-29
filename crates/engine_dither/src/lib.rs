use engine_core::{Brick, Bricks};
use engine_faces::slots::PostEffectsList;

pub struct DitherBrick;

impl Brick for DitherBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.list::<PostEffectsList>();
    }
}
