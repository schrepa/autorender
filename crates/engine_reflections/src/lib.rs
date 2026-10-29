use engine_core::{Brick, Bricks};
use engine_faces::slots::PostEffectsList;

pub struct ReflectionsBrick;

impl Brick for ReflectionsBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.list::<PostEffectsList>();
    }
}
