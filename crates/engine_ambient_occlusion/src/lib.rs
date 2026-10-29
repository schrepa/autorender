use engine_core::{Brick, Bricks};
use engine_faces::slots::PostEffectsList;

pub struct AmbientOcclusionBrick;

impl Brick for AmbientOcclusionBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.list::<PostEffectsList>();
    }
}
