use engine_core::{Brick, Bricks};
use engine_faces::slots::GroundCoverList;

pub struct GroundCoverBrick;

impl Brick for GroundCoverBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.list::<GroundCoverList>();
    }
}
