use engine_core::{Brick, Bricks};
use engine_faces::slots::WaterList;

pub struct PondBrick;

impl Brick for PondBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.list::<WaterList>();
    }
}
