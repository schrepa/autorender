use engine_core::{Brick, Bricks};
use engine_faces::slots::WaterList;

pub struct WaterBrick;

impl Brick for WaterBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.list::<WaterList>();
    }
}
