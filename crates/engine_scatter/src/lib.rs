use engine_core::{Brick, Bricks};
use engine_faces::slots::ScatterList;

pub struct ScatterBrick;

impl Brick for ScatterBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.list::<ScatterList>();
    }
}
