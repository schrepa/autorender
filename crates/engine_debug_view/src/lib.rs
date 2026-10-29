use engine_core::{Brick, Bricks};
use engine_faces::slots::DebugViewList;

pub struct DebugViewBrick;

impl Brick for DebugViewBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.list::<DebugViewList>();
    }
}
