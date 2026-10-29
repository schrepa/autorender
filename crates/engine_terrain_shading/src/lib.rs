use engine_core::{Brick, Bricks};
use engine_faces::slots::TerrainShadingList;

pub struct TerrainShadingBrick;

impl Brick for TerrainShadingBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.list::<TerrainShadingList>();
    }
}
