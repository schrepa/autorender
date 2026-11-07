include!(concat!(env!("OUT_DIR"), "/bricks/toon_lighting.rs"));

use engine_core::{Brick, Bricks, Source};
use engine_places_rendering::lighting::Lighting;
use engine_render::Lit;

pub struct ToonLightingBrick;

impl Brick for ToonLightingBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .fill::<Lighting>(Some(shaders::toon_lighting::faces::lighting::MODULE))
            .fill::<Lit>(Some(shaders::toon_lighting::faces::lighting::MODULE));
    }
}
