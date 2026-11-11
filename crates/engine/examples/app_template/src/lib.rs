use engine::prelude::*;
use engine_fly_camera::FlyCameraBrick;

pub mod bricks {
    pub mod glimmer;
    pub mod turning;
}

use crate::bricks::glimmer::GlimmerBrick;
use crate::bricks::turning::TurningBrick;

#[derive(Resource, Default)]
pub struct Template;

impl Game for Template {}

pub fn bricks<L: Listing>(list: L) -> L {
    engine_standard_rendering::rendering(list)
        .with::<FlyCameraBrick>()
        .with::<TurningBrick>()
        .with::<GlimmerBrick>()
}

engine::game!(
    pub struct TemplateGame(Template),
    "Template",
    bricks,
    scene = "template.scene.json"
);
