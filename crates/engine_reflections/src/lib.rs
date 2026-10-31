#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine_core::{Brick, Bricks, Fill, Join};
use engine_faces::places::{PostEffectsList, ReflectionsSlot};
use engine_render::composite::PostScene;
use engine_render::render_bricks::AddRenderParts;

mod gpu_layouts;
mod layouts;
mod part;
mod settings;
mod shaders;
mod world_trace;

pub use settings::Reflections;

pub struct ReflectionsBrick;

impl Brick for ReflectionsBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .fill(Fill::<ReflectionsSlot>::new().wgsl("reflections::faces::reflections"))
            .join(Join::<PostScene>::new().wgsl(part::MEMBER))
            .join(Join::<PostEffectsList>::new())
            .settings::<Reflections>()
            .render_part(&part::PART)
            .shaders(shaders::MODULES)
            .places(engine_faces::places::PLACES)
            .pipeline_layouts(layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::LAYOUTS, gpu_layouts::DECLARED);
    }
}
