#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

pub mod gpu_layouts;
pub mod particles;
pub mod pipeline_layouts;
pub mod shaders;

use engine_core::{
    Brick, Bricks, DetailConsumer, DetailConsumersList, Join, ParticleExtractor, ParticleSettings,
    extract_particles,
};
use engine_faces::places::ParticlesList;
use engine_render::render_bricks::AddRenderParts;

pub struct ParticlesBrick;

impl Brick for ParticlesBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join(Join::<ParticlesList>::new())
            .settings::<ParticleSettings>()
            .join(Join::<DetailConsumersList>::new().cpu(DetailConsumer {
                name: "particles",
                setting: "particles.max_particles",
                stat: "particles.particle_cut",
            }))
            .resource::<ParticleExtractor>()
            .extract(extract_particles)
            .render_part(&particles::PART)
            .shaders(shaders::MODULES)
            .places(engine_faces::places::PLACES)
            .pipeline_layouts(pipeline_layouts::PIPELINE_LAYOUTS)
            .mirrors(gpu_layouts::LAYOUTS, &[]);
    }
}
