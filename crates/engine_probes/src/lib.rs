#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

mod captures;
pub mod gpu_layouts;
pub mod probes;
pub mod shaders;

use engine_core::{
    Brick, Bricks, DetailConsumer, ProbeExtractor, ReflectionProbeSettings, extract_probes,
};
use engine_faces::slots::ProbesList;
use engine_render::render_bricks::AddRenderParts;

pub struct ReflectionProbesBrick;

impl Brick for ReflectionProbesBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .list::<ProbesList>()
            .settings::<ReflectionProbeSettings>()
            .detail_consumer(DetailConsumer {
                name: "probes",
                setting: "reflection_probes.max_probes",
                stat: "reflection_probes.probe_cut",
            })
            .resource::<ProbeExtractor>()
            .extract(extract_probes)
            .render_part(&probes::PART)
            .shaders(shaders::MODULES)
            .face_defaults(engine_faces::defaults::MODULES)
            .mirrors(&[], gpu_layouts::DECLARED);
    }
}
