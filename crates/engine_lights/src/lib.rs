#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod gpu;
mod lights;
mod part;
mod scene;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, ComponentKind, GpuStruct, SceneComponents, Schedule, Source};
use engine_places::lights::{Lights, LocalLight};
use engine_render::RenderBricks;

pub use lights::{LightStats, LocalLightSettings};

const GENERATED: &[&str] = &[
    gpu::WGSL,
    <gpu::Clusters as GpuStruct>::WGSL,
    <gpu::LightInput as GpuStruct>::WGSL,
];

pub struct LightsBrick;

impl Brick for LightsBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(lit)
            .settings::<LocalLightSettings>()
            .stats::<LightStats>()
            .resource::<lights::LightGather>()
            .resource::<lights::LightList>()
            .resource::<part::LightTotals>()
            .fill::<Lights>(Some(shaders::lights::faces::lights::MODULE))
            .join::<SceneComponents>(
                ComponentKind::of::<LocalLight>("light")
                    .check(scene::check_light)
                    .reach(scene::light_reach)
                    .schema(scene::light_schema)
                    .example(scene::light_example),
                None,
            )
            .systems(
                Schedule::ViewSync,
                (lights::gather_lights, lights::light_stats),
            )
            .off(lights::forget_lights)
            .generated(GENERATED)
            .part::<part::LightsGpu>();
    }
}

fn lit(settings: Res<LocalLightSettings>, lights: Query<(), With<LocalLight>>) -> bool {
    settings.active() && !lights.is_empty()
}
