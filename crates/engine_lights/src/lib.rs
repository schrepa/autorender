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
use engine_core::{
    Brick, Bricks, ComponentKind, GpuBits, GpuStruct, SceneComponents, Schedule, Source,
};
use engine_places_rendering::lights::{Lights, LocalLight};
use engine_render::RenderBricks;

pub use lights::{LightStats, LocalLightSettings};

const GENERATED: &[&str] = &[
    gpu::WGSL,
    <gpu::Clusters as GpuStruct>::WGSL,
    <gpu::LightInput as GpuStruct>::WGSL,
    <gpu::LightPlace as GpuBits>::WGSL,
    <gpu::LightShading as GpuBits>::WGSL,
];

pub struct LightsBrick;

impl Brick for LightsBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|settings: Res<LocalLightSettings>| settings.enabled)
            .settings::<LocalLightSettings>()
            .stats::<LightStats>()
            .resource::<lights::LightGather>()
            .resource::<lights::LightList>()
            .resource::<part::LightTotals>()
            .fill::<Lights>(Some(shaders::lights::faces::lights::MODULE))
            .join::<SceneComponents>(
                ComponentKind::of::<LocalLight>("light")
                    .check(scene::check_light)
                    .schema(scene::light_schema),
                None,
            )
            .systems(
                Schedule::ViewSync,
                (lights::gather_lights, lights::light_stats),
            )
            .generated(GENERATED)
            .part::<part::LightsGpu>();
    }
}
