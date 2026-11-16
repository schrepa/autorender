#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod clouds;
mod environment;
mod gpu_layouts;
#[cfg(not(target_arch = "wasm32"))]
mod pure;
mod settings;
mod sky;
pub mod tables;
mod views;

use bevy_ecs::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use engine_core::Corpus;
use engine_core::{Brick, Bricks, JobKind, Schedule, Source};
use engine_places_rendering::lighting::Ambient;
use engine_places_rendering::sky::SkyLight;
use engine_render::RenderBricks;

pub use settings::{Clouds, NightSky, Sky};

pub struct SkyBrick;

impl Brick for SkyBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&tables::SKY_TABLES, &tables::CLOUD_TABLES];
    #[cfg(not(target_arch = "wasm32"))]
    const CORPORA: &'static [Corpus] = &[Corpus {
        name: "tables",
        replays: Self::JOBS,
        run: pure::tables::run,
    }];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .switch::<Sky>()
            .settings::<NightSky>()
            .stats::<views::StaticTablesView>()
            .stats::<views::CloudStats>()
            .stats::<views::SkyStats>()
            .stats::<views::LightingStats>()
            .resource::<views::SkyLightLevel>()
            .resource::<tables::StaticTables>()
            .resource::<sky::SkyPlan>()
            .resource::<clouds::CloudSteps>()
            .resource::<environment::EnvironmentSteps>()
            .fill::<SkyLight>(Some(shaders::sky::faces::sky::MODULE))
            .join::<Ambient>((), Some(shaders::sky::ambient::MODULE))
            .within_switch::<Clouds>(|_| {})
            .systems(
                Schedule::ViewSync,
                (
                    tables::ask_tables,
                    sky::plan_sky,
                    publish_sky_light,
                    tables::table_stats,
                    views::sky_stats,
                    views::lighting_stats,
                ),
            )
            .generated(gpu_layouts::GENERATED)
            .part::<sky::SkyGpu>();
    }
}

fn publish_sky_light(sky: Res<Sky>, plan: Res<sky::SkyPlan>, mut light: ResMut<SkyLight>) {
    light.set_if_neq(SkyLight {
        sun_size: sky.sun_size,
        environment: plan.environment(),
    });
}
