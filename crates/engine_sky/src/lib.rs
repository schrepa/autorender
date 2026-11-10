#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod clouds;
mod environment;
mod gpu_layouts;
mod haze;
#[cfg(not(target_arch = "wasm32"))]
mod pure;
mod settings;
mod sky;
pub mod tables;
mod views;

use bevy_ecs::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use engine_core::Corpus;
use engine_core::{
    AttachPoint, Brick, Bricks, JobKind, Pass, PassName, PassPurpose, PassViews, Schedule, Slot,
    SlotSet, Source, Writes, never,
};
use engine_places_rendering::lighting::{Ambient, SunBlockers};
use engine_places_rendering::sky::SkyLight;
use engine_render::RenderBricks;

pub use settings::{Clouds, Haze, NightSky, Sky};

pub(crate) struct HazePass;

impl PassName for HazePass {
    const NAME: &'static str = "haze";
    const PASS: Pass = Pass {
        writes: Writes::NOTHING,
        reads: SlotSet::NONE,
        views: PassViews::Main,
        at: AttachPoint::Post,
        purpose: PassPurpose::Scene,
        takes: never,
    };
}

#[derive(Resource, Default)]
struct CloudShadows;

impl Slot for CloudShadows {
    const NAME: &'static str = "sky_cloud_shadows";
    const WGSL: Option<&'static str> = Some(include_str!("places/cloud_shadows.wgsl"));
}

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
            .stats::<views::HazeStats>()
            .resource::<views::SkyLightLevel>()
            .resource::<tables::StaticTables>()
            .resource::<sky::SkyPlan>()
            .resource::<clouds::CloudSteps>()
            .resource::<haze::HazeSteps>()
            .resource::<environment::EnvironmentSteps>()
            .fill::<SkyLight>(Some(shaders::sky::faces::sky::MODULE))
            .join::<SunBlockers>((), Some(shaders::sky::blockers::MODULE))
            .join::<Ambient>((), Some(shaders::sky::ambient::MODULE))
            .within_switch::<Clouds>(|bricks| {
                bricks.fill::<CloudShadows>(Some(shaders::sky::faces::cloud_shadows::MODULE));
            })
            .within_switch::<Haze>(|bricks| {
                bricks.applies::<HazePass>(shaders::sky::haze::MODULE);
            })
            .systems(
                Schedule::ViewSync,
                (
                    tables::ask_tables,
                    sky::plan_sky,
                    publish_sky_light,
                    tables::table_stats,
                    views::sky_stats,
                ),
            )
            .generated(gpu_layouts::GENERATED)
            .part::<sky::SkyGpu>();
    }
}

fn publish_sky_light(sky: Res<Sky>, plan: Res<sky::SkyPlan>, mut light: ResMut<SkyLight>) {
    light.set_if_neq(SkyLight {
        sun_intensity: sky.sun_intensity,
        sun_size: sky.sun_size,
        adaptation: sky.adaptation,
        environment: plan.environment(),
    });
}
