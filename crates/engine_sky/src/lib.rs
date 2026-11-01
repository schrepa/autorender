#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod clouds;
mod environment;
mod gpu_layouts;
mod haze;
mod settings;
mod sky;
mod tables;
mod under_eye;
mod views;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, JobKind, Schedule, Slot, Source};
use engine_places::probes::ProbeViews;
use engine_places::sky::SkyLight;
use engine_render::{Layer, MainView, PostLight, RenderBricks};

pub use settings::{Clouds, Haze, NightSky, Sky, SkyLighting, SkyLightingValues};

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

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<Sky>()
            .settings::<Clouds>()
            .settings::<Haze>()
            .settings::<NightSky>()
            .settings::<SkyLighting>()
            .stats::<views::StaticTablesView>()
            .stats::<views::CloudStats>()
            .stats::<views::SkyStats>()
            .resource::<tables::StaticTables>()
            .resource::<under_eye::SurfaceUnderEye>()
            .resource::<views::SkyCounts>()
            .fill::<SkyLight>(Some(shaders::sky::faces::sky::MODULE))
            .within(casts_cloud_shadows, |bricks| {
                bricks.fill::<CloudShadows>(Some(shaders::sky::faces::cloud_shadows::MODULE));
            })
            .within(
                |haze: Res<Haze>| haze.enabled,
                |bricks| {
                    bricks.join::<PostLight>((), Some(shaders::sky::haze::MODULE));
                },
            )
            .systems(
                Schedule::ViewSync,
                (
                    tables::ask_tables,
                    under_eye::find_surface_under_eye,
                    publish_sky_light,
                    tables::table_stats,
                    views::sky_stats,
                ),
            )
            .draws::<sky::SkyGpu, MainView>(Layer::Backdrop, sky::SkyGpu::draw_dome)
            .draws::<sky::SkyGpu, ProbeViews>((), sky::SkyGpu::draw_probe_face);
    }
}

fn casts_cloud_shadows(clouds: Res<Clouds>) -> bool {
    clouds.enabled && clouds.coverage > 0.0 && clouds.shadows > 0.0
}

fn publish_sky_light(sky: Res<Sky>, mut light: ResMut<SkyLight>) {
    light.set_if_neq(SkyLight {
        on: sky.enabled,
        sun_intensity: sky.sun_intensity,
        sun_size: sky.sun_size,
        adaptation: sky.adaptation,
    });
}
