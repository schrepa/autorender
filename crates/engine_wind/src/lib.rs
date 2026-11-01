#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

pub mod constants;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, FrameTime, Schedule, Source, ViewWind, settings};
use engine_places::wind::{GustWaves, Wind};
use engine_render::RenderBricks;

use constants::{
    GUST_QUICK_ACROSS, GUST_QUICK_RATE, GUST_QUICK_SHARE, GUST_QUICK_TRAVEL, GUST_SLOW_RATE,
    GUST_SLOW_SHARE, GUST_SLOW_TRAVEL,
};

const WAVES: GustWaves = GustWaves {
    slow_share: GUST_SLOW_SHARE,
    slow_rate: GUST_SLOW_RATE,
    slow_travel: GUST_SLOW_TRAVEL,
    quick_share: GUST_QUICK_SHARE,
    quick_rate: GUST_QUICK_RATE,
    quick_travel: GUST_QUICK_TRAVEL,
    quick_across: GUST_QUICK_ACROSS,
};

const GENERATED: &[&str] = &[constants::WGSL];

settings! {
    #[section("wind")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneWind")]
    pub struct WindSettings {
        pub velocity: [f32; 2] = [2.5, 1.0] => range(-40.0, 40.0),
        pub gusts: f32 = 0.45 => range(0.0, 1.0),
    }
}

pub struct WindBrick;

impl Brick for WindBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<WindSettings>()
            .fill::<Wind>(Some(shaders::wind::faces::wind::MODULE))
            .fill::<ViewWind>(None)
            .generated(GENERATED)
            .systems(Schedule::FixedSimulation, publish_wind)
            .systems(Schedule::ViewSync, (publish_wind, publish_view_wind));
    }
}

fn publish_wind(settings: Res<WindSettings>, time: Res<FrameTime>, mut wind: ResMut<Wind>) {
    wind.set_if_neq(Wind {
        velocity: settings.velocity,
        gusts: settings.gusts,
        clock: time.elapsed_seconds as f32,
        waves: WAVES,
    });
}

fn publish_view_wind(settings: Res<WindSettings>, mut view: ResMut<ViewWind>) {
    view.set_if_neq(ViewWind {
        velocity: settings.velocity,
        gusts: settings.gusts,
    });
}
