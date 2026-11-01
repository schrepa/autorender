include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod atlas;
pub mod constants;
mod gpu_layouts;
mod inspect;
mod part;
mod plan;
mod regions;
mod settings;
mod shadow_fit;
mod sky;
mod smokes;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Runs, Schedule, Smoke, Source};
use engine_places::shadows::{CasterRoutes, ShadowCasters, ShadowMaps};
use engine_render::RenderBricks;

pub use constants::{MAX_MOVING_TILES, MAX_SHADOW_LEVELS};
pub use settings::{ShadowSettings, ShadowStats, SkyOcclusion};

pub struct ShadowsBrick;

impl Brick for ShadowsBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[Smoke {
        name: "shadow_tiles_settle",
        runs: Runs::Gpu,
        run: smokes::tiles_settle,
    }];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|shadows: Res<ShadowSettings>| shadows.enabled)
            .settings::<ShadowSettings>()
            .settings::<SkyOcclusion>()
            .stats::<ShadowStats>()
            .resource::<plan::ShadowFeedback>()
            .resource::<plan::ShadowPlan>()
            .piece::<gpu_layouts::ShadowPiece>()
            .fill::<ShadowMaps>(Some(shaders::shadows::faces::shadows::MODULE))
            .fill::<CasterRoutes>(Some(shaders::shadows::faces::routes::MODULE))
            .fill::<ShadowCasters>(None)
            .systems(Schedule::ViewSync, (plan::plan_shadows, plan::shadow_stats))
            .off(plan::forget_shadows)
            .observation::<inspect::InspectShadowTile>()
            .part::<part::ShadowsGpu>();
    }
}
