include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod atlas;
pub mod constants;
mod debug;
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
use engine_core::{
    AttachPoint, Brick, Bricks, DrawKind, Pass, PassName, PassViews, Runs, Schedule, SlotSet,
    Smoke, Source, Writes,
};
use engine_places_rendering::lighting::SunBlockers;
use engine_places_rendering::shadows::{CasterRoutes, ShadowMaps};
use engine_render::{DebugModes, PostScene, RenderBricks};

pub use settings::{ShadowSettings, ShadowStats, SkyOcclusion};

pub struct ShadowPass;

impl PassName for ShadowPass {
    const NAME: &'static str = "shadow";
    const PASS: Pass = Pass {
        writes: Writes::DEPTH,
        reads: SlotSet::MASK,
        views: PassViews::Owned,
        at: AttachPoint::Depth,
        takes: casts,
    };
}

fn casts(kind: DrawKind) -> bool {
    !matches!(
        kind.name(),
        "particles" | "sprites" | "text" | "ui" | "water" | "cover"
    )
}

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
            .resource::<plan::ShadowDrawn>()
            .resource::<plan::ShadowPlan>()
            .piece::<gpu_layouts::ShadowPiece>()
            .fill::<ShadowMaps>(Some(shaders::shadows::faces::shadows::MODULE))
            .fill::<CasterRoutes>(Some(shaders::shadows::faces::routes::MODULE))
            .join::<SunBlockers>((), Some(shaders::shadows::blockers::MODULE))
            .join::<PostScene>((), Some(shaders::shadows::sky_open::MODULE));
        for (index, (name, _)) in debug::MODES.into_iter().enumerate() {
            let wgsl = (index == 0).then_some(shaders::shadows::debug::MODULE);
            bricks.join::<DebugModes>(debug::member(name), wgsl);
        }
        bricks
            .systems(Schedule::ViewSync, (plan::plan_shadows, plan::shadow_stats))
            .observation::<inspect::InspectShadowTile>()
            .generated(gpu_layouts::GENERATED)
            .pass::<ShadowPass>()
            .part::<part::ShadowsGpu>();
    }
}
