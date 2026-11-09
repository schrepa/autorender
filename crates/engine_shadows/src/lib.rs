#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod atlas;
mod changes;
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

use bevy_ecs::prelude::*;
use engine_core::{
    AttachPoint, Brick, Bricks, KindDecl, Pass, PassCull, PassName, PassRaster, PassViews,
    Schedule, SlotSet, Source, Writes,
};
use engine_places_rendering::lighting::SunBlockers;
use engine_places_rendering::sky::SkyVisibility;
use engine_places_rendering::shadows::{CasterRoutes, SHADOW_PASS, ShadowMaps};
use engine_render::{DebugModes, RenderBricks};

pub use settings::{ShadowSettings, ShadowStats, SkyOcclusion};

pub struct ShadowPass;

impl PassName for ShadowPass {
    const NAME: &'static str = SHADOW_PASS;
    const PASS: Pass = Pass {
        writes: Writes::DEPTH,
        reads: SlotSet::MASK,
        views: PassViews::Owned,
        at: AttachPoint::Depth,
        takes: casts,
    };
    const RASTER: PassRaster = PassRaster {
        bias: 1,
        slope: 0.0,
        clamp_near: true,
        cull: PassCull::None,
    };
}

fn casts(kind: &KindDecl) -> bool {
    kind.casts_shadows
}

pub struct ShadowsBrick;

impl Brick for ShadowsBrick {
    const SOURCE: &'static Source = &SOURCE;
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
            .fill::<SkyVisibility>(Some(shaders::shadows::faces::sky_visibility::MODULE));
        for (index, (name, _)) in debug::MODES.into_iter().enumerate() {
            let wgsl = (index == 0).then_some(shaders::shadows::debug::MODULE);
            bricks.join::<DebugModes>(debug::member(name), wgsl);
        }
        bricks
            .systems(
                Schedule::ViewSync,
                (
                    changes::publish_shadow_changes,
                    plan::plan_shadows,
                    plan::shadow_stats,
                ),
            )
            .observation::<inspect::InspectShadowTile>()
            .generated(gpu_layouts::GENERATED)
            .pass::<ShadowPass>()
            .part::<part::ShadowsGpu>();
    }
}
