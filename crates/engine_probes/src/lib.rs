#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod captures;
mod gather;
mod probes;

use bevy_ecs::prelude::*;
use engine_core::{
    Brick, Bricks, ComponentKind, Cut, SceneComponents, Schedule, Source, gpu_consts, settings,
    stats,
};
use engine_places::probes::{ProbeCaptures, ProbeLight, ProbeViews, ReflectionProbe};
use engine_render::RenderBricks;

gpu_consts! {
    pub const PROBE_ENTRY_WORDS: u32 = 3;
}

settings! {
    #[section("reflection_probes")]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneReflectionProbes")]
    pub struct ReflectionProbeSettings {
        pub enabled: bool = true => any,
        pub resolution: u32 = 128 => range(16, 1024),
        pub max_probes: Option<u32> = None => at_least(1),
        pub faces_per_frame: u32 = 1 => at_least(1),
    }
}

stats! {
    #[section("reflection_probes_stats")]
    pub struct ReflectionProbeStats {
        pub ready: u32,
        pub faces: u32,
        pub probe_cut: Cut,
    }
}

pub struct ReflectionProbesBrick;

impl Brick for ReflectionProbesBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|probes: Res<ReflectionProbeSettings>| probes.enabled)
            .settings::<ReflectionProbeSettings>()
            .stats::<ReflectionProbeStats>()
            .resource::<gather::ProbeList>()
            .resource::<probes::ProbeState>()
            .resource::<probes::ProbeWords>()
            .join::<SceneComponents>(
                ComponentKind::of::<ReflectionProbe>("reflection_probe"),
                None,
            )
            .fill::<ProbeLight>(Some(shaders::probes::faces::probe_light::MODULE))
            .fill::<ProbeCaptures>(None)
            .fill::<ProbeViews>(None)
            .systems(
                Schedule::ViewSync,
                (
                    gather::gather_probes,
                    probes::plan_captures,
                    probes::publish_captures,
                ),
            )
            .off(probes::forget_probes)
            .part::<captures::ProbesGpu>();
    }
}
