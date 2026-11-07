#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod captures;
mod component;
mod gather;
mod probes;

use bevy_ecs::prelude::*;
use engine_core::{
    AttachPoint, Brick, Bricks, ComponentKind, Cut, DrawKind, Pass, PassName, PassViews,
    SceneComponents, Schedule, SlotSet, Source, Writes, gpu_consts, settings, stats,
};
use engine_places_rendering::lighting::{Ambient, Interiors};
use engine_places_rendering::probes::{ProbeCaptures, ReflectionProbe};
use engine_render::RenderBricks;

pub use component::{SceneInterior, SceneReflectionProbe};

gpu_consts! {
    pub const PROBE_ENTRY_WORDS: u32 = 3;
}

pub struct ProbePass;

impl PassName for ProbePass {
    const NAME: &'static str = "probe";
    const PASS: Pass = Pass {
        writes: Writes::LIT,
        reads: SlotSet::ALL,
        views: PassViews::Owned,
        at: AttachPoint::Opaque,
        takes: reflected,
    };
}

fn reflected(kind: DrawKind) -> bool {
    !matches!(
        kind.name(),
        "particles" | "sprites" | "text" | "ui" | "water" | "cover"
    )
}

settings! {
    #[section("reflection_probes")]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneReflectionProbes")]
    pub struct ReflectionProbeSettings {
        pub enabled: bool = true => any,
        pub resolution: u32 = 128 => power_of_two(16, 1024),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_probes: Option<u32> = None => budget,
    }
}

stats! {
    #[section("reflection_probes_stats")]
    pub struct ReflectionProbeStats {
        pub ready: u32,
        #[pending] pub waiting: u32,
        pub faces: u32,
        pub probe_cut: Cut,
    }
}

const GENERATED: &[&str] = &[WGSL];

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
            .resource::<probes::ProbeAcks>()
            .join::<SceneComponents>(
                ComponentKind::of::<ReflectionProbe>("reflection_probe")
                    .check(component::check)
                    .schema(component::schema)
                    .example(component::example),
                None,
            )
            .join::<Ambient>((), Some(shaders::probes::ambient::MODULE))
            .join::<Interiors>((), Some(shaders::probes::interior::MODULE))
            .fill::<ProbeCaptures>(None)
            .systems(
                Schedule::ViewSync,
                (
                    gather::gather_probes,
                    probes::plan_captures,
                    probes::publish_captures,
                ),
            )
            .generated(GENERATED)
            .pass::<ProbePass>()
            .part::<captures::ProbesGpu>();
    }
}
