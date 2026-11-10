#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod agents;
mod dev;
pub mod jobs;
mod nav;
mod scene;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, JobKind, SceneComponents, Schedule, Source};

pub use agents::{
    NavAgents, NavQuery, NavStats, Navigation, ObstacleColliders, SceneNavAgent, WalkArea,
};
pub use dev::{
    NavAgentQuery, NavAgentRow, NavAgentView, NavAgentsQuery, NavAgentsView, NavGoControl,
    NavGoResult, NavPathQuery, NavPathResult, WalkResult,
};
pub use jobs::{PORTALS, PortalsJob, portals_output};
pub use nav::{
    Gait, HOLE_WORDS, NavOptions, NavPath, PORTAL_RUN, Portal, Side, Stepped, TilePortals,
    WALK_SAMPLES, WalkData, WalkGround, WalkObstacle, WalkSearch, Want,
};

pub struct NavigationBrick;

impl Brick for NavigationBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&jobs::PORTALS];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|navigation: Res<Navigation>| navigation.enabled)
            .settings::<Navigation>()
            .errors(&[scene::INVALID_NAV_AGENT])
            .resource::<NavAgents>()
            .stats::<NavStats>()
            .systems(Schedule::FixedMove, agents::walk_agents)
            .systems(Schedule::WorldSync, agents::settle_walk_records)
            .operation::<NavGoControl>()
            .operation::<NavPathQuery>()
            .observation::<NavAgentQuery>()
            .observation::<NavAgentsQuery>()
            .join::<SceneComponents>(scene::nav_agent(), None);
    }
}
