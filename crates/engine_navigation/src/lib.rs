#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod agents;
mod dev;
pub mod jobs;
mod nav;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, ComponentKind, JobKind, SceneComponents, Schedule, Source};
use engine_places::navigation::NavAgent;

pub use agents::{NavAgents, NavQuery, NavStats, Navigation};
pub use dev::{
    GroundKey, NavAgentQuery, NavAgentRow, NavAgentView, NavAgentsQuery, NavAgentsView,
    NavGoControl, NavGoResult, NavPathQuery, NavPathResult, WalkResult,
};
pub use jobs::{GroundJob, PORTALS, PortalsJob, WALK_GROUND, portals_output};
pub use nav::{
    Gait, NavOptions, NavPath, PORTAL_RUN, Portal, Side, TilePortals, WalkData, WalkGround,
    WalkSearch, Want,
};

pub struct NavigationBrick;

impl Brick for NavigationBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&jobs::WALK_GROUND, &jobs::PORTALS];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|navigation: Res<Navigation>| navigation.enabled)
            .settings::<Navigation>()
            .stats::<NavStats>()
            .resource::<NavAgents>()
            .join::<SceneComponents>(ComponentKind::of::<NavAgent>("nav_agent"), None)
            .systems(
                Schedule::FixedMove,
                (agents::walk_agents, agents::hold_heights),
            )
            .off(agents::stop_walks)
            .operation::<NavGoControl>()
            .operation::<NavPathQuery>()
            .observation::<NavAgentsQuery>()
            .observation::<NavAgentQuery>();
    }
}
