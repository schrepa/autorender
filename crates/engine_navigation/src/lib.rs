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
use engine_core::{Brick, Bricks, Cook, CookList, JobKind, SceneComponents, Schedule, Source};
use engine_places_simulation::ground::UNKNOWN_TERRAIN;

pub use agents::{NavAgents, NavQuery, NavStats, Navigation, ObstacleColliders, SceneNavAgent};
pub use dev::{
    GroundKey, NavAgentQuery, NavAgentRow, NavAgentView, NavAgentsQuery, NavAgentsView,
    NavGoControl, NavGoResult, NavPathQuery, NavPathResult, WalkResult,
};
pub use jobs::{GroundJob, PORTALS, PortalsJob, WALK_GROUND, portals_output};
pub use nav::{
    Gait, NavOptions, NavPath, PORTAL_RUN, Portal, Side, Stepped, TilePortals, WalkData,
    WalkGround, WalkSearch, Want,
};

pub struct NavigationBrick;

impl Brick for NavigationBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&jobs::WALK_GROUND, &jobs::PORTALS];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|navigation: Res<Navigation>| navigation.enabled)
            .settings::<Navigation>()
            .errors(&[scene::INVALID_NAV_AGENT, UNKNOWN_TERRAIN])
            .resource::<NavAgents>()
            .stats::<NavStats>()
            .systems(Schedule::FixedMove, agents::walk_agents)
            .operation::<NavGoControl>()
            .operation::<NavPathQuery>()
            .observation::<NavAgentQuery>()
            .observation::<NavAgentsQuery>()
            .join::<SceneComponents>(scene::nav_agent(), None)
            .join::<CookList>(Cook::new(agents::cook), None);
    }
}
