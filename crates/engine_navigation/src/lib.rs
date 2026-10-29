#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/source.rs"));

mod agents;
mod dev;
pub mod jobs;
mod nav;
mod smokes;

pub use agents::{
    NavAgent, NavAgents, NavQuery, NavStats, NavStatus, NavSystems, NavWalk, Navigation,
    NavigationBrick,
};
pub use dev::{
    GroundKey, NavAgentRow, NavAgentView, NavAgentsView, NavGoControl, NavGoResult, NavPathQuery,
    NavPathResult, WalkResult,
};
pub use jobs::{NavigationWorker, PortalsJob, portals_output};
pub use nav::{
    Gait, NavOptions, NavPath, PORTAL_RUN, Portal, Side, TilePortals, WalkData, WalkGround,
    WalkSearch, Want,
};
