#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

extern crate engine_ecs as bevy_ecs;

pub mod ground;
pub mod navigation;
pub mod physics;
pub mod solids;
pub mod sweeps;
pub mod water;
pub mod wind;

use engine_place::PlaceDecl;

pub const PLACES: &[PlaceDecl] = &[
    PlaceDecl::slot::<ground::Ground>(),
    PlaceDecl::component::<ground::Grounded>("grounded"),
    PlaceDecl::component::<ground::GroundedAt>("grounded_at"),
    PlaceDecl::component::<navigation::NavAgent>("nav_agent"),
    PlaceDecl::state::<navigation::NavWalk>("nav_walk"),
    PlaceDecl::slot::<water::WaterObstacles>(),
    PlaceDecl::component::<water::WaterBody>("water_body"),
    PlaceDecl::slot::<wind::Wind>(),
    PlaceDecl::slot::<physics::Collision>(),
    PlaceDecl::component::<physics::Collider>("collider"),
    PlaceDecl::component::<physics::RigidBody>("rigid_body"),
    PlaceDecl::component::<physics::BodyPush>("body_push"),
    PlaceDecl::ask::<physics::BodyAsk>(),
    PlaceDecl::component::<solids::SolidSource>("solid_source"),
    PlaceDecl::slot::<sweeps::Sweeps>(),
];
