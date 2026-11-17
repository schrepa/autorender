#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

extern crate engine_ecs as bevy_ecs;

pub mod splines;
pub mod streaming;

use engine_place::PlaceDecl;

pub const PLACES: &[PlaceDecl] = &[
    PlaceDecl::component::<splines::SplineMesh>("spline_mesh"),
    PlaceDecl::component::<streaming::StreamingFocus>("streaming_focus"),
    PlaceDecl::list::<streaming::StreamedKeys>(),
];
