#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod scene;
mod smokes;
mod spline_mesh;
mod views;

use engine_core::{
    Brick, Bricks, ComponentKind, ErrorCode, Runs, SceneComponents, Schedule, Smoke, Source,
};
use engine_places_world::splines::SplineMesh;
use engine_places_world::streaming::{StreamedKey, StreamedKeys, Streams};

pub const INVALID_SPLINE: ErrorCode = ErrorCode::brick("invalid_spline");

const SPLINE: &str = "spline";

pub struct SplinesBrick;

impl Brick for SplinesBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[Smoke {
        name: "spline",
        runs: Runs::Gpu,
        run: smokes::run,
    }];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .errors(&[INVALID_SPLINE])
            .resource::<spline_mesh::SplineBuilds>()
            .join::<SceneComponents>(
                ComponentKind::of::<SplineMesh>(SPLINE)
                    .check(scene::check_spline)
                    .schema(scene::spline_schema)
                    .example(scene::spline_example),
                None,
            )
            .join::<StreamedKeys>(
                StreamedKey {
                    key: SPLINE,
                    streams: Streams::Reaching(scene::spline_reach),
                },
                None,
            )
            .systems(Schedule::WorldSync, spline_mesh::build)
            .on_remove::<SplineMesh, _>(spline_mesh::release)
            .observation::<views::InspectSplineMesh>();
    }
}
