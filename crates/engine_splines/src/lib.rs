include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod scene;
mod smokes;
mod spline_mesh;
mod views;

use engine_core::{Brick, Bricks, ComponentKind, Runs, SceneComponents, Schedule, Smoke, Source};
use engine_places::splines::SplineMesh;

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
            .resource::<spline_mesh::SplineBuilds>()
            .join::<SceneComponents>(
                ComponentKind::of::<SplineMesh>("spline")
                    .check(scene::check_spline)
                    .reach(scene::spline_reach)
                    .schema(scene::spline_schema)
                    .example(scene::spline_example),
                None,
            )
            .systems(Schedule::WorldSync, spline_mesh::build)
            .on_remove::<SplineMesh, _>(spline_mesh::release)
            .observation::<views::InspectSplineMesh>();
    }
}
