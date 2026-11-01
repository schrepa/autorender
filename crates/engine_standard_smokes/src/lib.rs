include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod agent_lifecycle;
mod debug_view_smoke;
mod frame_loop;
mod hierarchy;
mod host_parity;
mod navigation;
pub mod requests;
mod scene_kit;
mod scene_views;
mod stats;
mod streamed_scene;
mod surfaces;
mod terrain;
mod terrain_views;
mod texture;

use engine_core::{Brick, Bricks, Runs, Smoke, Source};

pub struct StandardSmokesBrick;

impl Brick for StandardSmokesBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[
        Smoke {
            name: "frame_loop",
            runs: Runs::Gpu,
            run: frame_loop::frame_loop,
        },
        Smoke {
            name: "host_parity",
            runs: Runs::Gpu,
            run: host_parity::host_parity,
        },
        Smoke {
            name: "host_images",
            runs: Runs::Gpu,
            run: host_parity::host_images,
        },
        Smoke {
            name: "host_faults",
            runs: Runs::Gpu,
            run: host_parity::host_faults,
        },
        Smoke {
            name: "agent_lifecycle",
            runs: Runs::Gpu,
            run: agent_lifecycle::agent_lifecycle,
        },
        Smoke {
            name: "hierarchy",
            runs: Runs::Gpu,
            run: hierarchy::run,
        },
        Smoke {
            name: "surfaces",
            runs: Runs::Gpu,
            run: surfaces::run,
        },
        Smoke {
            name: "texture",
            runs: Runs::Gpu,
            run: texture::run,
        },
        Smoke {
            name: "streamed_scene",
            runs: Runs::Gpu,
            run: streamed_scene::run,
        },
        Smoke {
            name: "scene_views",
            runs: Runs::Headless,
            run: scene_views::run,
        },
        Smoke {
            name: "debug_view_terrain",
            runs: Runs::Gpu,
            run: terrain_views::run,
        },
        Smoke {
            name: "debug_view",
            runs: Runs::Gpu,
            run: debug_view_smoke::run,
        },
        Smoke {
            name: "terrain_draws",
            runs: Runs::Gpu,
            run: terrain::draws,
        },
        Smoke {
            name: "terrain_shapes",
            runs: Runs::Gpu,
            run: terrain::shapes,
        },
        Smoke {
            name: "terrain_surface",
            runs: Runs::Gpu,
            run: terrain::surface,
        },
        Smoke {
            name: "terrain_water_sky",
            runs: Runs::Gpu,
            run: terrain::water_sky,
        },
        Smoke {
            name: "navigation_walks",
            runs: Runs::Gpu,
            run: navigation::walks,
        },
    ];

    fn register(_: &mut Bricks<'_>) {}
}
