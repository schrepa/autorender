include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod debug_view_smoke;
mod hierarchy;
mod host_parity;
mod requests;
mod scene_kit;
mod scene_views;
mod streamed_scene;
mod surfaces;
mod terrain_views;
mod texture;

use engine_core::{Brick, Bricks, Runs, Smoke, Source};

pub struct StandardSmokesBrick;

impl Brick for StandardSmokesBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[
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
    ];

    fn register(_: &mut Bricks<'_>) {}
}
