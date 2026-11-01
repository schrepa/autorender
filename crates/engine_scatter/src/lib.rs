#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod far_cover;
mod gpu;
mod groups;
mod layouts;
mod occupancy;
mod pool;
mod settings;
mod smokes;
mod systems;

use engine_core::{Brick, Bricks, Runs, Schedule, Smoke, Source};
use engine_places::ground::{FarCover, Foliage, Occupancy, PlacedTiles};
use engine_places::shadows::ShadowCasters;
use engine_render::{Layer, MainView, RenderBricks};

pub use settings::{
    FoliageSettings, OccupancySettings, PlacedLod, PlacedLook, PlacedPart, ScatterSettings,
    ScatterStats,
};

pub struct ScatterBrick;

impl Brick for ScatterBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[
        Smoke {
            name: "foliage_settings",
            runs: Runs::Gpu,
            run: smokes::foliage_settings,
        },
        Smoke {
            name: "slot_scatter_covers",
            runs: Runs::Gpu,
            run: smokes::scatter_covers,
        },
    ];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<ScatterSettings>()
            .settings::<FoliageSettings>()
            .settings::<OccupancySettings>()
            .stats::<ScatterStats>()
            .resource::<systems::ScatterState>()
            .resource::<systems::ScatterFeedback>()
            .resource::<occupancy::OccupancyWindows>()
            .resource::<far_cover::FarCoverTable>()
            .piece::<layouts::CanopyPiece>()
            .generated(layouts::GENERATED)
            .fill::<Foliage>(Some(shaders::scatter::faces::foliage::MODULE))
            .fill::<PlacedTiles>(Some(shaders::scatter::faces::placed_tiles::MODULE))
            .fill::<Occupancy>(Some(shaders::scatter::faces::occupancy::MODULE))
            .join::<FarCover>((), Some(shaders::scatter::faces::far_cover::MODULE))
            .systems(
                Schedule::ViewSync,
                (
                    systems::publish_foliage,
                    systems::take_impostors,
                    systems::place_windows,
                    systems::place_pool,
                    systems::publish_assets,
                    systems::plan_cull,
                    systems::mark_moved,
                    systems::publish_tiles,
                    systems::publish_changes,
                    systems::scatter_stats,
                ),
            )
            .draws::<gpu::ScatterGpu, MainView>(Layer::Occluders, gpu::ScatterGpu::draw_occluders)
            .draws::<gpu::ScatterGpu, MainView>(Layer::Late, gpu::ScatterGpu::draw_late)
            .draws::<gpu::ScatterGpu, ShadowCasters>((), gpu::ScatterGpu::draw_casters);
    }
}
