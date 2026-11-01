include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod covers;
mod growth;
mod layouts;
mod part;
mod settings;

use engine_core::{Brick, Bricks, Schedule, Source};
use engine_places::ground::{Cover, FarCover};
use engine_render::{Layer, MainView, RenderBricks};

pub use settings::{CoverLook, GroundCover, GroundCoverStats};

pub struct GroundCoverBrick;

impl Brick for GroundCoverBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<GroundCover>()
            .stats::<GroundCoverStats>()
            .resource::<covers::CoverTable>()
            .resource::<growth::CoverPlan>()
            .resource::<growth::BladeCounts>()
            .fill::<Cover>(Some(shaders::cover::faces::cover::MODULE))
            .join::<FarCover>((), Some(shaders::cover::faces::far::MODULE))
            .systems(
                Schedule::ViewSync,
                (covers::write_covers, growth::plan_cover, growth::cover_stats),
            )
            .generated(layouts::GENERATED)
            .draws::<part::CoverGpu, MainView>(Layer::Late, part::CoverGpu::draw);
    }
}
