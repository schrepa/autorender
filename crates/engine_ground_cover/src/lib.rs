#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod covers;
mod growth;
mod layouts;
mod part;
mod settings;
mod surface;

use engine_core::{
    Attributes, Brick, Bricks, KindDecl, KindSkip, PassRole, PassViews, Schedule, Source,
};
use engine_open_world_kit::ground::{Cover, FarCover};
use engine_render::RenderBricks;

pub use settings::{CoverLook, GroundCover, GroundCoverStats};

const SKIPS: &[KindSkip] = &[KindSkip::role(
    PassRole::Lit,
    PassViews::Owned,
    "ground cover grows around the main camera only",
)];

pub struct GroundCoverBrick;

impl Brick for GroundCoverBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .switch::<GroundCover>()
            .stats::<GroundCoverStats>()
            .resource::<covers::CoverTable>()
            .resource::<growth::CoverPlan>()
            .resource::<growth::BladeCounts>()
            .resource::<surface::CoverSurface>()
            .fill::<Cover>(Some(shaders::cover::faces::cover::MODULE))
            .join::<FarCover>((), Some(shaders::cover::faces::far::MODULE))
            .systems(
                Schedule::ViewSync,
                (
                    covers::write_covers,
                    growth::plan_cover,
                    growth::cover_stats,
                    surface::hold_surface,
                ),
            )
            .generated(layouts::GENERATED)
            .draw_kind(KindDecl {
                kind: part::COVER,
                needs: Attributes::NONE,
                streams: &[],
                skips: SKIPS,
            })
            .draws::<part::CoverGpu>(part::COVER, part::CoverGpu::draw);
        for module in surface::slot_modules() {
            bricks.material_slot(module);
        }
    }
}
