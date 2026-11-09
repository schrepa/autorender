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

use bevy_ecs::prelude::Res;
use engine_core::{Attributes, Brick, Bricks, KindDecl, Schedule, Source};
use engine_places_world::ground::{Cover, FarCover};
use engine_render::RenderBricks;

pub use settings::{CoverLook, GroundCover, GroundCoverStats};

pub struct GroundCoverBrick;

impl Brick for GroundCoverBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|cover: Res<GroundCover>| cover.enabled)
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
            .draw_kind(KindDecl {
                kind: part::COVER,
                vertex: shaders::cover::draw::VS_COVER.name(),
                needs: Attributes::NONE,
                casts_shadows: false,
                seen_by_probes: false,
            })
            .draws::<part::CoverGpu>(part::COVER, part::CoverGpu::draw);
    }
}
