#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod books;
mod cards;
mod meshes;
mod part;
mod settings;

use engine_core::{
    Attributes, Brick, Bricks, Cook, CookList, GpuBits, JobKind, KindDecl, KindSkip, PassRole,
    PassViews, Schedule, Source,
};
use engine_places_rendering::impostors::{ImpostorFlags, Impostors};
use engine_places_rendering::lighting::inputs;
use engine_render::{PointInput, RenderBricks};

pub use settings::{ImpostorSettings, ImpostorStats};

const GENERATED: &[&str] = &[
    <ImpostorFlags as GpuBits>::WGSL,
    <cards::CardFrames as GpuBits>::WGSL,
];

const SKIPS: &[KindSkip] = &[
    KindSkip::role(
        PassRole::Depth,
        PassViews::Owned,
        "impostor cards cast no shadows",
    ),
    KindSkip::role(
        PassRole::Lit,
        PassViews::Owned,
        "impostor cards stay out of probe captures",
    ),
];

const READS: [PointInput; 5] = [
    inputs::ROUGHNESS,
    inputs::METALNESS,
    inputs::OCCLUSION,
    inputs::DIRECT_SOFTNESS,
    inputs::DIRECT_BLOCKED,
];

const SETS: [PointInput; 7] = [
    inputs::THICKNESS,
    inputs::DIRECT_BLOCKED,
    inputs::TANGLED,
    inputs::ROUGHNESS,
    inputs::METALNESS,
    inputs::OCCLUSION,
    inputs::TRANSMISSION,
];

pub struct ImpostorsBrick;

impl Brick for ImpostorsBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&books::BAKE, &books::ASSEMBLY];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .within_switch::<ImpostorSettings>(|_| {})
            .stats::<ImpostorStats>()
            .resource::<books::ImpostorBooks>()
            .resource::<part::BakeReads>()
            .piece::<books::ImpostorPiece>()
            .fill::<Impostors>(Some(shaders::impostors::faces::impostors::MODULE))
            .join::<CookList>(Cook::new(books::cook), None)
            .systems(
                Schedule::ViewSync,
                (
                    meshes::ask_meshes,
                    books::assemble,
                    books::settle,
                    books::policy,
                    meshes::fill_levels,
                ),
            )
            .generated(GENERATED)
            .part::<part::ImpostorGpu>()
            .draw_kind(KindDecl {
                kind: cards::CARD,
                needs: Attributes::NONE,
                streams: &[],
                skips: SKIPS,
            })
            .draws::<cards::CardGpu>(cards::CARD, cards::CardGpu::draw);
        for input in READS {
            bricks.point_input(input);
        }
        for input in SETS {
            bricks.point_set(input.set());
        }
    }
}
