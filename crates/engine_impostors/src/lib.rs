#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod books;
mod part;
mod settings;

use engine_core::{Brick, Bricks, GpuBits, JobKind, Schedule, Source};
use engine_places::impostors::{ImpostorFlags, Impostors};
use engine_render::RenderBricks;

pub use settings::{ImpostorSettings, ImpostorStats};

const GENERATED: &[&str] = &[<ImpostorFlags as GpuBits>::WGSL];

pub struct ImpostorsBrick;

impl Brick for ImpostorsBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[&books::BAKE, &books::ASSEMBLY];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<ImpostorSettings>()
            .stats::<ImpostorStats>()
            .resource::<books::ImpostorBooks>()
            .resource::<part::BakeReads>()
            .fill::<Impostors>(Some(shaders::impostors::faces::impostors::MODULE))
            .systems(
                Schedule::ViewSync,
                (books::assemble, books::settle, books::policy),
            )
            .generated(GENERATED)
            .part::<part::ImpostorGpu>();
    }
}
