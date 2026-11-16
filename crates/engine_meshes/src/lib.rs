#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod assets;
mod debug;
mod draw;
mod drawn;
mod encode;
mod gpu;
mod layouts;
mod order;
mod prepare;
mod presentation;
mod settings;
mod uploads;

use engine_core::{Attributes, Brick, Bricks, KindDecl, Schedule, Source};
use engine_places::meshes::Instances;
use engine_render::RenderBricks;

pub use engine_places::meshes::{MESH, SKIN};
pub use settings::{CullingStats, DetailStats, MeshStats};

const GENERATED: &[&str] = &[
    layouts::WGSL,
    <layouts::CandidateState as engine_core::GpuBits>::WGSL,
    <layouts::PresentationFrame as engine_core::GpuStruct>::WGSL,
    <layouts::CandidateData as engine_core::GpuStruct>::WGSL,
    <layouts::BatchData as engine_core::GpuStruct>::WGSL,
];

pub use drawn::{MeshesListed, MeshesListing};

pub struct MeshesBrick;

impl Brick for MeshesBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .stats::<MeshStats>()
            .resource::<presentation::PresentationTotals>()
            .resource::<drawn::WatchedSlot>()
            .operation::<drawn::MeshDrawn>()
            .observation::<drawn::MeshesListing>()
            .fill::<Instances>(Some(shaders::meshes::faces::instances::MODULE))
            .systems(
                Schedule::ViewSync,
                (assets::publish_assets, settings::mesh_stats),
            )
            .generated(GENERATED)
            .draw_kind(KindDecl {
                kind: MESH,
                needs: Attributes::BASE,
                streams: &[],
                skips: &[],
            })
            .draw_kind(KindDecl {
                kind: SKIN,
                needs: Attributes::BASE,
                streams: engine_places::skin::STREAMS,
                skips: &[],
            })
            .draws::<gpu::MeshesGpu>(MESH, gpu::MeshesGpu::draw)
            .draws::<gpu::MeshesGpu>(SKIN, gpu::MeshesGpu::draw_skins);
    }
}
