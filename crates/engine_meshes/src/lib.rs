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

use engine_core::{Attributes, Brick, Bricks, DrawKind, KindDecl, Schedule, Source};
use engine_places::meshes::Instances;
use engine_render::RenderBricks;

pub use settings::{CullingStats, DetailStats, MeshStats};

pub const MESH: DrawKind = DrawKind::MESH;
pub const SKIN: DrawKind = DrawKind::SKIN;

const GENERATED: &[&str] = &[
    layouts::WGSL,
    <layouts::CandidateState as engine_core::GpuBits>::WGSL,
    <layouts::LodHistory as engine_core::BitValue>::WGSL,
    <layouts::LodFade as engine_core::GpuBits>::WGSL,
    <layouts::PresentationFrame as engine_core::GpuStruct>::WGSL,
    <layouts::CandidateData as engine_core::GpuStruct>::WGSL,
    <layouts::BatchData as engine_core::GpuStruct>::WGSL,
    <layouts::BlendedView as engine_core::GpuStruct>::WGSL,
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
                skips: &[],
            })
            .draw_kind(KindDecl {
                kind: SKIN,
                needs: Attributes::SKINNED,
                skips: &[],
            })
            .draws::<gpu::MeshesGpu>(MESH, gpu::MeshesGpu::draw)
            .draws::<gpu::MeshesGpu>(SKIN, gpu::MeshesGpu::draw_skins);
    }
}
