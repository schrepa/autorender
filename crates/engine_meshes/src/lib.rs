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
mod explain;
mod gpu;
mod layouts;
mod order;
mod prepare;
mod presentation;
mod settings;
mod smokes;
mod uploads;

use engine_core::{Attributes, Brick, Bricks, DrawKind, KindDecl, Runs, Schedule, Smoke, Source};
use engine_places::meshes::Instances;
use engine_render::RenderBricks;

pub use settings::{
    Culling, CullingStats, DetailStats, MeshStats, OcclusionCulling, SubpixelCulling,
};

pub const MESH: DrawKind = DrawKind::MESH;
pub const SKIN: DrawKind = DrawKind::SKIN;

const GENERATED: &[&str] = &[
    layouts::WGSL,
    <layouts::CandidateState as engine_core::GpuBits>::WGSL,
    <layouts::LodHistory as engine_core::BitValue>::WGSL,
    <layouts::LodFade as engine_core::GpuBits>::WGSL,
    <layouts::CardFrames as engine_core::GpuBits>::WGSL,
    <layouts::PresentationFrame as engine_core::GpuStruct>::WGSL,
    <layouts::CandidateData as engine_core::GpuStruct>::WGSL,
    <layouts::BatchData as engine_core::GpuStruct>::WGSL,
    <layouts::BlendedView as engine_core::GpuStruct>::WGSL,
];

pub struct MeshesBrick;

impl Brick for MeshesBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[
        Smoke {
            name: "culling_settings",
            runs: Runs::Gpu,
            run: smokes::culling_settings,
        },
        Smoke {
            name: "meshes",
            runs: Runs::Gpu,
            run: smokes::meshes,
        },
    ];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<Culling>()
            .settings::<OcclusionCulling>()
            .settings::<SubpixelCulling>()
            .stats::<MeshStats>()
            .resource::<order::MeshPlan>()
            .resource::<presentation::PresentationTotals>()
            .resource::<drawn::WatchedSlot>()
            .operation::<drawn::MeshDrawn>()
            .observation::<drawn::MeshesListing>()
            .explain::<explain::Culled>()
            .explain::<explain::Level>()
            .fill::<Instances>(Some(shaders::meshes::faces::instances::MODULE))
            .systems(
                Schedule::ViewSync,
                (
                    order::plan_draws,
                    assets::publish_assets,
                    settings::mesh_stats,
                ),
            )
            .generated(GENERATED)
            .draw_kind(KindDecl {
                kind: MESH,
                vertex: shaders::meshes::vertex::VS_MESH.name(),
                needs: Attributes::BASE,
            })
            .draw_kind(KindDecl {
                kind: SKIN,
                vertex: shaders::meshes::vertex::VS_SKIN.name(),
                needs: Attributes::SKINNED,
            })
            .draws::<gpu::MeshesGpu>(MESH, gpu::MeshesGpu::draw)
            .draws::<gpu::MeshesGpu>(SKIN, gpu::MeshesGpu::draw_skins);
    }
}
