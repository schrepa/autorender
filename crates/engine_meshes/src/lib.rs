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
mod shadow_changes;
mod smokes;
mod uploads;

use engine_core::{
    Attributes, Brick, Bricks, DrawKind, ImpostorPolicy, KindDecl, Runs, Schedule, Smoke, Source,
};
use engine_places::meshes::Instances;
use engine_render::RenderBricks;

pub use settings::{
    Culling, CullingStats, DetailStats, Impostors, MeshStats, OcclusionCulling, SubpixelCulling,
};

pub const MESH: DrawKind = DrawKind::MESH;

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
            .settings::<Impostors>()
            .stats::<MeshStats>()
            .resource::<order::MeshPlan>()
            .resource::<presentation::PresentationTotals>()
            .resource::<drawn::WatchedSlot>()
            .operation::<drawn::MeshDrawn>()
            .observation::<drawn::MeshesListing>()
            .explain::<explain::Culled>()
            .explain::<explain::Level>()
            .fill::<Instances>(Some(shaders::meshes::faces::instances::MODULE))
            .fill::<ImpostorPolicy>(None)
            .systems(
                Schedule::ViewSync,
                (
                    order::plan_draws,
                    assets::publish_assets,
                    shadow_changes::publish_shadow_changes,
                    assets::impostor_policy,
                    settings::mesh_stats,
                ),
            )
            .generated(GENERATED)
            .draw_kind(KindDecl {
                kind: MESH,
                vertex: shaders::meshes::vertex::VS_MESH.name(),
                needs: Attributes::BASE,
            })
            .draws::<gpu::MeshesGpu>(MESH, gpu::MeshesGpu::draw);
    }
}
