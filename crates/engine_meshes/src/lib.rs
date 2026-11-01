include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod batches;
mod casters;
mod draw;
mod encode;
mod gpu;
mod layouts;
mod order;
mod prepare;
mod presentation;
mod retained;
mod settings;
mod smokes;
mod uploads;

use engine_core::{Brick, Bricks, ComponentKind, Runs, Schedule, SceneComponents, Smoke, Source};
use engine_places::meshes::{Instances, Tint};
use engine_places::probes::ProbeViews;
use engine_places::shadows::ShadowCasters;
use engine_render::{Layer, MainView, RenderBricks};

pub use settings::{Culling, Impostors, MeshStats, OcclusionCulling, SubpixelCulling};

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
            .resource::<retained::RetainedRender>()
            .resource::<batches::BatchPlan>()
            .resource::<order::DrawOrder>()
            .resource::<order::MeshPlan>()
            .resource::<casters::CasterPlan>()
            .resource::<presentation::PresentationTotals>()
            .resource::<presentation::CasterTotals>()
            .fill::<Instances>(Some(shaders::meshes::faces::instances::MODULE))
            .join::<SceneComponents>(ComponentKind::of::<Tint>("tint"), None)
            .systems(
                Schedule::ViewSync,
                (
                    retained::retain_instances,
                    order::plan_draws,
                    casters::plan_casters,
                    settings::mesh_stats,
                ),
            )
            .draws::<gpu::MeshesGpu, MainView>(Layer::Occluders, gpu::MeshesGpu::draw_prepass)
            .draws::<gpu::MeshesGpu, MainView>(Layer::Opaque, gpu::MeshesGpu::draw_opaque)
            .draws::<gpu::MeshesGpu, MainView>(Layer::Backdrop, gpu::MeshesGpu::draw_second)
            .draws::<gpu::MeshesGpu, ShadowCasters>((), gpu::MeshesGpu::draw_casters)
            .draws::<gpu::MeshesGpu, ProbeViews>((), gpu::MeshesGpu::draw_probe_face);
    }
}
