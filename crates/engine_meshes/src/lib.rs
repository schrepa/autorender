include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod assets;
mod batches;
mod casters;
mod debug;
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

use engine_core::{
    Brick, Bricks, ComponentKind, Error, ErrorCode, ImpostorPolicy, Rule, Runs, Schedule,
    SceneComponents, Setting, Smoke, Source,
};
use engine_places::meshes::{Instances, Tint};
use engine_places::probes::ProbeViews;
use engine_places::shadows::ShadowCasters;
use engine_render::{Layer, MainView, RenderBricks};
use serde_json::{Value, json};

pub use settings::{
    Culling, CullingStats, DetailStats, Impostors, MeshStats, OcclusionCulling, SubpixelCulling,
};

const GENERATED: &[&str] = &[
    layouts::WGSL,
    <layouts::PresentationFrame as engine_core::GpuStruct>::WGSL,
    <layouts::CandidateData as engine_core::GpuStruct>::WGSL,
    <layouts::BatchData as engine_core::GpuStruct>::WGSL,
    <layouts::CasterFrame as engine_core::GpuStruct>::WGSL,
];

const TINT_RULE: Rule = Rule::range(0.0, 1.0);

fn check_tint(tint: &Tint, path: &str) -> Result<(), Error> {
    Setting::check(&tint.0, "tint", TINT_RULE).map_err(|invalid| Error {
        code: ErrorCode::InvalidValue,
        path: format!(
            "{path}{}",
            invalid.field.strip_prefix("tint").unwrap_or_default()
        ),
        detail: Box::new(invalid.detail),
    })
}

fn tint_schema() -> Value {
    let mut schema = schemars::schema_for!([f32; 4]);
    engine_core::stated(TINT_RULE)(&mut schema);
    serde_json::to_value(schema).unwrap_or(Value::Null)
}

fn tint_example() -> Value {
    json!([1.0, 1.0, 1.0, 1.0])
}

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
            .fill::<ImpostorPolicy>(None)
            .join::<SceneComponents>(
                ComponentKind::of::<Tint>("tint")
                    .check(check_tint)
                    .schema(tint_schema)
                    .example(tint_example),
                None,
            )
            .systems(
                Schedule::ViewSync,
                (
                    retained::retain_instances,
                    order::plan_draws,
                    assets::publish_assets,
                    assets::impostor_policy,
                    casters::plan_casters,
                    settings::mesh_stats,
                ),
            )
            .generated(GENERATED)
            .draws::<gpu::MeshesGpu, MainView>(Layer::Occluders, gpu::MeshesGpu::draw_prepass)
            .draws::<gpu::MeshesGpu, MainView>(Layer::Opaque, gpu::MeshesGpu::draw_opaque)
            .draws::<gpu::MeshesGpu, MainView>(Layer::Backdrop, gpu::MeshesGpu::draw_second)
            .draws::<gpu::MeshesGpu, ShadowCasters>((), gpu::MeshesGpu::draw_casters)
            .draws::<gpu::MeshesGpu, ProbeViews>((), gpu::MeshesGpu::draw_probe_face);
    }
}
