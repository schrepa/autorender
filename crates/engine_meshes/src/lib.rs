include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod assets;
mod batches;
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
mod retained;
mod settings;
mod smokes;
mod uploads;

use engine_core::{
    Attributes, Brick, Bricks, ComponentKind, DrawKind, Error, ErrorCode, ImpostorPolicy,
    KindDecl, Mesh3d, Rule, Runs, SceneComponents, Schedule, Setting, Smoke, Source,
};
use engine_places::meshes::{Cube, FirstPersonLayer, Instances, Tint};
use engine_render::RenderBricks;
use serde_json::{Value, json};

pub use settings::{
    Culling, CullingStats, DetailStats, Impostors, MeshStats, OcclusionCulling, SubpixelCulling,
};

pub const MESH: DrawKind = DrawKind("mesh");

const GENERATED: &[&str] = &[
    layouts::WGSL,
    <layouts::CandidateState as engine_core::GpuBits>::WGSL,
    <layouts::LodHistory as engine_core::BitValue>::WGSL,
    <layouts::LodFade as engine_core::GpuBits>::WGSL,
    <layouts::CardFrames as engine_core::GpuBits>::WGSL,
    <engine_core::Mobility as engine_core::BitValue>::WGSL,
    <engine_core::NodeWord as engine_core::GpuBits>::WGSL,
    <layouts::PresentationFrame as engine_core::GpuStruct>::WGSL,
    <layouts::CandidateData as engine_core::GpuStruct>::WGSL,
    <layouts::BatchData as engine_core::GpuStruct>::WGSL,
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
            .resource::<order::MeshPlan>()
            .resource::<presentation::PresentationTotals>()
            .resource::<drawn::WatchedSlot>()
            .operation::<drawn::MeshDrawn>()
            .observation::<drawn::MeshesListing>()
            .explain::<explain::Culled>()
            .explain::<explain::Level>()
            .fill::<Instances>(Some(shaders::meshes::faces::instances::MODULE))
            .fill::<ImpostorPolicy>(None)
            .join::<SceneComponents>(
                ComponentKind::of::<Tint>("tint")
                    .check(check_tint)
                    .schema(tint_schema)
                    .example(tint_example),
                None,
            )
            .on_remove::<Mesh3d, _>(retained::source_removed)
            .on_remove::<Cube, _>(retained::source_removed)
            .on_remove::<Tint, _>(retained::source_removed)
            .on_remove::<FirstPersonLayer, _>(retained::source_removed)
            .systems(
                Schedule::ViewSync,
                (
                    retained::retain_instances,
                    order::plan_draws,
                    assets::publish_assets,
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
