engine::brick!(figures);

mod draw;
mod gpu;
mod marks;
mod reveal;

use engine_core::{
    Brick, Bricks, ComponentKind, GpuStruct, KindDecl, Pace, SceneComponents, Schedule, Source,
    stats,
};
use engine_render::RenderBricks;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::names::COLOR_PASS;
use crate::play::figures::{FIGURE, Forms, GROUND, REVEALED};
use crate::play::light::Light;
use crate::play::puppet::ShadeMaps;

pub use gpu::FigureLight;

const GENERATED: &[&str] = &[<FigureLight as GpuStruct>::WGSL, engine_meshpack::WGSL];

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FigureForms {
    pub entity: String,
    pub mid: bool,
    pub far: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FigureView {
    pub drawn: u32,
    pub out_of_view: u32,
    pub behind_map: u32,
    pub held: u32,
    pub full: u32,
    pub mid: u32,
    pub far: u32,
    pub vertices: u64,
    pub nearest_m: f32,
    pub farthest_m: f32,
}

stats! {
    #[section("figures")]
    pub struct FiguresStats {
        pub figures: u32,
        pub indices: u64,
        pub triangles: u64,
        pub lit: bool,
        pub revealed: u32,
        pub view: FigureView,
        pub maps: Option<ShadeMaps>,
        pub forms: Vec<FigureForms>,
    }
}

pub struct FiguresBrick;

impl Brick for FiguresBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<Forms>("figure_forms").schema(engine_core::schema_of::<Forms>),
            )
            .join::<SceneComponents>(
                ComponentKind::of::<Light>("light")
                    .schema(engine_core::schema_of::<Light>)
                    .check(Light::checked),
            )
            .resource::<draw::FigureForming>()
            .moves::<reveal::RevealOf>(Pace::Tick)
            .draw_kind(KindDecl {
                kind: FIGURE,
                passes: &[COLOR_PASS],
            })
            .draw_kind(KindDecl {
                kind: REVEALED,
                passes: &[COLOR_PASS],
            })
            .draw_kind(KindDecl {
                kind: GROUND,
                passes: &[COLOR_PASS],
            })
            .generated(GENERATED)
            .draws::<gpu::FiguresGpu>(FIGURE, gpu::FiguresGpu::draw_figures)
            .draws::<gpu::FiguresGpu>(REVEALED, gpu::FiguresGpu::draw_revealed)
            .draws::<gpu::FiguresGpu>(GROUND, gpu::FiguresGpu::draw_ground)
            .systems(Schedule::Presentation, draw::place)
            .systems(Schedule::Presentation, reveal::outline)
            .systems(Schedule::Presentation, marks::fold)
            .dev(|dev| {
                dev.stats::<FiguresStats>()
                    .systems(Schedule::Presentation, draw::count);
            });
    }
}
