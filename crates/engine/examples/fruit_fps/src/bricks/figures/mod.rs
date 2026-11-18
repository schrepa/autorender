engine::brick!(figures);

mod clock;
mod files;
mod gpu;
mod play;
mod reveal;

use engine_core::{
    AttributeFormat, Attributes, Brick, Bricks, ComponentKind, DrawKind, GpuStruct, KindDecl, Pace,
    SceneComponents, Schedule, Source, StreamShape, stats,
};
use engine_render::RenderBricks;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::names::COLOR_PASS;
use crate::play::figures::{FIGURE, Figure, FigureJoints, FigurePlay, GROUND, Held, REVEALED};
use crate::play::light::Light;
use crate::play::puppet::ShadeMaps;

pub use clock::FigureClocks;
pub use files::FigureFiles;
pub use gpu::FigureLight;
pub use play::{FigureDrawing, figure_tick};

pub const LEVEL: DrawKind = DrawKind("level");

pub const LIGHTMAP_MAP: &str = "lightmap";

pub const UV1: StreamShape = StreamShape {
    name: "uv1",
    format: AttributeFormat::Float32x2,
};

pub const JOINTS: StreamShape = StreamShape {
    name: "joints",
    format: AttributeFormat::Uint32x4,
};

pub const WEIGHTS: StreamShape = StreamShape {
    name: "weights",
    format: AttributeFormat::Unorm16x4,
};

pub const FIGURES_PART: &str = "figures";

const GENERATED: &[&str] = &[<FigureLight as GpuStruct>::WGSL];

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FigureHead {
    pub entity: String,
    pub time: f32,
    pub head: Option<[f32; 3]>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FigureForms {
    pub entity: String,
    pub named: bool,
    pub mid: bool,
    pub far: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FigureRefused {
    pub entity: String,
    pub part: String,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FigureView {
    pub posed: u32,
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
    pub unposed: u32,
}

stats! {
    #[section("figures")]
    pub struct FiguresStats {
        pub figures: u32,
        pub sampled: u32,
        pub ticked: u32,
        pub joints: u32,
        pub palette_bytes: u64,
        pub skeletons: u32,
        pub clips: u32,
        pub indices: u64,
        pub triangles: u64,
        pub lit: bool,
        pub revealed: u32,
        pub view: FigureView,
        pub maps: Option<ShadeMaps>,
        pub heads: Vec<FigureHead>,
        pub forms: Vec<FigureForms>,
        pub refused: Vec<FigureRefused>,
    }
}

pub struct FiguresBrick;

impl Brick for FiguresBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<Figure>("figure")
                    .schema(engine_core::schema_of::<Figure>)
                    .files(Figure::files)
                    .prepare(Figure::present),
            )
            .join::<SceneComponents>(
                ComponentKind::of::<FigurePlay>("figure_play")
                    .schema(engine_core::schema_of::<FigurePlay>)
                    .check(FigurePlay::checked),
            )
            .join::<SceneComponents>(
                ComponentKind::of::<FigureJoints>("figure_joints")
                    .schema(engine_core::schema_of::<FigureJoints>),
            )
            .join::<SceneComponents>(
                ComponentKind::of::<Held>("held")
                    .schema(engine_core::schema_of::<Held>)
                    .check(Held::checked)
                    .prepare(Held::resolved),
            )
            .join::<SceneComponents>(
                ComponentKind::of::<Light>("light")
                    .schema(engine_core::schema_of::<Light>)
                    .check(Light::checked),
            )
            .stats::<FiguresStats>()
            .resource::<FigureFiles>()
            .resource::<FigureClocks>()
            .resource::<FigureDrawing>()
            .moves::<Held>(Pace::Tick)
            .moves::<reveal::RevealOf>(Pace::Tick)
            .draw_kind(KindDecl {
                kind: FIGURE,
                needs: Attributes::BASE,
                streams: &[JOINTS, WEIGHTS],
                passes: &[COLOR_PASS],
            })
            .draw_kind(KindDecl {
                kind: REVEALED,
                needs: Attributes::BASE,
                streams: &[JOINTS, WEIGHTS],
                passes: &[COLOR_PASS],
            })
            .draw_kind(KindDecl {
                kind: GROUND,
                needs: Attributes::BASE,
                streams: &[],
                passes: &[COLOR_PASS],
            })
            .generated(GENERATED)
            .draws::<gpu::FiguresGpu>(FIGURE, gpu::FiguresGpu::draw_figures)
            .draws::<gpu::FiguresGpu>(REVEALED, gpu::FiguresGpu::draw_revealed)
            .draws::<gpu::FiguresGpu>(GROUND, gpu::FiguresGpu::draw_ground)
            .systems(Schedule::FixedSimulation, figure_tick)
            .systems(Schedule::Presentation, play::figure_draw)
            .systems(Schedule::Presentation, reveal::outline)
            .systems(Schedule::Presentation, play::count);
    }
}
