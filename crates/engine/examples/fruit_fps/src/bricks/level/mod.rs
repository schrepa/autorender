engine::brick!(level);

mod chunks;
mod gpu;
mod numbers;
mod scene;
mod sets;

use engine::prelude::*;
use engine_core::{AssetKind, AssetKinds, DrawKind, GpuStruct, KindDecl};

use crate::names::COLOR_PASS;
use crate::play::effects::GroundMarks;
use crate::play::light::Lighting;

pub use numbers::{
    AmbientNumbers, BlobNumbers, FogNumbers, GradeNumbers, LightingNumbers, SkyNumbers,
};

pub const SKY: DrawKind = DrawKind("sky");

pub const LEVEL: DrawKind = DrawKind("level");

pub const LEVEL_DETAIL: DrawKind = DrawKind("level_detail");

pub const LIGHT_FIELD: DrawKind = DrawKind("light_field");

pub const FIELD_MAP: &str = "field";

pub const FIELD_RECT: [&str; 4] = ["min_x", "min_z", "max_x", "max_z"];

pub const FIELD_HEIGHT: &str = "height_m";

pub const MIRROR: &str = "mirror";

pub const LIGHTMAP_SCALE: &str = "lightmap_scale";

pub const LIGHTMAP_CURVE: &str = "lightmap_curve";

pub const LIGHTMAP_MAP: &str = "lightmap";

pub const MACRO_MAP: &str = "macro";

pub const MACRO_COLOR: &str = "macro_color";

pub const MACRO_ROUGH: &str = "macro_rough";

pub const MACRO_TILE: &str = "macro_tile_m";

pub const EMISSIVE_STRENGTH: &str = "emissive_strength";

pub const PICTURE_MAP: &str = "base_color";

pub const SURFACE_MAPS: [&str; 3] = ["base_color", "normal", "orm"];

pub const SKY_BELOW: &str = "below_deg";

pub const SKY_ABOVE: &str = "above_deg";

pub const SKY_FADE: &str = "fade_deg";

pub const SKY_HAZE: &str = "haze";

pub const SKY_TURNS: &str = "turns";

const GENERATED: &[&str] = &[
    <gpu::LevelLight as GpuStruct>::WGSL,
    <gpu::LevelSurface as GpuStruct>::WGSL,
    engine_meshpack::WGSL,
];

stats! {
    #[section("level")]
    pub struct LevelStats {
        pub refusal: Option<String>,
        pub lights: u32,
        pub draws: u32,
        pub chunks: u32,
        pub submitted: u32,
        pub skipped: u32,
        pub waiting: u32,
        pub vertices: u64,
        pub skipped_small: u32,
        pub triangles_submitted: u64,
        pub sets_loaded: bool,
        pub sets_unknown: u32,
        pub sets: bool,
        pub set_cell: Option<[u32; 2]>,
        pub kept_by_frustum: u32,
        pub kept_by_set: u32,
        pub triangles: u64,
        pub materials: u32,
        pub baked: u32,
        pub detail: u32,
        pub cutout: u32,
        pub glowing: u32,
        pub macro_strengths: u32,
        pub problems: Vec<String>,
        pub lightmap: bool,
        pub sky: bool,
        pub picture: bool,
        pub plain: u32,
        pub mirrors: u32,
        pub field: bool,
        pub field_refusal: Option<String>,
        pub zenith: [f32; 3],
        pub numbers: Option<LightingNumbers>,
        pub blobs: u32,
    }
}

pub struct LevelBrick;

impl Brick for LevelBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .resource::<Lighting>()
            .resource::<GroundMarks>()
            .resource::<scene::Setup>()
            .resource::<chunks::Pieces>()
            .resource::<sets::SetIndex>()
            .join::<SceneComponents>(
                ComponentKind::of::<sets::LevelSets>("level_sets")
                    .schema(engine_core::schema_of::<sets::LevelSets>),
            )
            .join::<AssetKinds>(AssetKind::of("level_sets", sets::parsed))
            .join::<SceneComponents>(
                ComponentKind::of::<chunks::LevelPiece>("level_piece")
                    .schema(engine_core::schema_of::<chunks::LevelPiece>),
            )
            .draw_kind(KindDecl {
                kind: LEVEL,
                passes: &[COLOR_PASS],
            })
            .draw_kind(KindDecl {
                kind: LEVEL_DETAIL,
                passes: &[COLOR_PASS],
            })
            .draw_kind(KindDecl {
                kind: LIGHT_FIELD,
                passes: &[COLOR_PASS],
            })
            .draw_kind(KindDecl {
                kind: SKY,
                passes: &[COLOR_PASS],
            })
            .generated(GENERATED)
            .draws::<gpu::LevelGpu>(LEVEL, gpu::LevelGpu::draw_level)
            .draws::<gpu::LevelGpu>(LEVEL_DETAIL, gpu::LevelGpu::draw_detail)
            .draws::<gpu::LevelGpu>(SKY, gpu::LevelGpu::draw_sky)
            .draws::<gpu::LevelGpu>(LIGHT_FIELD, gpu::LevelGpu::draw_field)
            .systems(Schedule::Presentation, scene::setup)
            .systems(Schedule::Presentation, scene::light)
            .systems(Schedule::Presentation, scene::blobs)
            .systems(Schedule::Presentation, chunks::pieces)
            .systems(Schedule::Presentation, sets::adopt)
            .dev(|dev| {
                dev.stats::<LevelStats>()
                    .systems(Schedule::Presentation, scene::count);
            });
    }
}
