engine::brick!(level);

mod gpu;
mod numbers;
mod scene;

use engine::prelude::*;
use engine_core::{Attribute, Attributes, DrawKind, GpuStruct, KindDecl};

use crate::bricks::figures::LEVEL;
use crate::names::COLOR_PASS;
use crate::play::effects::GroundMarks;
use crate::play::light::Lighting;

pub use numbers::{
    AmbientNumbers, Antialias, BlobNumbers, FogNumbers, GradeNumbers, LightingNumbers, SkyNumbers,
};

pub const SKY: DrawKind = DrawKind("sky");

pub const LIGHTMAP_SCALE: &str = "lightmap_scale";

pub const LIGHTMAP_CURVE: &str = "lightmap_curve";

const GENERATED: &[&str] = &[<gpu::LevelLight as GpuStruct>::WGSL];

stats! {
    #[section("level")]
    pub struct LevelStats {
        pub refusal: Option<String>,
        pub lights: u32,
        pub draws: u32,
        pub triangles: u64,
        pub materials: u32,
        pub baked: u32,
        pub lightmap: bool,
        pub sky: bool,
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
            .stats::<LevelStats>()
            .resource::<Lighting>()
            .resource::<GroundMarks>()
            .resource::<scene::Setup>()
            .draw_kind(KindDecl {
                kind: LEVEL,
                needs: Attributes::BASE.with(Attribute::Uv1),
                streams: &[],
                passes: &[COLOR_PASS],
            })
            .draw_kind(KindDecl {
                kind: SKY,
                needs: Attributes::NONE,
                streams: &[],
                passes: &[COLOR_PASS],
            })
            .generated(GENERATED)
            .draws::<gpu::LevelGpu>(LEVEL, gpu::LevelGpu::draw_level)
            .draws::<gpu::LevelGpu>(SKY, gpu::LevelGpu::draw_sky)
            .systems(Schedule::Presentation, scene::setup)
            .systems(Schedule::Presentation, scene::light)
            .systems(Schedule::Presentation, scene::blobs)
            .systems(Schedule::Presentation, scene::count);
    }
}
