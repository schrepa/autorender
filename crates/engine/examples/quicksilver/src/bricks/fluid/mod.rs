engine::brick!(fluid);

mod gpu;

use engine::prelude::*;
use engine_core::{DrawKind, KindDecl, Pass, PassName, PassViews, TargetName, Writes};

use crate::controls::{StudioCamera, attach_camera_lens, handle_input_and_camera};
use crate::state::QuicksilverState;

pub const FLUID_KIND: DrawKind = DrawKind("fluid");
pub const FLUID_SPLAT_PASS: &str = "fluid_splat";

pub const FLUID_DEPTH_0: TargetName = TargetName("fluid_depth_0");
pub const FLUID_DEPTH_1: TargetName = TargetName("fluid_depth_1");
pub const FLUID_DEPTH_2: TargetName = TargetName("fluid_depth_2");
pub const FLUID_DEPTH_3: TargetName = TargetName("fluid_depth_3");
pub const FLUID_DEPTH_4: TargetName = TargetName("fluid_depth_4");
pub const FLUID_AUX: TargetName = TargetName("fluid_aux");
pub const HW_DEPTH: TargetName = TargetName("fluid_hw_depth");
pub const FLUID_THICK_0: TargetName = TargetName("fluid_thick_0");
pub const FLUID_THICK_1: TargetName = TargetName("fluid_thick_1");
pub const FLUID_THICK_2: TargetName = TargetName("fluid_thick_2");
pub const HDR_COLOR: TargetName = TargetName("fluid_hdr");
pub const BLOOM_0: TargetName = TargetName("fluid_bloom_0");
pub const BLOOM_1: TargetName = TargetName("fluid_bloom_1");
pub const BLOOM_2: TargetName = TargetName("fluid_bloom_2");

pub struct FluidSplatPass;

impl PassName for FluidSplatPass {
    const NAME: &'static str = FLUID_SPLAT_PASS;
    const PASS: Pass = Pass {
        writes: Writes(&[FLUID_DEPTH_0, FLUID_AUX, HW_DEPTH]),
        views: PassViews::Main,
    };
}

pub struct FluidBrick;

impl Brick for FluidBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<StudioCamera>("studio_camera")
                    .schema(engine_core::schema_of::<StudioCamera>),
            )
            .moves::<StudioCamera>(Pace::Frame)
            .resource::<QuicksilverState>()
            .pass::<FluidSplatPass>()
            .draw_kind(KindDecl {
                kind: FLUID_KIND,
                passes: &[FLUID_SPLAT_PASS],
            })
            .draws::<gpu::FluidGpu>(FLUID_KIND, gpu::FluidGpu::draw_splat)
            .systems(Schedule::Presentation, attach_camera_lens)
            .systems(Schedule::Presentation, handle_input_and_camera);
    }
}
