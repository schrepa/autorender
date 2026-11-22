#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

engine::brick!();

mod bridge;
mod gpu;
mod types;
mod world;

use engine::prelude::*;
use engine_core::{DrawKind, KindDecl, Pass, PassName, PassViews, Source, TargetName, Writes};

pub use bridge::NaniteBridgeHandle;
pub use types::{
    GpuBvhNode, GpuClusterHeader, HeapBudget, HizBuildInfo, LodPreset, MAX_HEAP_PAGES,
    NaniteCamera, NaniteConfig, NaniteInstance, NaniteTelemetry, NaniteUniforms, SHADOW_C0_DIM,
    SHADOW_C1_DIM, UNMAPPED_PAGE, ViewMode, VisibleCluster, extract_frustum_planes,
    reverse_z_infinite_proj,
};
pub use world::{MeshEntry, NaniteStreaming, NaniteWorld, PageFetchRequest, PageUpload};

pub const NANITE_DEPTH: TargetName = TargetName("nanite_depth");
pub const NANITE_VIS_HW: TargetName = TargetName("nanite_vis_hw");
pub const NANITE_DEPTH_P1: TargetName = TargetName("nanite_depth_p1");

pub const NANITE_P1_PASS: &str = "nanite_p1";
pub const NANITE_P2_PASS: &str = "nanite_p2";
pub const NANITE_KIND: DrawKind = DrawKind("nanite_world");

pub struct NaniteP1Pass;

impl PassName for NaniteP1Pass {
    const NAME: &'static str = NANITE_P1_PASS;
    const PASS: Pass = Pass {
        writes: Writes(&[NANITE_VIS_HW, NANITE_DEPTH_P1, NANITE_DEPTH]),
        views: PassViews::Main,
    };
}

pub struct NaniteP2Pass;

impl PassName for NaniteP2Pass {
    const NAME: &'static str = NANITE_P2_PASS;
    const PASS: Pass = Pass {
        writes: Writes(&[NANITE_VIS_HW, NANITE_DEPTH]),
        views: PassViews::Main,
    };
}

pub struct NaniteBrick;

impl Brick for NaniteBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .pass::<NaniteP1Pass>()
            .pass::<NaniteP2Pass>()
            .draw_kind(KindDecl {
                kind: NANITE_KIND,
                passes: &[NANITE_P1_PASS, NANITE_P2_PASS],
            })
            .generated(types::GENERATED)
            .resource::<NaniteWorld>()
            .resource::<NaniteCamera>()
            .resource::<NaniteConfig>()
            .resource::<NaniteTelemetry>()
            .resource::<NaniteStreaming>()
            .resource::<NaniteBridgeHandle>()
            .systems(Schedule::Presentation, bridge::pump_nanite_streaming)
            .draws::<gpu::NaniteGpu>(NANITE_KIND, gpu::NaniteGpu::draw_scene);
    }
}
