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
use engine_core::{Source, TargetName};

pub use bridge::NaniteBridgeHandle;
pub use types::{
    GpuBvhNode, GpuClusterHeader, HeapBudget, HizBuildInfo, LodPreset, MAX_HEAP_PAGES,
    NaniteCamera, NaniteConfig, NaniteInstance, NaniteTelemetry, NaniteUniforms, SHADOW_C0_DIM,
    SHADOW_C1_DIM, UNMAPPED_PAGE, ViewMode, VisibleCluster, extract_frustum_planes,
    reverse_z_infinite_proj, safe_heap_pages,
};
pub use world::{MeshEntry, NaniteStreaming, NaniteWorld, PageFetchRequest, PageUpload};

pub const NANITE_VIS_HW: TargetName = TargetName("nanite_vis_hw");

pub struct NaniteBrick;

impl Brick for NaniteBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .generated(types::GENERATED)
            .resource::<NaniteWorld>()
            .resource::<NaniteCamera>()
            .resource::<NaniteConfig>()
            .resource::<NaniteTelemetry>()
            .resource::<NaniteStreaming>()
            .resource::<NaniteBridgeHandle>()
            .systems(Schedule::Presentation, bridge::pump_nanite_streaming)
            .part::<gpu::NaniteGpu>();
    }
}
