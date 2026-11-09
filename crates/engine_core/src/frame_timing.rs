use bevy_ecs::resource::Resource;

use crate::Slot;

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameTiming {
    pub gpu_timed: bool,
    pub gpu: Option<GpuFrameTime>,
    pub render_size: Option<(u32, u32)>,
    pub interval_ms: Option<f64>,
    pub refresh_ms: Option<f64>,
    pub suspends: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuFrameTime {
    pub ms: f64,
    pub render_size: (u32, u32),
}

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct RenderScale(pub f64);

impl Default for RenderScale {
    fn default() -> Self {
        Self(1.0)
    }
}

impl Slot for RenderScale {
    const NAME: &'static str = "render_scale";
}
