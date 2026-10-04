//! The color targets of the frame's main passes (decision 050's
//! [`Targets`]): the formats every pipeline drawing into them declares, in
//! pipeline order.

use crate::post::{GLOW_FORMAT, SCENE_FORMAT, SURFACE_FORMAT};
pub(crate) use crate::render_graph::Targets;
use crate::temporal::MOTION_FORMAT;

impl Targets {
    /// The color target formats in pipeline order: the canvas's `canvas`
    /// format; the screen effects' scene, glow and surface targets
    /// (decision 021); temporal frames' motion target after the scene
    /// (decision 017).
    pub(crate) fn formats(self, canvas: wgpu::TextureFormat) -> Vec<wgpu::TextureFormat> {
        match self {
            Self::Canvas => vec![canvas],
            Self::Post => vec![SCENE_FORMAT, GLOW_FORMAT, SURFACE_FORMAT],
            Self::Temporal => vec![SCENE_FORMAT, MOTION_FORMAT, GLOW_FORMAT, SURFACE_FORMAT],
        }
    }

    /// Opaque, unblended color target states of [`Self::formats`].
    pub(crate) fn color_targets(
        self,
        canvas: wgpu::TextureFormat,
    ) -> Vec<Option<wgpu::ColorTargetState>> {
        self.formats(canvas)
            .into_iter()
            .map(|format| Some(format.into()))
            .collect()
    }
}
