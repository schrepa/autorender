//! The color targets of the frame's main passes (decision 050's
//! [`Targets`]): the formats every pipeline drawing into them declares, in
//! pipeline order, and the constants their fragment stages take.

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

    /// Pipeline constants of fragment stages drawing into these targets:
    /// whether the canvas encodes sRGB itself.
    pub(crate) fn constants(self, canvas: wgpu::TextureFormat) -> Vec<(&'static str, f64)> {
        match self {
            Self::Canvas => vec![("SURFACE_IS_SRGB", if canvas.is_srgb() { 1.0 } else { 0.0 })],
            Self::Post | Self::Temporal => Vec::new(),
        }
    }

    /// The suffix of fragment entry points drawing into these targets (the
    /// sky's, impostors' and the visibility buffer's resolves): none for the
    /// canvas.
    pub(crate) fn entry_suffix(self) -> &'static str {
        match self {
            Self::Canvas => "",
            Self::Post => "_post",
            Self::Temporal => "_temporal",
        }
    }
}
