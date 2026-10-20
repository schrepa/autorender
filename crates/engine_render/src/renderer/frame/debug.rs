//! The debug views' part of a frame: their pass and targets, what the
//! features prepare for frames showing one, and the records their draws
//! read.

use super::*;
use crate::debug_view::{DebugViewFrame, DebugViewPass};

impl<'s> Features<'s> {
    /// What a frame showing a debug view prepares: no lights, screen
    /// effects, refraction, transparent surfaces or decals; the features
    /// keep what they made.
    pub(super) fn debug(
        occlusion: Option<(u32, u32, u32)>,
        gpu_prepared: Option<GpuPrepared>,
        shadow_frame: Option<ShadowFrame>,
        probe_list: &'s [engine_core::RenderReflectionProbe],
    ) -> Self {
        Self {
            occlusion,
            gpu_prepared,
            shadow_frame,
            probe_list,
            refraction: false,
            frames: FeatureFrames::default(),
        }
    }
}

impl Renderer {
    /// Whether the frame of `snapshot` draws into the debug views' targets:
    /// while its settings show a view, unless the feature is disabled; an
    /// armed pixel probe chooses for its frames.
    pub(super) fn shows_debug_view(&self, snapshot: &RenderSnapshot) -> bool {
        let showing = snapshot.environment.debug_view.showing();
        #[cfg(feature = "dev")]
        let showing = self.display.probe.forces().unwrap_or(showing);
        showing && !self.features.debug_view.disabled()
    }

    /// The debug view's pass and targets, in frames showing one.
    pub(super) fn prepare_debug_view(
        &mut self,
        snapshot: &RenderSnapshot,
        kind: FrameKind,
    ) -> Option<Ready<DebugViewFrame>> {
        let (device, queue) = (&self.device, &self.queue);
        let (size, format) = (self.display.size(), self.display.format());
        let sun = snapshot
            .environment
            .atmosphere
            .sun_direction
            .normalize_or_zero()
            .to_array();
        self.features.debug_view.prepare(
            (device, kind.debug),
            || DebugViewPass::new(device, format),
            |pass| pass.prepare(device, queue, size, snapshot.environment.debug_view, sun),
        )
    }

    /// The debug view's records, in frames showing one.
    pub(in crate::renderer) fn debug_records(
        &self,
        state: &FrameState<'_>,
    ) -> Option<&DebugViewPass> {
        (state.shape.targets == Targets::Debug)
            .then(|| self.features.debug_view.get())
            .flatten()
    }
}
