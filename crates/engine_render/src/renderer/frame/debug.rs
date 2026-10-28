use super::*;
use crate::debug_view::{DebugViewFrame, DebugViewPass};

impl<'s> Features<'s> {
    pub(super) fn debug(
        gpu_prepared: Option<GpuPrepared>,
        shadow_frame: Option<ShadowFrame>,
        probe_list: &'s [engine_core::RenderReflectionProbe],
    ) -> Self {
        Self {
            gpu_prepared,
            shadow_frame,
            probe_list,
            frames: FeatureFrames::default(),
        }
    }
}

impl Renderer {
    pub(super) fn shows_debug_view(&self, snapshot: &RenderSnapshot) -> bool {
        let showing = snapshot.environment.debug_view.showing();
        #[cfg(feature = "dev")]
        let showing = self.display.probe.forces().unwrap_or(showing);
        showing && !self.features.debug_view.disabled()
    }

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

    pub(in crate::renderer) fn debug_records(
        &self,
        state: &FrameState<'_>,
    ) -> Option<&DebugViewPass> {
        (state.shape.targets == Targets::Debug)
            .then(|| self.features.debug_view.get())
            .flatten()
    }
}
