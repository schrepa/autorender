//! A frame's submissions: its prelude once its features are prepared, its
//! passes a batch at a time as they record, then its last batch, the
//! read-backs it asks for and its presentation.

use super::*;

/// The batch of the frame's passes `pass` records into, in render graph
/// order: the work before the opaque passes (GPU presentation, placed
/// objects and their occupancy, shadow casters and maps, lights, the
/// clouds' dome, tracks, the virtual texture's bakes, bounce light and
/// ground cover), then the opaque passes with the visibility buffer's,
/// deferred terrain's and occlusion culling's between them, then the rest.
/// Each batch goes to the GPU once recorded, which works on it while the
/// next records.
pub(super) fn batch(pass: Pass) -> u8 {
    let first_opaque = Pass::Opaque { phase: Phase::One }.index();
    let last_opaque = Pass::Resolve { phase: Phase::Two }.index();
    let index = pass.index();
    if index < first_opaque {
        0
    } else if index <= last_opaque {
        1
    } else {
        2
    }
}

impl Renderer {
    /// Submits `prelude`, which the frame's passes read, so the GPU works on
    /// it while they prepare and record; nothing when it holds no work.
    /// Profiling captures time each part (but the sky's, which times its
    /// own passes) between marks of its own: a part's mark, its work, its
    /// end's mark.
    pub(super) fn submit_prelude(&self, prelude: Prelude) {
        let capturing = self.profiler.capturing();
        let mark = |span: &'static str, end: bool| {
            let span = self.profiler.span(span)?;
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("engine/profile-mark"),
                });
            span.mark(&mut encoder, end);
            Some(encoder.finish())
        };
        let mut parts: [[Option<wgpu::CommandBuffer>; 3]; Prelude::SPANS.len()] =
            Default::default();
        for ((span, buffer), part) in Prelude::SPANS
            .into_iter()
            .zip(prelude.finish())
            .zip(&mut parts)
        {
            let span = span.filter(|_| capturing && buffer.is_some());
            let begin = span.and_then(|span| mark(span, false));
            let end = span.and_then(|span| mark(span, true));
            *part = [begin, buffer, end];
        }
        if parts.iter().flatten().any(Option::is_some) {
            let _profile = engine_profile::span("render", "submit_prelude");
            self.queue.submit(parts.into_iter().flatten().flatten());
        }
    }

    /// An encoder of the frame's own passes.
    pub(super) fn frame_encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/frame-encoder"),
            })
    }

    /// Submits the batch of passes `encoder` recorded; returns the encoder
    /// the frame records on into.
    pub(super) fn submit_batch(&self, encoder: wgpu::CommandEncoder) -> wgpu::CommandEncoder {
        let _profile = engine_profile::span("render", "submit_batch");
        self.queue.submit([encoder.finish()]);
        self.frame_encoder()
    }

    /// Submits the frame's last batch, `encoder`'s, after the batches before
    /// it; then the work the frame read back asks for its maps, and the
    /// frame presents.
    pub(super) fn submit_frame(
        &mut self,
        state: &FrameState<'_>,
        frame: DisplayFrame,
        encoder: wgpu::CommandEncoder,
    ) {
        let FrameKind { gpu_mode, .. } = state.kind;
        let virtual_frame = state.virtual_frame;

        let profile_submit = engine_profile::span("render", "submit_and_present");
        self.queue.submit([encoder.finish()]);
        self.instances.pose_uploads.after_submit();
        if gpu_mode {
            self.gpu_presentation.after_submit();
        }
        self.shadows.after_submit();
        self.terrain.vt.after_submit();
        // Impostors' records read back from bakes the frame submitted, and
        // settled water from its slots.
        self.impostors.atlas.after_submit();
        self.impostors.placed.after_submit();
        self.water_settle.after_submit();
        if virtual_frame {
            self.virtual_draws.pass.after_submit();
        }
        self.features.after_submit(&state.features.frames);
        #[cfg(feature = "dev")]
        self.map_probe();
        self.profiler.submitted();
        frame.present(&self.queue);
        // Native interfaces run finished work's callbacks, read-backs and
        // staging maps among them, when polled.
        crate::platform::poll(&self.device);
        drop(profile_submit);
    }
}
