//! A frame's submission after the work it reads, and its presentation.

use super::*;

impl Renderer {
    /// Submits the frame after its prelude, the work it reads, and presents
    /// it.
    pub(super) fn submit_frame(
        &mut self,
        state: &mut FrameState<'_>,
        frame: DisplayFrame,
        encoder: wgpu::CommandEncoder,
    ) {
        let FrameKind { gpu_mode, .. } = state.kind;
        let virtual_frame = state.virtual_frame;
        let prelude = std::mem::take(&mut state.prelude);

        let profile_submit = engine_profile::span("render", "submit_and_present");
        // Profiling captures time each part of the prelude between marks of
        // its own: a part's mark, its work, its end's mark.
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
        let mut before: [[Option<wgpu::CommandBuffer>; 3]; Prelude::SPANS.len()] =
            Default::default();
        for ((span, buffer), part) in Prelude::SPANS
            .into_iter()
            .zip(prelude.finish())
            .zip(&mut before)
        {
            let span = span.filter(|_| capturing && buffer.is_some());
            let begin = span.and_then(|span| mark(span, false));
            let end = span.and_then(|span| mark(span, true));
            *part = [begin, buffer, end];
        }
        crate::gpu_calls::submit(
            &self.queue,
            before
                .into_iter()
                .flatten()
                .flatten()
                .chain([encoder.finish()]),
        );
        self.instances.pose_uploads.after_submit();
        if gpu_mode {
            self.gpu_presentation.after_submit();
        }
        self.shadows.after_submit();
        self.terrain.vt.after_submit();
        if virtual_frame {
            self.virtual_draws.pass.after_submit();
        }
        self.features.after_submit(&state.features.frames);
        self.profiler.submitted();
        frame.present(&self.queue);
        // Native interfaces run finished work's callbacks, read-backs and
        // staging maps among them, when polled.
        crate::platform::poll(&self.device);
        drop(profile_submit);
    }
}
