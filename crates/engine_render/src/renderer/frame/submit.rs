//! A frame's submission after the work it reads, and its presentation.

use super::*;

impl Renderer {
    /// Submits the frame after the work it reads (asset levels, staged pose
    /// and input copies, environment maps) and presents it.
    pub(super) fn submit_frame(
        &mut self,
        state: &mut FrameState<'_>,
        frame: DisplayFrame,
        encoder: wgpu::CommandEncoder,
    ) {
        let FrameKind { gpu_mode, .. } = state.kind;
        let virtual_frame = state.virtual_frame;
        let sky_encoder = state.features.sky_encoder.take();
        let mip_encoder = state.assets.encoder.take();
        let staged_poses = state.writes.staged_poses.take();
        let input_copies = state.features.input_copies.take();
        let environment_encoder = state.features.environment_encoder.take();
        let temporal_frame = &state.packing.temporal_frame;

        let profile_submit = engine_profile::span("render", "submit_and_present");
        // Profiling captures time each of the command buffers before the
        // frame's between marks of their own.
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
        let timed = |span: &'static str, buffer: Option<wgpu::CommandBuffer>| {
            let marked = buffer.is_some() && self.profiler.capturing();
            let begin = marked.then(|| mark(span, false)).flatten();
            let end = marked.then(|| mark(span, true)).flatten();
            begin.into_iter().chain(buffer).chain(end)
        };
        // The sky's light precedes everything that reads it; staged
        // pose/input copies precede the passes that read them.
        let before: Vec<wgpu::CommandBuffer> = sky_encoder
            .map(wgpu::CommandEncoder::finish)
            .into_iter()
            .chain(timed(
                "pre_mips",
                mip_encoder.map(wgpu::CommandEncoder::finish),
            ))
            .chain(timed("pre_poses", staged_poses))
            .chain(timed(
                "pre_inputs",
                input_copies.map(wgpu::CommandEncoder::finish),
            ))
            .chain(timed(
                "pre_environment",
                environment_encoder.map(wgpu::CommandEncoder::finish),
            ))
            .collect();
        self.queue
            .submit(before.into_iter().chain([encoder.finish()]));
        self.instances.pose_uploads.after_submit();
        if gpu_mode {
            self.gpu_presentation.after_submit();
        }
        self.shadows.after_submit();
        self.ground_cover.after_submit();
        self.terrain.vt.after_submit();
        if let Some(lights) = self.lights.as_mut() {
            lights.after_submit();
        }
        if virtual_frame {
            self.virtual_draws.pass.after_submit();
        }
        if let (Some(frame), Some(pass)) = (&temporal_frame, &mut self.temporal) {
            pass.submitted(frame);
        }
        self.profiler.submitted();
        frame.present(&self.queue);
        // Native interfaces run finished work's callbacks, read-backs and
        // staging maps among them, when polled.
        crate::platform::poll(&self.device);
        drop(profile_submit);
    }
}
