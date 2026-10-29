use super::*;
use crate::gpu_calls::{Handover, Recorded};
use crate::profiling::Span;

const BATCHES: [&str; 3] = ["batch_0", "batch_1", "batch_2"];

pub(super) fn batch(pass: Pass) -> u8 {
    let first_opaque = Pass::Opaque { phase: Phase::One }.index();
    let last_opaque = Pass::Opaque { phase: Phase::Two }.index();
    let index = pass.index();
    if index < first_opaque {
        0
    } else if index <= last_opaque {
        1
    } else {
        2
    }
}

pub(super) fn mark_between(
    encoder: &mut wgpu::CommandEncoder,
    end: Option<Span<'_>>,
    begin: Option<Span<'_>>,
) {
    if let Some(end) = end {
        end.mark(encoder, true);
    }
    if let Some(begin) = begin {
        begin.mark(encoder, false);
    }
}

impl Renderer {
    pub(super) fn hand_prelude(&self, prelude: Prelude) {
        let capturing = self.profiler.capturing();
        let mut handover = Handover::default();
        let mut open = None;
        for (span, part) in Prelude::SPANS.into_iter().zip(prelude.parts()) {
            let Some(part) = part else {
                continue;
            };
            let span = capturing.then_some(span);
            self.mark_prelude(&mut handover, open, span);
            handover.push(part);
            open = span;
        }
        self.mark_prelude(&mut handover, open, None);
        if !handover.is_empty() {
            self.queue.hand(handover);
        }
    }

    fn mark_prelude(
        &self,
        handover: &mut Handover,
        end: Option<&'static str>,
        begin: Option<&'static str>,
    ) {
        let span = |name: Option<&'static str>| self.profiler.busy(name?);
        let (end, begin) = (span(end), span(begin));
        if end.is_none() && begin.is_none() {
            return;
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/profile-mark"),
            });
        mark_between(&mut encoder, end, begin);
        handover.push(Recorded::Encoder(encoder));
    }

    pub(super) fn frame_encoder(&self, batch: u8) -> wgpu::CommandEncoder {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/frame-encoder"),
            });
        self.mark_batch(&mut encoder, None, Some(batch));
        encoder
    }

    fn mark_batch(&self, encoder: &mut wgpu::CommandEncoder, end: Option<u8>, begin: Option<u8>) {
        let span = |batch: Option<u8>| self.profiler.busy(BATCHES.get(usize::from(batch?))?);
        mark_between(encoder, span(end), span(begin));
    }

    pub(super) fn hand_batch(
        &self,
        mut encoder: wgpu::CommandEncoder,
        (ended, next): (u8, u8),
    ) -> wgpu::CommandEncoder {
        self.mark_batch(&mut encoder, Some(ended), None);
        self.queue.hand(Handover::from(encoder));
        self.frame_encoder(next)
    }

    pub(super) fn close_frame(&mut self, encoder: &mut wgpu::CommandEncoder, batch: u8) {
        self.mark_batch(encoder, Some(batch), None);
        self.profiler.end_frame(encoder);
    }

    pub(super) fn submit_frame(
        &mut self,
        state: &FrameState<'_>,
        frame: DisplayFrame,
        encoder: wgpu::CommandEncoder,
    ) {
        let FrameKind { gpu_mode, .. } = state.kind;

        let profile_submit = engine_profile::span("render", "submit_and_present");
        self.queue.hand(Handover::from(encoder));
        self.queue.settle();
        self.instances.pose_uploads.after_submit();
        if gpu_mode {
            self.gpu_presentation.after_submit();
        }
        self.shadows.after_submit();
        self.terrain.surface.after_submit();
        self.impostors.atlas.after_submit();
        self.impostors.placed.after_submit();
        self.water_settle.after_submit();
        self.features.after_submit(&state.features.frames);
        #[cfg(feature = "dev")]
        self.map_probe();
        self.profiler.submitted();
        frame.present(&self.queue);
        crate::platform::poll(&self.device);
        drop(profile_submit);
    }
}
