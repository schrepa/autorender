use super::*;
use crate::gpu_calls::{Handover, Recorded};
use crate::profiling::Span;

const BATCHES: [&str; 3] = ["batch_0", "batch_1", "batch_2"];

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
    pub(super) fn hand_prelude(&mut self, prelude: Prelude) {
        #[cfg(feature = "profiling")]
        for &(span, owner) in prelude.owners() {
            match self
                .prelude_owners
                .iter_mut()
                .find(|(known, _)| *known == span)
            {
                Some(known) => known.1 = owner,
                None => self.prelude_owners.push((span, owner)),
            }
        }
        let capturing = self.core.profiler.capturing();
        let mut handover = Handover::default();
        let mut open = None;
        for (span, part) in prelude.take() {
            let span = capturing.then_some(span);
            if span.is_none() || span != open {
                self.mark_prelude(&mut handover, open, span);
            }
            handover.push(part);
            open = span;
        }
        self.mark_prelude(&mut handover, open, None);
        if !handover.is_empty() {
            self.core.queue.hand(handover);
        }
    }

    fn mark_prelude(
        &self,
        handover: &mut Handover,
        end: Option<&'static str>,
        begin: Option<&'static str>,
    ) {
        let span = |name: Option<&'static str>| self.core.profiler.busy(name?);
        let (end, begin) = (span(end), span(begin));
        if end.is_none() && begin.is_none() {
            return;
        }
        let mut encoder =
            self.core
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("engine/profile-mark"),
                });
        mark_between(&mut encoder, end, begin);
        handover.push(Recorded::Encoder(encoder));
    }

    pub(super) fn frame_encoder(&self, batch: u8) -> wgpu::CommandEncoder {
        let mut encoder =
            self.core
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("engine/frame-encoder"),
                });
        self.mark_batch(&mut encoder, None, Some(batch));
        encoder
    }

    fn mark_batch(&self, encoder: &mut wgpu::CommandEncoder, end: Option<u8>, begin: Option<u8>) {
        let span = |batch: Option<u8>| self.core.profiler.busy(BATCHES.get(usize::from(batch?))?);
        mark_between(encoder, span(end), span(begin));
    }

    pub(super) fn hand_batch(
        &self,
        mut encoder: wgpu::CommandEncoder,
        (ended, next): (u8, u8),
    ) -> wgpu::CommandEncoder {
        self.mark_batch(&mut encoder, Some(ended), None);
        self.core.queue.hand(Handover::from(encoder));
        self.frame_encoder(next)
    }

    pub(super) fn close_frame(&mut self, encoder: &mut wgpu::CommandEncoder, batch: u8) {
        self.mark_batch(encoder, Some(batch), None);
        self.core.profiler.end_frame(encoder);
    }

    pub(super) fn submit_frame(
        &mut self,
        kind: FrameKind,
        frame: DisplayFrame,
        encoder: wgpu::CommandEncoder,
    ) {
        let profile_submit = engine_profile::span("render", "submit_and_present");
        let core = &mut self.core;
        core.queue.hand(Handover::from(encoder));
        core.queue.settle();
        core.instances.pose_uploads.after_submit();
        if kind.gpu_mode {
            core.gpu_presentation.after_submit();
        }
        core.impostors.atlas.after_submit();
        #[cfg(feature = "dev")]
        self.map_probe();
        self.core.profiler.submitted();
        frame.present(&self.core.queue);
        crate::platform::poll(&self.core.device);
        drop(profile_submit);
    }
}
