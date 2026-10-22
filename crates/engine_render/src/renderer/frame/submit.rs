//! A frame's submissions: its prelude once its features are prepared and
//! its passes a batch at a time as they record, handed over to the queue's
//! submission side, which finishes and submits them in that order while
//! the frame records on; then, once all are submitted, the read-backs they
//! ask for and the frame's presentation.

use super::*;
use crate::gpu_calls::{Handover, Recorded};
use crate::profiling::Span;

/// The batch of the frame's passes `pass` records into, in render graph
/// order: the work before the opaque passes (GPU presentation, placed
/// objects and their occupancy, shadow casters and maps, lights, the
/// clouds' dome, tracks, the virtual texture's bakes, bounce light and
/// ground cover), then the opaque passes with the visibility buffer's,
/// deferred terrain's and occlusion culling's between them, then the rest.
/// Each batch is handed over once recorded, and the GPU works on it while
/// the next records.
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

/// Writes `end`'s end and `begin`'s begin, those there are, from one empty
/// compute pass, into `encoder`: between consecutive work profiling
/// captures time.
pub(super) fn mark_between(
    encoder: &mut wgpu::CommandEncoder,
    end: Option<Span<'_>>,
    begin: Option<Span<'_>>,
) {
    match (end, begin) {
        (Some(end), Some(begin)) => end.mark_until(begin, encoder),
        (Some(end), None) => end.mark(encoder, true),
        (None, Some(begin)) => begin.mark(encoder, false),
        (None, None) => {}
    }
}

impl Renderer {
    /// Hands `prelude` over, which the frame's passes read, so the GPU works
    /// on it while they prepare and record; nothing when it holds no work.
    /// Profiling captures time each part (but the sky's, which times its
    /// own passes) between marks of their own: a part's begin, its work,
    /// then its end with the next part's begin.
    pub(super) fn hand_prelude(&self, prelude: Prelude) {
        let capturing = self.profiler.capturing();
        let mut handover = Handover::default();
        // The span of the part handed over last, until its end is marked.
        let mut open = None;
        for (span, part) in Prelude::SPANS.into_iter().zip(prelude.parts()) {
            let Some(part) = part else {
                continue;
            };
            let span = span.filter(|_| capturing);
            self.mark_prelude(&mut handover, open, span);
            handover.push(part);
            open = span;
        }
        self.mark_prelude(&mut handover, open, None);
        if !handover.is_empty() {
            self.queue.hand(handover);
        }
    }

    /// Adds to `handover` an encoder of its own marking span `end`'s end and
    /// `begin`'s begin, those this frame records.
    fn mark_prelude(
        &self,
        handover: &mut Handover,
        end: Option<&'static str>,
        begin: Option<&'static str>,
    ) {
        let span = |name: Option<&'static str>| self.profiler.span(name?);
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

    /// An encoder of the frame's own passes.
    pub(super) fn frame_encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/frame-encoder"),
            })
    }

    /// Hands over the batch of passes `encoder` recorded, which the
    /// submission side finishes and submits while the frame records the
    /// next; returns the encoder the frame records on into.
    pub(super) fn hand_batch(&self, encoder: wgpu::CommandEncoder) -> wgpu::CommandEncoder {
        self.queue.hand(Handover::from(encoder));
        self.frame_encoder()
    }

    /// Hands over the frame's last batch, `encoder`'s, after the batches
    /// before it, and waits until the submission side has submitted them
    /// all; then the work the frame read back asks for its maps, and the
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
        self.queue.hand(Handover::from(encoder));
        // What follows maps what the frame's submissions copy for reading
        // back, and presents what they draw.
        self.queue.settle();
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
