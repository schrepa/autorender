//! The GPU queue and device calls the renderer makes through one function
//! each: Clippy refuses the wgpu methods anywhere else (`clippy.toml`'s
//! disallowed methods). Pipeline layouts and bind groups have theirs in
//! [`crate::pipelines`]. The queue's submission side (`submission`)
//! finishes and submits the work frames hand over to it.

use std::cell::Cell;

mod submission;

pub(crate) use submission::{Handover, Recorded};

/// The renderer's GPU queue, the one way it submits work: its frames' own,
/// handed over to its submission side, what the host asks for between
/// frames, and, counted, what modules submit at once.
pub(crate) struct Queue {
    /// Where the work frames hand over is finished and submitted. Declared
    /// first, so its threads stop, and drop their handles to the queue,
    /// before this one drops: the queue's last handle then drops here, on
    /// the renderer's thread, before the surface and device it outlived.
    submission: submission::Submission,
    queue: wgpu::Queue,
    /// Submissions made at once since [`Self::take_immediate`].
    immediate: Cell<u32>,
}

impl std::ops::Deref for Queue {
    type Target = wgpu::Queue;

    fn deref(&self) -> &wgpu::Queue {
        &self.queue
    }
}

impl Queue {
    pub(crate) fn new(queue: wgpu::Queue) -> Self {
        Self {
            submission: submission::Submission::new(&queue),
            queue,
            immediate: Cell::new(0),
        }
    }

    /// Hands a frame's own work over to the submission side, which finishes
    /// and submits it after the work handed over before: natively on its
    /// threads while the frame records on, in browsers at once. A write
    /// queued after a handover takes effect at the next submission, which
    /// is the handed work's own while it is still being finished, as it was
    /// when a frame went in one submission: passes read every write queued
    /// before they were recorded, and rely on no later write of their frame
    /// staying out of them. Natively, the map callbacks the submission
    /// side's submissions completed run here, as they ran in submissions of
    /// the renderer's own.
    pub(crate) fn hand(&self, handover: Handover) {
        self.submission.hand(&self.queue, handover);
    }

    /// Waits until the submission side has submitted all the work handed
    /// over, as what reads those submissions requires: maps of what they
    /// copy for reading back, presentation, and submissions of the
    /// caller's own. The map callbacks those submissions completed run
    /// then.
    pub(crate) fn settle(&self) {
        self.submission.settle();
    }

    /// Submits work the native host asks for between frames, after the
    /// work handed over.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn submit(
        &self,
        buffers: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) -> wgpu::SubmissionIndex {
        self.settle();
        submit(&self.queue, buffers)
    }

    /// Submits `buffers` at once, counted: growth copies and uploads that
    /// later writes depend on. They follow the work the frame handed over
    /// before them, which this waits for, and run before what it hands
    /// over after.
    pub(crate) fn submit_now(&self, buffers: impl IntoIterator<Item = wgpu::CommandBuffer>) {
        self.settle();
        self.immediate.set(self.immediate.get() + 1);
        submit(&self.queue, buffers);
    }

    /// The submissions made at once since the last call.
    pub(crate) fn take_immediate(&self) -> u32 {
        self.immediate.take()
    }
}

/// Submits `buffers` to `queue`: the one way the renderer submits work.
#[expect(clippy::disallowed_methods, reason = "the renderer's one submission")]
fn submit(
    queue: &wgpu::Queue,
    buffers: impl IntoIterator<Item = wgpu::CommandBuffer>,
) -> wgpu::SubmissionIndex {
    queue.submit(buffers)
}

/// Opens a validation error scope on `device`, which catches the errors of
/// the device's calls until the returned guard pops: the one way the
/// renderer opens error scopes. Popping one is a future (in browsers, a
/// promise), so scopes cover creation, never a frame's work.
#[expect(clippy::disallowed_methods, reason = "the renderer's one error scope")]
pub(crate) fn validation_scope(device: &wgpu::Device) -> wgpu::ErrorScopeGuard {
    device.push_error_scope(wgpu::ErrorFilter::Validation)
}
