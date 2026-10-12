//! The GPU queue and device calls the renderer makes through one function
//! each: Clippy refuses the wgpu methods anywhere else (`clippy.toml`'s
//! disallowed methods). Pipeline layouts and bind groups have theirs in
//! [`crate::pipelines`].

use std::cell::Cell;

/// The renderer's GPU queue, the one way it submits work: its frames' own,
/// and, counted, what modules submit at once.
pub(crate) struct Queue {
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
            queue,
            immediate: Cell::new(0),
        }
    }

    /// Submits a frame's own work, or work the host asks for between frames.
    pub(crate) fn submit(
        &self,
        buffers: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) -> wgpu::SubmissionIndex {
        submit(&self.queue, buffers)
    }

    /// Submits `buffers` at once, counted: growth copies and uploads that
    /// later writes depend on. The queue runs submissions in order, so work
    /// submitted while a frame records runs before every command of the
    /// frame's own encoder, which is submitted last.
    pub(crate) fn submit_now(&self, buffers: impl IntoIterator<Item = wgpu::CommandBuffer>) {
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
