//! The GPU queue and device calls the renderer makes through one function
//! each: Clippy refuses the wgpu methods anywhere else (`clippy.toml`'s
//! disallowed methods). Pipeline layouts and bind groups have theirs in
//! [`crate::pipelines`].

/// Submits `buffers` to `queue`: the one way the renderer submits work. The
/// queue runs submissions in order, so work submitted while a frame records
/// runs before every command of the frame's own encoder, which is submitted
/// last.
#[expect(clippy::disallowed_methods, reason = "the renderer's one submission")]
pub(crate) fn submit(
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
