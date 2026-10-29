use std::cell::Cell;

mod submission;

pub(crate) use submission::{Handover, Recorded};

pub(crate) struct Queue {
    submission: submission::Submission,
    queue: wgpu::Queue,
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

    pub(crate) fn hand(&self, handover: Handover) {
        self.submission.hand(&self.queue, handover);
    }

    pub(crate) fn settle(&self) {
        self.submission.settle();
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn submit(
        &self,
        buffers: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) -> wgpu::SubmissionIndex {
        self.settle();
        submit(&self.queue, buffers)
    }

    pub(crate) fn submit_now(&self, buffers: impl IntoIterator<Item = wgpu::CommandBuffer>) {
        self.settle();
        self.immediate.set(self.immediate.get() + 1);
        submit(&self.queue, buffers);
    }

    pub(crate) fn take_immediate(&self) -> u32 {
        self.immediate.take()
    }

    pub(crate) fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        crate::memory::uploaded_buffer(buffer, offset, data);
        write_buffer(&self.queue, buffer, offset, data);
    }

    pub(crate) fn write_texture(
        &self,
        texture: wgpu::TexelCopyTextureInfo<'_>,
        data: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        crate::memory::uploaded_texture(&texture, data);
        write_texture(&self.queue, texture, data, layout, size);
    }
}

#[expect(
    clippy::disallowed_methods,
    reason = "the renderer's one buffer upload, which records repeated uploads"
)]
fn write_buffer(queue: &wgpu::Queue, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
    queue.write_buffer(buffer, offset, data);
}

#[expect(
    clippy::disallowed_methods,
    reason = "the renderer's one texture upload, which records repeated uploads"
)]
fn write_texture(
    queue: &wgpu::Queue,
    texture: wgpu::TexelCopyTextureInfo<'_>,
    data: &[u8],
    layout: wgpu::TexelCopyBufferLayout,
    size: wgpu::Extent3d,
) {
    queue.write_texture(texture, data, layout, size);
}

#[expect(clippy::disallowed_methods, reason = "the renderer's one submission")]
fn submit(
    queue: &wgpu::Queue,
    buffers: impl IntoIterator<Item = wgpu::CommandBuffer>,
) -> wgpu::SubmissionIndex {
    queue.submit(buffers)
}

#[expect(clippy::disallowed_methods, reason = "the renderer's one error scope")]
pub(crate) fn validation_scope(device: &wgpu::Device) -> wgpu::ErrorScopeGuard {
    device.push_error_scope(wgpu::ErrorFilter::Validation)
}
