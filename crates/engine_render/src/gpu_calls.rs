use std::cell::Cell;
use std::rc::Rc;

mod submission;

pub use submission::Handover;

#[derive(Clone)]
pub struct Queue(Rc<Shared>);

struct Shared {
    submission: submission::Submission,
    queue: wgpu::Queue,
    immediate: Cell<u32>,
}

impl std::ops::Deref for Queue {
    type Target = wgpu::Queue;

    fn deref(&self) -> &wgpu::Queue {
        &self.0.queue
    }
}

impl Queue {
    pub fn new(queue: wgpu::Queue) -> Self {
        Self(Rc::new(Shared {
            submission: submission::Submission::new(&queue),
            queue,
            immediate: Cell::new(0),
        }))
    }

    pub fn hand(&self, handover: Handover) {
        self.0.submission.hand(&self.0.queue, handover);
    }

    pub fn settle(&self) {
        self.0.submission.settle();
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn submit(
        &self,
        buffers: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) -> wgpu::SubmissionIndex {
        self.settle();
        submit(&self.0.queue, buffers)
    }

    pub fn submit_now(&self, buffers: impl IntoIterator<Item = wgpu::CommandBuffer>) {
        self.settle();
        self.0.immediate.set(self.0.immediate.get() + 1);
        submit(&self.0.queue, buffers);
    }

    pub fn take_immediate(&self) -> u32 {
        self.0.immediate.take()
    }

    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        crate::memory::uploaded_buffer(buffer, offset, data);
        write_buffer(&self.0.queue, buffer, offset, data);
    }

    pub fn write_texture(
        &self,
        texture: wgpu::TexelCopyTextureInfo<'_>,
        data: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        crate::memory::uploaded_texture(&texture, data);
        write_texture(&self.0.queue, texture, data, layout, size);
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
pub fn validation_scope(device: &wgpu::Device) -> wgpu::ErrorScopeGuard {
    device.push_error_scope(wgpu::ErrorFilter::Validation)
}
