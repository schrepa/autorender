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
