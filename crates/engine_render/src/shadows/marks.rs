//! The marks of the clipmap's tiles the frame's pixels read: a compute
//! pass over the depth buffer after the opaque passes
//! (`shadow_marks.wgsl`), its mask of every level's cells read back a few
//! frames later, which gives those tiles slots and draws.

use std::{cell::RefCell, rc::Rc};

use super::pipelines::ShadowPipelines;
use crate::bindings::SHADOW_MARKS;
use crate::buffers::ReadbackRing;
use crate::cached::{Bound, GroupCache};
use crate::gpu_layouts::constants::SHADOW_MARKS_GROUP;
use crate::profiling::{GpuProfiler, Span};

/// The marks read back and not yet taken, merged, and the clipmap epoch
/// they were recorded in.
#[derive(Default)]
struct MarksRead {
    epoch: u64,
    words: Vec<u32>,
}

pub(super) struct Marks {
    // The mask the pass marks, of `words` words.
    buffer: crate::cached::Versioned<wgpu::Buffer>,
    words: usize,
    group: GroupCache<4>,
    readbacks: ReadbackRing<u64>,
    read_back: Rc<RefCell<MarksRead>>,
}

impl Marks {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            buffer: crate::cached::Versioned::new(Self::mask(device, 1)),
            words: 0,
            group: GroupCache::default(),
            readbacks: ReadbackRing::new("engine/shadow-marks-readback"),
            read_back: Rc::new(RefCell::new(MarksRead::default())),
        }
    }

    fn mask(device: &wgpu::Device, words: usize) -> wgpu::Buffer {
        crate::buffers::storage(
            device,
            "engine/shadow-marks",
            (words * 4) as u64,
            wgpu::BufferUsages::COPY_SRC,
        )
    }

    /// Bytes of the mask.
    pub(super) fn capacity_bytes(&self) -> u64 {
        self.buffer.size()
    }

    /// Sizes the mask for `words` words, dropping what it had not read
    /// back when they change.
    pub(super) fn ensure(&mut self, device: &wgpu::Device, words: usize) {
        if words == self.words {
            return;
        }
        self.words = words;
        self.buffer.replace(Self::mask(device, words.max(1)));
        self.readbacks.clear();
    }

    /// Takes the marks read back since the last call, recorded in epoch
    /// `epoch` of the clipmap (others name cells that no longer hold what
    /// they did).
    pub(super) fn take(&self, epoch: u64) -> Vec<u32> {
        let mut read = self.read_back.borrow_mut();
        let words = std::mem::take(&mut read.words);
        if read.epoch == epoch {
            words
        } else {
            Vec::new()
        }
    }

    /// Records the marks of the frame's pixels, `size` of them from the
    /// depth buffer's top-left, read through `camera` (its uniform) and
    /// `lookups` (the shadow lookups' uniform) from `depth`, then their
    /// read-back tagged with the clipmap's `epoch`; timed by `profiler`
    /// (`shadow_marks`).
    #[expect(clippy::too_many_arguments, reason = "a pass's disjoint inputs")]
    pub(super) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &ShadowPipelines,
        (camera, lookups): (&wgpu::Buffer, &wgpu::Buffer),
        depth: Bound<'_>,
        size: (u32, u32),
        epoch: u64,
        profiler: &GpuProfiler,
    ) {
        if self.words == 0 || size.0 == 0 || size.1 == 0 {
            return;
        }
        let group = self.group.bind(
            device,
            "engine/shadow-marks-group",
            (&SHADOW_MARKS, &pipelines.marks_layout),
            [
                ("camera", Bound::fixed(camera.as_entire_binding())),
                ("shadow", Bound::fixed(lookups.as_entire_binding())),
                ("shadow_depth", depth),
                ("shadow_marks", self.buffer.bound()),
            ],
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/shadow-marks"),
                timestamp_writes: profiler.span("shadow_marks").map(Span::compute),
            });
            pass.set_pipeline(&pipelines.marks);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(
                size.0.div_ceil(SHADOW_MARKS_GROUP),
                size.1.div_ceil(SHADOW_MARKS_GROUP),
                1,
            );
        }
        // Unread marks stay until a read-back copies them.
        let bytes = (self.words * 4) as u64;
        if self
            .readbacks
            .record(device, encoder, &self.buffer, bytes, epoch)
        {
            encoder.clear_buffer(&self.buffer, 0, Some(bytes));
        }
    }

    /// Requests maps of the marks the frame just submitted read back.
    pub(super) fn after_submit(&mut self) {
        let read_back = Rc::clone(&self.read_back);
        let words = self.words;
        self.readbacks.after_submit(move |read, epoch| {
            let mut marks = read_back.borrow_mut();
            if marks.epoch != epoch {
                *marks = MarksRead {
                    epoch,
                    words: Vec::new(),
                };
            }
            marks.words.resize(words, 0);
            for (index, word) in marks.words.iter_mut().enumerate() {
                *word |= read.get(index);
            }
        });
    }
}
