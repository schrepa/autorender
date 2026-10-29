use std::{cell::RefCell, rc::Rc};

use engine_render::buffers::ReadbackRing;
use engine_render::cached::{Bound, GroupCache};
use engine_render::profiling::{GpuProfiler, Span};

use super::pipelines::ShadowPipelines;
use crate::bindings::SHADOW_MARKS;
use crate::gpu_layouts::SHADOW_MARKS_GROUP;

#[derive(Default)]
struct MarksRead {
    epoch: u64,
    words: Vec<u32>,
}

pub(super) struct Marks {
    buffer: engine_render::cached::Versioned<wgpu::Buffer>,
    words: usize,
    group: GroupCache<4>,
    readbacks: ReadbackRing<u64>,
    read_back: Rc<RefCell<MarksRead>>,
}

impl Marks {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            buffer: engine_render::cached::Versioned::new(Self::mask(device, 1)),
            words: 0,
            group: GroupCache::default(),
            readbacks: ReadbackRing::new("engine/shadow-marks-readback"),
            read_back: Rc::new(RefCell::new(MarksRead::default())),
        }
    }

    fn mask(device: &wgpu::Device, words: usize) -> wgpu::Buffer {
        engine_render::buffers::storage(
            device,
            "engine/shadow-marks",
            (words * 4) as u64,
            wgpu::BufferUsages::COPY_SRC,
        )
    }

    pub(super) fn capacity_bytes(&self) -> u64 {
        self.buffer.size()
    }

    pub(super) fn ensure(&mut self, device: &wgpu::Device, words: usize) {
        if words == self.words {
            return;
        }
        self.words = words;
        self.buffer.replace(Self::mask(device, words.max(1)));
        self.readbacks.clear();
    }

    pub(super) fn take(&self, epoch: u64) -> Vec<u32> {
        let mut read = self.read_back.borrow_mut();
        let words = std::mem::take(&mut read.words);
        if read.epoch == epoch {
            words
        } else {
            Vec::new()
        }
    }

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
        let bytes = (self.words * 4) as u64;
        if self
            .readbacks
            .record(device, encoder, &self.buffer, bytes, epoch)
        {
            encoder.clear_buffer(&self.buffer, 0, Some(bytes));
        }
    }

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
