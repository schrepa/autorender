use std::{cell::Cell, rc::Rc};

use super::VirtualStats;
use crate::buffers::ReadbackRing;

pub(super) struct Feedback {
    pub(super) stats: VirtualStats,
    pub(super) drops: [u32; 4],
    pub(super) used: Vec<u32>,
    pub(super) requested: Vec<u32>,
    pub(super) frame: u64,
}

pub(super) struct Readbacks {
    readbacks: ReadbackRing<(u64, (usize, usize))>,
    read: Rc<Cell<Option<Feedback>>>,
    last: Option<VirtualStats>,
}

impl Readbacks {
    pub(super) fn new() -> Self {
        Self {
            readbacks: ReadbackRing::new("engine/virtual-counters-readback"),
            read: Rc::new(Cell::new(None)),
            last: None,
        }
    }

    pub(super) fn take(&mut self) -> Option<Feedback> {
        let feedback = self.read.take()?;
        self.last = Some(feedback.stats);
        Some(feedback)
    }

    pub(super) fn last(&self) -> Option<VirtualStats> {
        self.last
    }

    pub(super) fn clear(&mut self) {
        self.readbacks.clear();
    }

    pub(super) fn record(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        counters: &wgpu::Buffer,
        frame: u64,
        feedback: (usize, usize),
    ) {
        self.readbacks.record(
            device,
            encoder,
            counters,
            counters.size(),
            (frame, feedback),
        );
    }

    pub(super) fn after_submit(&mut self, retest_capacity: u32) {
        let read = Rc::clone(&self.read);
        self.readbacks
            .after_submit(move |words, (frame, (feedback, feedback_words))| {
                let drops = [words.get(3), words.get(4), words.get(5)];
                let bits = |start: usize| -> Vec<u32> {
                    (start..start + feedback_words)
                        .map(|index| words.get(index))
                        .collect()
                };
                read.set(Some(Feedback {
                    stats: VirtualStats {
                        clusters: words.get(6),
                        triangles: words.get(7),
                        shadow_clusters: words.get(8),
                        shadow_triangles: words.get(9),
                        occluded: words.get(11),
                        dropped: drops.iter().sum(),
                        ..VirtualStats::default()
                    },
                    drops: [
                        drops[0],
                        drops[1],
                        drops[2],
                        u32::from(words.get(10) > retest_capacity),
                    ],
                    used: bits(feedback),
                    requested: bits(feedback + feedback_words),
                    frame,
                }));
            });
    }
}
