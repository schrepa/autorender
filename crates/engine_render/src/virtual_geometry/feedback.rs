//! What selection reads back a few frames late: the counts
//! of clusters and triangles each view selected and of entries dropped for
//! want of room, and the groups it used and would refine into, whose pages
//! streaming loads.

use std::{cell::Cell, rc::Rc};

use super::VirtualStats;
use crate::buffers::ReadbackRing;

/// What a frame read back: its counts, dropped queue, group and cluster
/// entries and whether its retest list overflowed, and the groups it used
/// and would refine into (bits of global groups).
pub(super) struct Feedback {
    pub(super) stats: VirtualStats,
    pub(super) drops: [u32; 4],
    pub(super) used: Vec<u32>,
    pub(super) requested: Vec<u32>,
    pub(super) frame: u64,
}

pub(super) struct Readbacks {
    // Each read-back's frame, and where its used and requested group bits
    // start and their words each.
    readbacks: ReadbackRing<(u64, (usize, usize))>,
    // Feedback mapped since the last frame took it.
    read: Rc<Cell<Option<Feedback>>>,
    // The counts of the last frame taken.
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

    /// Takes the feedback mapped since the last call, keeping its counts.
    pub(super) fn take(&mut self) -> Option<Feedback> {
        let feedback = self.read.take()?;
        self.last = Some(feedback.stats);
        Some(feedback)
    }

    /// The counts of the last frame taken.
    pub(super) fn last(&self) -> Option<VirtualStats> {
        self.last
    }

    /// Drops the read-back buffers, sized for counters since replaced.
    pub(super) fn clear(&mut self) {
        self.readbacks.clear();
    }

    /// Records a copy of frame `frame`'s `counters`, whose used and
    /// requested group bits start at `feedback.0` with `feedback.1` words
    /// each, when a read-back buffer is free.
    pub(super) fn record(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        counters: &wgpu::Buffer,
        frame: u64,
        feedback: (usize, usize),
    ) {
        // Counters, bins and both feedback bitsets.
        self.readbacks.record(
            device,
            encoder,
            counters,
            counters.size(),
            (frame, feedback),
        );
    }

    /// Requests maps of counters recorded in the frame just submitted, whose
    /// retest list had room for `retest_capacity` clusters.
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
