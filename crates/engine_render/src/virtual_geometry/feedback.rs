//! What selection reads back a few frames late (decision 044): the counts
//! of clusters and triangles each view selected and of entries dropped for
//! want of room, and the groups it used and would refine into, whose pages
//! streaming loads.

use std::{cell::Cell, rc::Rc};

use super::VirtualStats;

const READBACKS: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadbackState {
    Idle,
    Recorded,
    Pending,
}

struct Readback {
    buffer: wgpu::Buffer,
    state: Rc<Cell<ReadbackState>>,
    // The frame it recorded, where its used and requested group bits start
    // and their words each.
    frame: u64,
    feedback: (usize, usize),
}

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
    readbacks: Vec<Readback>,
    // Feedback mapped since the last frame took it.
    read: Rc<Cell<Option<Feedback>>>,
    // The counts of the last frame taken.
    last: Option<VirtualStats>,
}

impl Readbacks {
    pub(super) fn new() -> Self {
        Self {
            readbacks: Vec::new(),
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
        let index = match self
            .readbacks
            .iter()
            .position(|readback| readback.state.get() == ReadbackState::Idle)
        {
            Some(index) => index,
            None if self.readbacks.len() < READBACKS => {
                // Counters, bins and both feedback bitsets.
                self.readbacks.push(Readback {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("engine/virtual-counters-readback"),
                        size: counters.size(),
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    state: Rc::new(Cell::new(ReadbackState::Idle)),
                    frame: 0,
                    feedback: (0, 0),
                });
                self.readbacks.len() - 1
            }
            None => return,
        };
        let readback = &mut self.readbacks[index];
        encoder.copy_buffer_to_buffer(counters, 0, &readback.buffer, 0, readback.buffer.size());
        readback.state.set(ReadbackState::Recorded);
        readback.frame = frame;
        readback.feedback = feedback;
    }

    /// Requests maps of counters recorded in the frame just submitted, whose
    /// retest list had room for `retest_capacity` clusters.
    pub(super) fn after_submit(&mut self, retest_capacity: u32) {
        for readback in &self.readbacks {
            if readback.state.get() != ReadbackState::Recorded {
                continue;
            }
            readback.state.set(ReadbackState::Pending);
            let state = Rc::clone(&readback.state);
            let read = Rc::clone(&self.read);
            let buffer = readback.buffer.clone();
            let (frame, (feedback, feedback_words)) = (readback.frame, readback.feedback);
            readback
                .buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        if let Ok(view) = buffer.slice(..).get_mapped_range() {
                            let word = |index: usize| {
                                view.get(index * 4..index * 4 + 4).map_or(0, |bytes| {
                                    u32::from_le_bytes(bytes.try_into().unwrap_or([0; 4]))
                                })
                            };
                            let drops = [word(3), word(4), word(5)];
                            let bits = |start: usize| -> Vec<u32> {
                                (start..start + feedback_words).map(word).collect()
                            };
                            read.set(Some(Feedback {
                                stats: VirtualStats {
                                    clusters: word(6),
                                    triangles: word(7),
                                    shadow_clusters: word(8),
                                    shadow_triangles: word(9),
                                    occluded: word(11),
                                    dropped: drops.iter().sum(),
                                    ..VirtualStats::default()
                                },
                                drops: [
                                    drops[0],
                                    drops[1],
                                    drops[2],
                                    u32::from(word(10) > retest_capacity),
                                ],
                                used: bits(feedback),
                                requested: bits(feedback + feedback_words),
                                frame,
                            }));
                        }
                        buffer.unmap();
                    }
                    state.set(ReadbackState::Idle);
                });
        }
    }
}
