use std::{cell::Cell, rc::Rc};

use crate::buffers::ReadbackRing;

const TOTALS_BYTES: u64 = 56;

pub(super) const WATCHED_OFFSET: u64 = 52;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct FrameTotals {
    pub(super) visible: u32,
    pub(super) occluded: u32,
    pub(super) subpixel: u32,
    pub(super) levels: [u32; 4],
    pub(super) triangles: u32,
    pub(super) watched: Option<(u32, u32)>,
    pub(super) frame: u64,
}

pub(super) struct Totals {
    pub(super) buffer: wgpu::Buffer,
    readbacks: ReadbackRing<(Option<u32>, u64)>,
    last: Rc<Cell<Option<FrameTotals>>>,
    frames: u64,
}

impl Totals {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            buffer: crate::memory::buffer(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("engine/presentation-totals"),
                    size: TOTALS_BYTES,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            ),
            readbacks: ReadbackRing::new("engine/presentation-visible-readback"),
            last: Rc::new(Cell::new(None)),
            frames: 0,
        }
    }

    pub(super) fn last(&self) -> Option<FrameTotals> {
        self.last.get()
    }

    pub(super) fn age(&self) -> Option<u64> {
        self.last.get().map(|totals| self.frames - totals.frame)
    }

    pub(super) fn read_back(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        watched: Option<u32>,
    ) {
        let tag = (watched, self.frames + 1);
        self.readbacks
            .record(device, encoder, &self.buffer, TOTALS_BYTES, tag);
    }

    pub(super) fn after_submit(&mut self) {
        self.frames += 1;
        let last = Rc::clone(&self.last);
        self.readbacks.after_submit(move |words, (watched, frame)| {
            let (hidden, tiny) = (words.get(2), words.get(3));
            last.set(Some(FrameTotals {
                visible: words
                    .get(0)
                    .saturating_add(words.get(1))
                    .saturating_add(hidden)
                    .saturating_add(tiny),
                occluded: hidden,
                subpixel: tiny,
                levels: std::array::from_fn(|level| {
                    words.get(4 + level).saturating_add(words.get(8 + level))
                }),
                triangles: words.get(12),
                watched: watched.map(|slot| (slot, words.get(13))),
                frame,
            }));
        });
    }
}
