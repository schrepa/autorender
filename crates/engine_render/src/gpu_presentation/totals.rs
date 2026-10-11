//! Totals GPU presentation reads back a few frames late: visible, hidden
//! and sub-pixel instances, instances per
//! level of detail and the triangles drawn, with how many frames late.

use std::{cell::Cell, rc::Rc};

use crate::buffers::ReadbackRing;

// Totals read back: phase one's (or the only) visible count, phase two's,
// the candidates phase two found hidden, and those covering no pixel center;
// then phase one's and phase two's drawn counts per level of
// detail; then the triangles both phases' draws submit, and the level of
// detail state of the slot development tools watch.
const TOTALS_BYTES: u64 = 56;

/// Where the watched slot's level of detail state lands in the totals.
pub(super) const WATCHED_OFFSET: u64 = 52;

/// One frame's totals: instances visible (drawn in either phase, hidden,
/// or covering no pixel center), those occlusion culling left undrawn,
/// those covering no pixel center, instances drawn at each level of detail
/// (the mesh, its two simplifications, then impostors), and the triangles
/// the camera's draws submitted, and the slot watched with its level of
/// detail state; and the GPU-presented frame they come from, counted as
/// [`Totals::age`] counts.
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
    // The frame's totals, copied into a read-back buffer.
    pub(super) buffer: wgpu::Buffer,
    // Tagged with the watched slot and the frame recording them.
    readbacks: ReadbackRing<(Option<u32>, u64)>,
    last: Rc<Cell<Option<FrameTotals>>>,
    // GPU-presented frames submitted.
    frames: u64,
}

impl Totals {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine/presentation-totals"),
                size: TOTALS_BYTES,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            readbacks: ReadbackRing::new("engine/presentation-visible-readback"),
            last: Rc::new(Cell::new(None)),
            frames: 0,
        }
    }

    /// The latest totals read back, a few frames old.
    pub(super) fn last(&self) -> Option<FrameTotals> {
        self.last.get()
    }

    /// GPU-presented frames submitted since the one whose totals
    /// [`Self::last`] holds: 1 when they arrived as soon as they could, more
    /// while the GPU falls behind, since a frame finding every read-back
    /// buffer in flight records none.
    pub(super) fn age(&self) -> Option<u64> {
        self.last.get().map(|totals| self.frames - totals.frame)
    }

    /// Records a copy of the totals into a free read-back buffer, if any,
    /// with the slot whose state they hold.
    pub(super) fn read_back(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        watched: Option<u32>,
    ) {
        // This frame, as `after_submit` will have counted it.
        let tag = (watched, self.frames + 1);
        self.readbacks
            .record(device, encoder, &self.buffer, TOTALS_BYTES, tag);
    }

    /// Counts the GPU-presented frame just submitted and requests maps of
    /// the read-backs it recorded.
    pub(super) fn after_submit(&mut self) {
        self.frames += 1;
        let last = Rc::clone(&self.last);
        self.readbacks.after_submit(move |words, (watched, frame)| {
            // Visible: drawn in either phase, hidden, or covering no pixel
            // center.
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
