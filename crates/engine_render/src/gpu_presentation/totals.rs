//! Totals GPU presentation reads back a few frames late: visible, hidden
//! (decision 040) and sub-pixel (decision 041) instances, instances per
//! level of detail (decision 047) and the triangles drawn.

use std::{cell::Cell, rc::Rc};

use crate::buffers::ReadbackRing;

// Totals read back: phase one's (or the only) visible count, phase two's,
// the candidates phase two found hidden, and those covering no pixel center
// (decision 041); then phase one's and phase two's drawn counts per level of
// detail (decision 047); then the triangles both phases' draws submit.
const TOTALS_BYTES: u64 = 52;

/// One frame's totals: instances visible (drawn in either phase, hidden,
/// or covering no pixel center), those occlusion culling left undrawn,
/// those covering no pixel center, instances drawn at each level of detail
/// (the mesh, its two simplifications, then impostors), and the triangles
/// the camera's draws submitted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct FrameTotals {
    pub(super) visible: u32,
    pub(super) occluded: u32,
    pub(super) subpixel: u32,
    pub(super) levels: [u32; 4],
    pub(super) triangles: u32,
}

pub(super) struct Totals {
    // The frame's totals, copied into a read-back buffer.
    pub(super) buffer: wgpu::Buffer,
    readbacks: ReadbackRing<()>,
    last: Rc<Cell<Option<FrameTotals>>>,
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
        }
    }

    /// The latest totals read back, a few frames old.
    pub(super) fn last(&self) -> Option<FrameTotals> {
        self.last.get()
    }

    /// Records a copy of the totals into a free read-back buffer, if any.
    pub(super) fn read_back(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.readbacks
            .record(device, encoder, &self.buffer, TOTALS_BYTES, ());
    }

    /// Requests maps of read-backs recorded in the frame just submitted.
    pub(super) fn after_submit(&mut self) {
        let last = Rc::clone(&self.last);
        self.readbacks.after_submit(move |words, ()| {
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
            }));
        });
    }
}
