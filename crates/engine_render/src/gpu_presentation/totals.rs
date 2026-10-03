//! Totals GPU presentation reads back a few frames late: visible, hidden
//! (decision 040) and sub-pixel (decision 041) instances, instances per
//! level of detail (decision 047) and the triangles drawn.

use std::{cell::Cell, rc::Rc};

const READBACKS: usize = 3;

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

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Idle,
    Recorded,
    Pending,
}

struct Readback {
    buffer: wgpu::Buffer,
    state: Rc<Cell<State>>,
}

pub(super) struct Totals {
    // The frame's totals, copied into a read-back buffer.
    pub(super) buffer: wgpu::Buffer,
    readbacks: Vec<Readback>,
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
            readbacks: Vec::new(),
            last: Rc::new(Cell::new(None)),
        }
    }

    /// The latest totals read back, a few frames old.
    pub(super) fn last(&self) -> Option<FrameTotals> {
        self.last.get()
    }

    /// Records a copy of the totals into a free read-back buffer, if any.
    pub(super) fn read_back(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let index = match self
            .readbacks
            .iter()
            .position(|readback| readback.state.get() == State::Idle)
        {
            Some(index) => index,
            None if self.readbacks.len() < READBACKS => {
                self.readbacks.push(Readback {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("engine/presentation-visible-readback"),
                        size: TOTALS_BYTES,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    state: Rc::new(Cell::new(State::Idle)),
                });
                self.readbacks.len() - 1
            }
            None => return,
        };
        let readback = &self.readbacks[index];
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &readback.buffer, 0, TOTALS_BYTES);
        readback.state.set(State::Recorded);
    }

    /// Requests maps of read-backs recorded in the frame just submitted.
    pub(super) fn after_submit(&mut self) {
        for readback in &self.readbacks {
            if readback.state.get() != State::Recorded {
                continue;
            }
            readback.state.set(State::Pending);
            let state = Rc::clone(&readback.state);
            let last = Rc::clone(&self.last);
            let buffer = readback.buffer.clone();
            readback
                .buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        if let Ok(view) = buffer.slice(..).get_mapped_range() {
                            let word = |index: usize| {
                                let bytes: [u8; 4] =
                                    view[index * 4..index * 4 + 4].try_into().unwrap_or([0; 4]);
                                u32::from_le_bytes(bytes)
                            };
                            // Visible: drawn in either phase, hidden, or
                            // covering no pixel center.
                            let hidden = word(2);
                            let tiny = word(3);
                            last.set(Some(FrameTotals {
                                visible: word(0)
                                    .saturating_add(word(1))
                                    .saturating_add(hidden)
                                    .saturating_add(tiny),
                                occluded: hidden,
                                subpixel: tiny,
                                levels: std::array::from_fn(|level| {
                                    word(4 + level).saturating_add(word(8 + level))
                                }),
                                triangles: word(12),
                            }));
                        }
                        buffer.unmap();
                    }
                    state.set(State::Idle);
                });
        }
    }
}
