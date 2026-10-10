//! Persistently mapped staging buffers for large retained pose uploads.
//!
//! Firefox 156 backs every `writeBuffer` of 1 KiB or more with newly created
//! shared memory: it maps a fresh section, copies the data (faulting every new
//! page) and unmaps it again. Buffers created with map usage instead keep one
//! shared mapping for their lifetime, and `getMappedRange` exposes it without a
//! copy. Pose ranges are therefore copied from Wasm memory straight into an
//! already mapped staging buffer, which is unmapped and copied on the GPU by a
//! command recorded before the frame's draws.
//!
//! Maps resolve asynchronously once the GPU has finished with a buffer. The
//! ring never waits: without a mapped buffer the caller uses `write_buffer`.
//! At most [`MAX_BUFFERS`] buffers exist, each sized to the largest upload so
//! far; the ring is dropped when the device-side instance storage changes.

use std::{cell::Cell, rc::Rc};

use crate::uploads::UploadRange;

/// Smaller uploads keep `write_buffer`; they are cheap on every browser.
pub(crate) const MIN_STAGED_BYTES: usize = 256 * 1024;
/// A buffer maps again only once the GPU finished the frame that copied from
/// it, a few frames later when the GPU limits the frame rate: with three,
/// half of Metro's frames found none mapped (Firefox, Xe-LPG). The ring adds
/// buffers only while none is mapped.
const MAX_BUFFERS: usize = 6;
/// Staging sizes grow in 256 KiB steps to avoid reallocating per frame.
const SIZE_STEP: u64 = 256 * 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    /// Mapped for writing and not referenced by submitted work.
    Mapped,
    /// Unmapped by this frame; must be submitted before its map is requested.
    Recorded,
    /// Unmapped holding data staged ahead of the frame that copies it; see
    /// [`Prestaged`]. Released to `Recorded` when that frame uses or drops it.
    Prestaged,
    /// Map requested after submission; the callback moves it to Mapped.
    Pending,
    /// The browser rejected the map (device loss or destruction).
    Failed,
}

struct Staging {
    buffer: wgpu::Buffer,
    size: u64,
    state: Rc<Cell<State>>,
}

/// Data staged into an unmapped ring buffer ahead of the frame that copies
/// it. Using or dropping it returns the buffer to the ring after that frame's
/// submission.
pub(crate) struct Prestaged {
    buffer: wgpu::Buffer,
    state: Rc<Cell<State>>,
    bytes: u64,
}

impl Prestaged {
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Records the copy of the staged bytes to `destination` at `offset`.
    pub(crate) fn copy_to(
        self,
        encoder: &mut wgpu::CommandEncoder,
        destination: &wgpu::Buffer,
        offset: u64,
    ) {
        encoder.copy_buffer_to_buffer(&self.buffer, 0, destination, offset, self.bytes);
    }
}

impl Drop for Prestaged {
    fn drop(&mut self) {
        // Submitted with (or unused by) this frame: map it again afterwards.
        if self.state.get() == State::Prestaged {
            self.state.set(State::Recorded);
        }
    }
}

#[derive(Default)]
pub(crate) struct StagingRing {
    buffers: Vec<Staging>,
}

/// Outcome of one staged upload attempt, for profiling counters.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Staged {
    Recorded,
    NoMappedBuffer,
}

/// Records of one source slice, each range copied to
/// `base + range.start * stride` of `destination`.
pub(crate) struct Segment<'a> {
    pub destination: &'a wgpu::Buffer,
    pub base: u64,
    pub records: &'a [u8],
    pub stride: usize,
    pub ranges: &'a [UploadRange],
}

impl Segment<'_> {
    pub(crate) fn bytes(&self) -> usize {
        self.ranges
            .iter()
            .map(|range| (range.end - range.start) * self.stride)
            .sum()
    }
}

impl StagingRing {
    /// Copy `ranges` of `records` into one mapped staging buffer at packed
    /// offsets and record GPU copies to the same offsets of `destination`.
    /// Returns `NoMappedBuffer` without side effects when none is ready.
    pub(crate) fn record(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        destination: &wgpu::Buffer,
        records: &[u8],
        stride: usize,
        ranges: &[UploadRange],
    ) -> Staged {
        self.record_segments(
            device,
            encoder,
            &[Segment {
                destination,
                base: 0,
                records,
                stride,
                ranges,
            }],
        )
    }

    /// Packs every segment into one mapped staging buffer (one buffer per
    /// frame however many streams upload) and records their GPU copies.
    pub(crate) fn record_segments(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        segments: &[Segment<'_>],
    ) -> Staged {
        let bytes: usize = segments.iter().map(Segment::bytes).sum();
        let Some(index) = self.mapped(device, bytes) else {
            return Staged::NoMappedBuffer;
        };
        let staging = &self.buffers[index];
        let mut offset = 0;
        let writes = segments.iter().flat_map(|segment| {
            let stride = segment.stride;
            segment
                .ranges
                .iter()
                .map(move |range| &segment.records[range.start * stride..range.end * stride])
        });
        let writes = writes.map(|source| {
            let at = offset;
            offset += source.len();
            (at, source)
        });
        if !crate::platform::write_mapped(staging.buffer.slice(0..bytes as u64), writes) {
            staging.state.set(State::Failed);
            return Staged::NoMappedBuffer;
        }
        staging.buffer.unmap();
        staging.state.set(State::Recorded);
        let mut offset = 0;
        for segment in segments {
            for range in segment.ranges {
                let length = ((range.end - range.start) * segment.stride) as u64;
                encoder.copy_buffer_to_buffer(
                    &staging.buffer,
                    offset,
                    segment.destination,
                    segment.base + (range.start * segment.stride) as u64,
                    length,
                );
                offset += length;
            }
        }
        Staged::Recorded
    }

    /// Copies `bytes` into a mapped staging buffer now, between frames, for a
    /// later frame to copy on the GPU. `None` without side effects when no
    /// buffer is ready.
    pub(crate) fn prestage(&mut self, device: &wgpu::Device, bytes: &[u8]) -> Option<Prestaged> {
        let index = self.mapped(device, bytes.len())?;
        let staging = &self.buffers[index];
        if !crate::platform::write_mapped(staging.buffer.slice(0..bytes.len() as u64), [(0, bytes)])
        {
            staging.state.set(State::Failed);
            return None;
        }
        staging.buffer.unmap();
        staging.state.set(State::Prestaged);
        Some(Prestaged {
            buffer: staging.buffer.clone(),
            state: Rc::clone(&staging.state),
            bytes: bytes.len() as u64,
        })
    }

    /// A mapped buffer of at least `bytes`, creating one within the ring's
    /// bound; `None` (after retiring a buffer too small once mapped) otherwise.
    fn mapped(&mut self, device: &wgpu::Device, bytes: usize) -> Option<usize> {
        self.buffers
            .retain(|staging| staging.state.get() != State::Failed);
        let needed = (bytes as u64).div_ceil(SIZE_STEP) * SIZE_STEP;
        let index = match self.buffers.iter().position(|staging| {
            staging.state.get() == State::Mapped && staging.size >= bytes as u64
        }) {
            Some(index) => index,
            None if self.buffers.len() < MAX_BUFFERS => {
                // A new buffer is created mapped; nothing waits for it.
                self.buffers.push(Staging {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("engine/pose-staging"),
                        size: needed,
                        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: true,
                    }),
                    size: needed,
                    state: Rc::new(Cell::new(State::Mapped)),
                });
                self.buffers.len() - 1
            }
            None => {
                // Replace a buffer that is too small once it maps again.
                if let Some(small) = self.buffers.iter().position(|staging| {
                    staging.state.get() == State::Mapped && staging.size < bytes as u64
                }) {
                    self.buffers.swap_remove(small);
                }
                return None;
            }
        };
        Some(index)
    }

    /// Request maps for buffers whose copies were just submitted.
    pub(crate) fn after_submit(&mut self) {
        for staging in &self.buffers {
            if staging.state.get() != State::Recorded {
                continue;
            }
            staging.state.set(State::Pending);
            let state = Rc::clone(&staging.state);
            staging.buffer.slice(..).map_async(
                wgpu::MapMode::Write,
                crate::platform::map_callback(move |result| {
                    state.set(if result.is_ok() {
                        State::Mapped
                    } else {
                        State::Failed
                    });
                }),
            );
        }
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.buffers.iter().map(|staging| staging.size).sum()
    }
}
