use std::{cell::Cell, rc::Rc};

use crate::uploads::UploadRange;

pub const MIN_STAGED_BYTES: usize = 256 * 1024;
const MAX_BUFFERS: usize = 6;
const SIZE_STEP: u64 = 256 * 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Mapped,
    Recorded,
    Prestaged,
    Pending,
    Failed,
}

struct Staging {
    buffer: wgpu::Buffer,
    size: u64,
    state: Rc<Cell<State>>,
}

pub struct Prestaged {
    buffer: wgpu::Buffer,
    state: Rc<Cell<State>>,
    bytes: u64,
}

impl Prestaged {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn copy_to(
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
        if self.state.get() == State::Prestaged {
            self.state.set(State::Recorded);
        }
    }
}

#[derive(Default)]
pub struct StagingRing {
    buffers: Vec<Staging>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Staged {
    Recorded,
    NoMappedBuffer,
}

pub struct Segment<'a> {
    pub destination: &'a wgpu::Buffer,
    pub base: u64,
    pub records: &'a [u8],
    pub stride: usize,
    pub ranges: &'a [UploadRange],
}

impl Segment<'_> {
    pub fn bytes(&self) -> usize {
        self.ranges
            .iter()
            .map(|range| (range.end - range.start) * self.stride)
            .sum()
    }
}

impl StagingRing {
    pub fn record(
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

    pub fn record_segments(
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

    pub fn prestage(&mut self, device: &wgpu::Device, bytes: &[u8]) -> Option<Prestaged> {
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

    fn mapped(&mut self, device: &wgpu::Device, bytes: usize) -> Option<usize> {
        self.buffers
            .retain(|staging| staging.state.get() != State::Failed);
        let needed = (bytes as u64).div_ceil(SIZE_STEP) * SIZE_STEP;
        let index = match self.buffers.iter().position(|staging| {
            staging.state.get() == State::Mapped && staging.size >= bytes as u64
        }) {
            Some(index) => index,
            None if self.buffers.len() < MAX_BUFFERS => {
                self.buffers.push(Staging {
                    buffer: crate::memory::buffer(
                        device,
                        &wgpu::BufferDescriptor {
                            label: Some("engine/pose-staging"),
                            size: needed,
                            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                            mapped_at_creation: true,
                        },
                    ),
                    size: needed,
                    state: Rc::new(Cell::new(State::Mapped)),
                });
                self.buffers.len() - 1
            }
            None => {
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

    pub fn after_submit(&mut self) {
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

    pub fn capacity_bytes(&self) -> u64 {
        self.buffers.iter().map(|staging| staging.size).sum()
    }
}
