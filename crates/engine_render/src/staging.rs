use std::{cell::Cell, ops::Range, rc::Rc};

pub const MIN_STAGED_BYTES: usize = 256 * 1024;
const MAX_BUFFERS: usize = 6;
const SIZE_STEP: u64 = 256 * 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Mapped,
    Recorded,
    Pending,
    Failed,
}

struct Staging {
    buffer: wgpu::Buffer,
    size: u64,
    state: Rc<Cell<State>>,
}

#[derive(Default)]
pub(crate) struct StagingRing {
    buffers: Vec<Staging>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Staged {
    Recorded,
    NoMappedBuffer,
}

impl StagingRing {
    pub(crate) fn record(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        destination: &wgpu::Buffer,
        (records, stride): (&[u8], usize),
        ranges: &[Range<usize>],
    ) -> Staged {
        let bytes: usize = ranges
            .iter()
            .map(|range| (range.end - range.start) * stride)
            .sum();
        if bytes == 0 {
            return Staged::NoMappedBuffer;
        }
        let Some(index) = self.mapped(device, bytes) else {
            return Staged::NoMappedBuffer;
        };
        let staging = &self.buffers[index];
        let mut offset = 0;
        let writes = ranges.iter().map(|range| {
            let source = &records[range.start * stride..range.end * stride];
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
        for range in ranges {
            let length = ((range.end - range.start) * stride) as u64;
            encoder.copy_buffer_to_buffer(
                &staging.buffer,
                offset,
                destination,
                (range.start * stride) as u64,
                length,
            );
            offset += length;
        }
        Staged::Recorded
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
