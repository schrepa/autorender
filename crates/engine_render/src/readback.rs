use std::cell::Cell;
use std::ops::Range;
use std::rc::Rc;

use engine_core::FrameId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reader {
    FrameTimer,
    #[cfg_attr(
        not(feature = "dev"),
        expect(dead_code, reason = "only dev and profiling builds read GPU spans")
    )]
    Profiler(u32),
    #[cfg_attr(
        not(feature = "dev"),
        expect(dead_code, reason = "only dev and profiling builds count shader work")
    )]
    Work,
    Part(u32, u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Idle,
    Recorded,
    Mapping,
    Mapped,
    Failed,
}

pub(crate) enum Arrival<'a> {
    Read(&'a [u8]),
    Failed,
}

#[derive(Clone, Copy)]
struct Asked {
    written: FrameId,
    bytes: u64,
}

struct Slot {
    buffer: wgpu::Buffer,
    capacity: u64,
    reader: Reader,
    phase: Rc<Cell<Phase>>,
    asked: Option<Asked>,
}

impl Slot {
    fn reusable(&self, reader: Reader) -> bool {
        self.reader == reader && self.phase.get() == Phase::Idle
    }
}

#[derive(Default)]
pub(crate) struct Readbacks {
    slots: Vec<Slot>,
}

impl Readbacks {
    pub(crate) fn record(
        &mut self,
        (device, encoder): (&wgpu::Device, &mut wgpu::CommandEncoder),
        (source, range): (&wgpu::Buffer, Range<u64>),
        (reader, written): (Reader, FrameId),
        most: usize,
    ) -> bool {
        let bytes = range.end.saturating_sub(range.start).next_multiple_of(4);
        let found = self
            .slots
            .iter()
            .position(|slot| slot.reusable(reader) && slot.capacity >= bytes);
        let at = match found {
            Some(at) => at,
            None => {
                self.slots
                    .retain(|slot| !(slot.reusable(reader) && slot.capacity < bytes));
                let held = self.slots.iter().filter(|slot| slot.reader == reader).count();
                if held >= most.max(1) {
                    return false;
                }
                let capacity = bytes.max(4);
                self.slots.push(Slot {
                    buffer: crate::gpu_calls::create_buffer(device, &wgpu::BufferDescriptor {
                        label: Some("engine/readback"),
                        size: capacity,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    capacity,
                    reader,
                    phase: Rc::new(Cell::new(Phase::Idle)),
                    asked: None,
                });
                self.slots.len() - 1
            }
        };
        let slot = &mut self.slots[at];
        encoder.copy_buffer_to_buffer(source, range.start, &slot.buffer, 0, bytes);
        slot.phase.set(Phase::Recorded);
        slot.asked = Some(Asked { written, bytes });
        true
    }

    pub(crate) fn map(&self) {
        for slot in &self.slots {
            if slot.phase.get() != Phase::Recorded {
                continue;
            }
            slot.phase.set(Phase::Mapping);
            let phase = Rc::clone(&slot.phase);
            slot.buffer.slice(..).map_async(
                wgpu::MapMode::Read,
                crate::platform::map_callback(move |result| {
                    phase.set(if result.is_ok() {
                        Phase::Mapped
                    } else {
                        Phase::Failed
                    });
                }),
            );
        }
    }

    pub(crate) fn take(&mut self, mut each: impl FnMut(Reader, FrameId, Arrival<'_>)) {
        for slot in &mut self.slots {
            let phase = slot.phase.get();
            match phase {
                Phase::Idle | Phase::Recorded | Phase::Mapping => continue,
                Phase::Mapped | Phase::Failed => {}
            }
            if let Some(asked) = slot.asked.take() {
                let mapped = (phase == Phase::Mapped)
                    .then(|| slot.buffer.slice(..).get_mapped_range().ok())
                    .flatten();
                match &mapped {
                    Some(view) => {
                        let end = usize::try_from(asked.bytes)
                            .unwrap_or(usize::MAX)
                            .min(view.len());
                        each(slot.reader, asked.written, Arrival::Read(&view[..end]));
                    }
                    None => each(slot.reader, asked.written, Arrival::Failed),
                }
                drop(mapped);
            }
            if phase == Phase::Mapped {
                slot.buffer.unmap();
            }
            slot.phase.set(Phase::Idle);
        }
    }
}
