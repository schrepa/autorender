use std::cell::Cell;
use std::ops::Range;
use std::rc::Rc;

use bevy_ecs::world::World;
use engine_core::{FrameId, Readback};

pub(crate) type Deliver = fn(&mut World, u32, Readback<&[u8]>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reader {
    Part { part: u32, tag: u32 },
    FrameTimer,
    #[cfg_attr(
        not(feature = "profiling"),
        expect(dead_code, reason = "only profiling builds read GPU spans")
    )]
    Profiler(u32),
    Impostor(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Keep {
    Ring(usize),
    Once,
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
    deliver: Option<Deliver>,
}

struct Slot {
    buffer: wgpu::Buffer,
    capacity: u64,
    reader: Reader,
    once: bool,
    phase: Rc<Cell<Phase>>,
    asked: Option<Asked>,
}

impl Slot {
    fn reusable(&self, reader: Reader) -> bool {
        self.reader == reader && !self.once && self.phase.get() == Phase::Idle
    }
}

#[derive(Default)]
pub(crate) struct Readbacks {
    slots: Vec<Slot>,
    overflows: u64,
}

impl Readbacks {
    pub(crate) fn overflows(&self) -> u64 {
        self.overflows
    }

    pub(crate) fn overflowed(&mut self) {
        self.overflows += 1;
    }

    pub(crate) fn room(&self, reader: Reader, most: usize) -> bool {
        let held = self
            .slots
            .iter()
            .filter(|slot| slot.reader == reader && !slot.once)
            .count();
        held < most.max(1) || self.slots.iter().any(|slot| slot.reusable(reader))
    }

    pub(crate) fn record(
        &mut self,
        recording: (&wgpu::Device, &mut wgpu::CommandEncoder),
        (source, range): (&wgpu::Buffer, Range<u64>),
        asked: (Reader, FrameId, Option<Deliver>),
        keep: Keep,
    ) -> bool {
        let bytes = range.end.saturating_sub(range.start).next_multiple_of(4);
        self.record_with(recording, bytes, asked, keep, |encoder, slot| {
            encoder.copy_buffer_to_buffer(source, range.start, slot, 0, bytes);
        })
    }

    pub(crate) fn record_with(
        &mut self,
        (device, encoder): (&wgpu::Device, &mut wgpu::CommandEncoder),
        bytes: u64,
        (reader, written, deliver): (Reader, FrameId, Option<Deliver>),
        keep: Keep,
        copy: impl FnOnce(&mut wgpu::CommandEncoder, &wgpu::Buffer),
    ) -> bool {
        let found = match keep {
            Keep::Once => None,
            Keep::Ring(most) => {
                let found = self
                    .slots
                    .iter()
                    .position(|slot| slot.reusable(reader) && slot.capacity >= bytes);
                if found.is_none() {
                    self.slots.retain(|slot| {
                        let small = slot.reusable(reader) && slot.capacity < bytes;
                        if small {
                            crate::memory::release_buffer(&slot.buffer);
                        }
                        !small
                    });
                    let held = self
                        .slots
                        .iter()
                        .filter(|slot| slot.reader == reader && !slot.once)
                        .count();
                    if held >= most.max(1) {
                        return false;
                    }
                }
                found
            }
        };
        let at = found.unwrap_or_else(|| {
            let capacity = bytes.max(4);
            self.slots.push(Slot {
                buffer: crate::memory::buffer(
                    device,
                    &wgpu::BufferDescriptor {
                        label: Some("engine/readback"),
                        size: capacity,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                ),
                capacity,
                reader,
                once: keep == Keep::Once,
                phase: Rc::new(Cell::new(Phase::Idle)),
                asked: None,
            });
            self.slots.len() - 1
        });
        let slot = &mut self.slots[at];
        copy(encoder, &slot.buffer);
        slot.phase.set(Phase::Recorded);
        slot.asked = Some(Asked {
            written,
            bytes,
            deliver,
        });
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

    pub(crate) fn take(
        &mut self,
        mut each: impl FnMut(Reader, FrameId, Option<Deliver>, Arrival<'_>),
    ) {
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
                        each(
                            slot.reader,
                            asked.written,
                            asked.deliver,
                            Arrival::Read(&view[..end]),
                        );
                    }
                    None => each(slot.reader, asked.written, asked.deliver, Arrival::Failed),
                }
                drop(mapped);
            }
            if phase == Phase::Mapped {
                slot.buffer.unmap();
            }
            slot.phase.set(Phase::Idle);
        }
        self.slots.retain(|slot| {
            let spent = slot.once && slot.phase.get() == Phase::Idle;
            if spent {
                crate::memory::release_buffer(&slot.buffer);
            }
            !spent
        });
    }
}
