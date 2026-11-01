use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub const READBACKS: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Idle,
    Recorded,
    Pending,
}

struct Slot<T> {
    buffer: wgpu::Buffer,
    state: Rc<Cell<State>>,
    tag: T,
}

pub struct Words<'a>(&'a [u8]);

impl Words<'_> {
    pub fn get(&self, index: usize) -> u32 {
        self.0.get(index * 4..index * 4 + 4).map_or(0, |bytes| {
            u32::from_le_bytes(bytes.try_into().unwrap_or([0; 4]))
        })
    }
}

pub struct ReadbackRing<T> {
    label: &'static str,
    slots: RefCell<Vec<Slot<T>>>,
}

impl<T: Copy + 'static> ReadbackRing<T> {
    pub fn new(label: &'static str) -> Self {
        Self {
            label,
            slots: RefCell::new(Vec::new()),
        }
    }

    pub fn record(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        bytes: u64,
        tag: T,
    ) -> bool {
        self.record_at(device, encoder, (source, 0), bytes, tag)
    }

    pub fn record_at(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        (source, offset): (&wgpu::Buffer, u64),
        bytes: u64,
        tag: T,
    ) -> bool {
        let mut slots = self.slots.borrow_mut();
        let index = match slots
            .iter()
            .position(|slot| slot.state.get() == State::Idle)
        {
            Some(index) => index,
            None if slots.len() < READBACKS => {
                slots.push(Slot {
                    buffer: crate::memory::buffer(
                        device,
                        &wgpu::BufferDescriptor {
                            label: Some(self.label),
                            size: bytes,
                            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        },
                    ),
                    state: Rc::new(Cell::new(State::Idle)),
                    tag,
                });
                slots.len() - 1
            }
            None => return false,
        };
        let slot = &mut slots[index];
        encoder.copy_buffer_to_buffer(source, offset, &slot.buffer, 0, bytes);
        slot.state.set(State::Recorded);
        slot.tag = tag;
        true
    }

    pub fn after_submit(&self, read: impl Fn(&Words<'_>, T) + Clone + 'static) {
        for slot in self.slots.borrow().iter() {
            if slot.state.get() != State::Recorded {
                continue;
            }
            slot.state.set(State::Pending);
            let state = Rc::clone(&slot.state);
            let buffer = slot.buffer.clone();
            let (read, tag) = (read.clone(), slot.tag);
            slot.buffer.slice(..).map_async(
                wgpu::MapMode::Read,
                crate::platform::map_callback(move |result| {
                    if result.is_ok() {
                        if let Ok(view) = buffer.slice(..).get_mapped_range() {
                            read(&Words(&view), tag);
                        }
                        buffer.unmap();
                    }
                    state.set(State::Idle);
                }),
            );
        }
    }
}
