//! Buffers every pass shares the shape of: storage buffers and their
//! whole-buffer bindings, growth by powers of two within what one binding
//! holds, and values read back a few frames late.

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

/// A storage buffer the queue writes, of at least `bytes` rounded up to a
/// multiple of 16 (and at least 16), with `extra` usages.
pub(crate) fn storage(
    device: &wgpu::Device,
    label: &str,
    bytes: u64,
    extra: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16).next_multiple_of(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
        mapped_at_creation: false,
    })
}

/// A uniform buffer of `bytes` the queue writes.
pub(crate) fn uniform(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// A [`storage`] buffer bound whole whose content sizes it: it grows by
/// powers of two to hold what it is asked to, but never past what one
/// storage binding of its device holds, its room. The one way such
/// buffers grow; what is asked past the room is the caller's to leave out
/// and count.
pub(crate) struct Growable {
    buffer: crate::cached::Versioned<wgpu::Buffer>,
    label: &'static str,
    extra: wgpu::BufferUsages,
    room: u64,
}

impl std::ops::Deref for Growable {
    type Target = crate::cached::Versioned<wgpu::Buffer>;

    fn deref(&self) -> &crate::cached::Versioned<wgpu::Buffer> {
        &self.buffer
    }
}

impl Growable {
    /// A [`storage`] buffer of `bytes` labeled `label`, with `extra`
    /// usages, on `device`.
    pub(crate) fn new(
        device: &wgpu::Device,
        label: &'static str,
        bytes: u64,
        extra: wgpu::BufferUsages,
    ) -> Self {
        let limits = device.limits();
        // Sizes are whole 16-byte words.
        let room = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            / 16
            * 16;
        Self {
            buffer: crate::cached::Versioned::new(storage(device, label, bytes.min(room), extra)),
            label,
            extra,
            room,
        }
    }

    /// Grows to hold `bytes`, to the next power of two of them within its
    /// room, unless it holds them already, advancing its generation;
    /// returns whether it did. The new buffer holds nothing of the old one.
    pub(crate) fn grow(&mut self, device: &wgpu::Device, bytes: u64) -> bool {
        let size = bytes.next_power_of_two().min(self.room);
        if bytes <= self.buffer.size() || size <= self.buffer.size() {
            return false;
        }
        self.buffer
            .replace(storage(device, self.label, size, self.extra));
        true
    }

    /// The most bytes it holds: what one storage binding of its device
    /// holds, in whole 16-byte words.
    pub(crate) fn room(&self) -> u64 {
        self.room
    }
}

/// A buffer bound whole as binding `name`.
pub(crate) fn entry<'a>(
    name: &'a str,
    buffer: &'a wgpu::Buffer,
) -> (&'a str, wgpu::BindingResource<'a>) {
    (name, buffer.as_entire_binding())
}

/// Read-backs in flight at most: a frame records one while the two before
/// it may still be mapping.
pub(crate) const READBACKS: usize = 3;

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

/// A mapped read-back's little-endian words; words past its end read 0.
pub(crate) struct Words<'a>(&'a [u8]);

impl Words<'_> {
    pub(crate) fn get(&self, index: usize) -> u32 {
        self.0.get(index * 4..index * 4 + 4).map_or(0, |bytes| {
            u32::from_le_bytes(bytes.try_into().unwrap_or([0; 4]))
        })
    }
}

/// Copies of one GPU buffer read back a few frames late, through at most
/// three buffers: [`Self::record`] copies into an idle one while a frame
/// records, [`Self::after_submit`] maps those the frame just submitted, and
/// each map hands its words and the tag recorded with them to a reader.
/// Nothing waits on the GPU; frames finding every buffer in flight skip.
pub(crate) struct ReadbackRing<T> {
    label: &'static str,
    // Recorded into while a frame encodes, which borrows passes shared.
    slots: RefCell<Vec<Slot<T>>>,
}

impl<T: Copy + 'static> ReadbackRing<T> {
    pub(crate) fn new(label: &'static str) -> Self {
        Self {
            label,
            slots: RefCell::new(Vec::new()),
        }
    }

    /// Records a copy of the first `bytes` of `source`, tagged `tag`, into
    /// an idle buffer, creating one of `bytes` while fewer than three exist;
    /// returns whether it did. Clear the ring when `bytes` changes.
    pub(crate) fn record(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        bytes: u64,
        tag: T,
    ) -> bool {
        self.record_at(device, encoder, (source, 0), bytes, tag)
    }

    /// [`Self::record`] of `bytes` from byte `offset` of `source`.
    pub(crate) fn record_at(
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
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(self.label),
                        size: bytes,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
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

    /// Requests maps of the read-backs recorded in the frame just submitted;
    /// `read` receives each one's words and tag once it is mapped.
    pub(crate) fn after_submit(&self, read: impl Fn(&Words<'_>, T) + Clone + 'static) {
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

    /// Bytes actually allocated for readback slots.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.slots
            .borrow()
            .iter()
            .map(|slot| slot.buffer.size())
            .sum()
    }

    /// Drops the buffers; maps in flight still complete.
    pub(crate) fn clear(&mut self) {
        self.slots.get_mut().clear();
    }
}
