//! Levels of detail kept and faded (decision 107): each slot's level as last
//! drawn, which culling's level gives way to only a margin past its
//! distance, and the cards fading in or out over their steady meshes, drawn
//! in one draw.

use engine_core::PoseRows;

use super::storage;

/// The share of its distance past a switch the eye goes before a candidate
/// switches back (`presentation.wgsl`'s `fade_levels`).
pub(super) const FADE_MARGIN: f32 = 0.1;

/// Vertices of an impostor card: two triangles.
const CARD_VERTICES: u32 = 6;

/// Each slot's level state, and the fading cards' draw arguments and
/// instances.
pub(super) struct Fades {
    pub(super) states: wgpu::Buffer,
    pub(super) cards: wgpu::Buffer,
    // Slots the states cover and cards the instances hold.
    slots: u64,
    card_capacity: usize,
    // Advances when either is replaced.
    generation: u64,
}

impl Fades {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            states: storage(
                device,
                "engine/presentation-lod-states",
                0,
                wgpu::BufferUsages::empty(),
            ),
            cards: cards(device, 0),
            slots: 0,
            card_capacity: 0,
            generation: 0,
        }
    }

    /// Keeps a state for every slot of `poses` and room for a fading card
    /// per candidate of `candidates`; replaced states start over, zeroed.
    pub(super) fn ensure(
        &mut self,
        device: &wgpu::Device,
        poses: &wgpu::Buffer,
        candidates: usize,
    ) {
        let slots = poses.size() / size_of::<PoseRows>() as u64;
        if slots > self.slots {
            self.states = storage(
                device,
                "engine/presentation-lod-states",
                slots * 4,
                wgpu::BufferUsages::empty(),
            );
            self.slots = slots;
            self.generation += 1;
        }
        if candidates > self.card_capacity {
            self.cards = cards(device, candidates);
            self.card_capacity = candidates;
            self.generation += 1;
        }
    }

    /// Advances when the storage is replaced.
    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    /// Starts this frame's fading cards at none.
    pub(super) fn begin_frame(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.cards,
            0,
            bytemuck::cast_slice(&[CARD_VERTICES, 0, 0, 0]),
        );
    }

    /// The fading cards' instances (each one's slot and batch word) and the
    /// buffer and offset of their draw's arguments.
    pub(super) fn source(&self) -> (wgpu::BufferSlice<'_>, &wgpu::Buffer, u64) {
        (self.cards.slice(16..), &self.cards, 0)
    }
}

/// The fading cards' storage: a draw's arguments, then `capacity`
/// instances of two words.
fn cards(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    storage(
        device,
        "engine/presentation-fading-cards",
        16 + capacity as u64 * 8,
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
    )
}
