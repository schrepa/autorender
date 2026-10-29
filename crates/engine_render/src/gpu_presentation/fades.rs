use engine_core::PoseRows;

use super::storage;
use crate::gpu_layouts::constants::CARD_VERTICES;

pub(super) const FADE_MARGIN: f32 = 0.1;

pub(super) struct Fades {
    pub(super) states: wgpu::Buffer,
    pub(super) cards: wgpu::Buffer,
    slots: u64,
    card_capacity: usize,
    generation: u64,
}

impl Fades {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            states: storage(
                device,
                "engine/presentation-lod-states",
                0,
                wgpu::BufferUsages::COPY_SRC,
            ),
            cards: cards(device, 0),
            slots: 0,
            card_capacity: 0,
            generation: 0,
        }
    }

    pub(super) fn ensure(
        &mut self,
        device: &wgpu::Device,
        poses: &wgpu::Buffer,
        candidates: usize,
    ) {
        let slots = poses.size() / size_of::<PoseRows>() as u64;
        if slots > self.slots {
            crate::memory::release_buffer(&self.states);
            self.states = storage(
                device,
                "engine/presentation-lod-states",
                slots * 4,
                wgpu::BufferUsages::COPY_SRC,
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

    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) fn begin_frame(&self, queue: &crate::gpu_calls::Queue) {
        queue.write_buffer(
            &self.cards,
            0,
            bytemuck::cast_slice(&[CARD_VERTICES, 0, 0, 0]),
        );
    }

    pub(super) fn source(&self) -> (wgpu::BufferSlice<'_>, &wgpu::Buffer, u64) {
        (self.cards.slice(16..), &self.cards, 0)
    }
}

fn cards(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    storage(
        device,
        "engine/presentation-fading-cards",
        16 + capacity as u64 * 8,
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
    )
}
