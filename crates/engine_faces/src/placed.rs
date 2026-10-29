use std::ops::Range;

use engine_core::TileKey;
use engine_render::gpu_layouts::impostors::ImpostorRecord;
use glam::Vec3;

pub const DEAD: u32 = u32::MAX;

pub struct Placements<'a> {
    pub tiles: Vec<((u64, TileKey), Range<u32>, usize)>,
    pub records: &'a [u8],
    pub poses: &'a [u8],
    pub impostors: &'a [u8],
    pub colors: &'a [[u32; 2]],
    pub revision: u64,
}

pub trait PlacedFace {
    fn moved(&self) -> &[(Vec3, Vec3)];

    fn restyled(&self) -> bool;

    fn impostors(&self) -> &[ImpostorRecord];

    fn placements(&self) -> Placements<'_>;

    fn slot_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer, u64);

    fn frames(&self) -> Option<(&wgpu::TextureView, u64)>;
}
