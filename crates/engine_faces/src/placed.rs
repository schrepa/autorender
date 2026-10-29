use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use engine_core::TileKey;
use engine_render::gpu_layouts::impostors::ImpostorRecord;
use glam::Vec3;

pub const DEAD: u32 = u32::MAX;

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub struct PlacedRecord as ["Record"] {
        pub center: [f32; 3],
        pub radius: f32,
        pub group: u32,
        pub scale: f32,
        pub terrain: u32,
        pub sink: f32,
        pub state: u32,
        pub pad: [u32; 3],
    }
}

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
