use std::ops::Range;

mod atlas;
mod part;
mod patches;
mod pipelines;
mod surface;
mod tables;

pub(crate) use part::TerrainGpu;
pub(crate) use patches::planes;
pub(crate) use surface::meshlet_indices;

const LAYER_SIZE: u32 = crate::gpu::TILE_STRIDE;
const SLOT: u64 = 256;
const VIEWS: u64 = 7;
const SURFACE_SLOTS: usize = 8;
const NORMAL_FORMAT: engine_render::wgpu::TextureFormat =
    engine_render::wgpu::TextureFormat::R32Uint;
const CONTROL_FORMAT: engine_render::wgpu::TextureFormat =
    engine_render::wgpu::TextureFormat::R32Uint;
const RIVER_FORMAT: engine_render::wgpu::TextureFormat =
    engine_render::wgpu::TextureFormat::Rgba16Uint;
const IDS_FORMAT: engine_render::wgpu::TextureFormat = engine_render::wgpu::TextureFormat::Rg32Uint;

const ARRIVAL_SECONDS: f32 = 0.4;

const NO_HOLES: u32 = u32::MAX;
const HOLE_WORDS: usize = ((LAYER_SIZE - 2) * (LAYER_SIZE - 2)).div_ceil(32) as usize;

fn widen(range: Range<usize>, index: usize) -> Range<usize> {
    if range.is_empty() {
        index..index + 1
    } else {
        range.start.min(index)..range.end.max(index + 1)
    }
}

fn changed_run<T: bytemuck::Pod>(was: &[T], now: &[T]) -> Range<usize> {
    let same = |a: &T, b: &T| bytemuck::bytes_of(a) == bytemuck::bytes_of(b);
    let shared = now.len().min(was.len());
    let Some(first) = (0..now.len()).find(|&at| at >= shared || !same(&now[at], &was[at])) else {
        return 0..0;
    };
    let end = if now.len() > was.len() {
        now.len()
    } else {
        (first..now.len())
            .rev()
            .find(|&at| !same(&now[at], &was[at]))
            .map_or(first + 1, |at| at + 1)
    };
    first..end
}

pub(crate) struct FrameTerrain {
    pub terrain: usize,
    pub holes: bool,
    pub maps: Vec<(usize, Range<u32>)>,
    pub screen: Option<[f32; 4]>,
}

impl FrameTerrain {
    pub(crate) fn scissor(&self, size: (u32, u32)) -> Option<[u32; 4]> {
        let (width, height) = (size.0 as f32, size.1 as f32);
        let Some([left, bottom, right, top]) = self.screen else {
            return (size.0 > 0 && size.1 > 0).then_some([0, 0, size.0, size.1]);
        };
        let pixel = |value: f32, extent: f32| engine_math::clamp(value, 0.0, extent) as u32;
        let x0 = pixel(((left * 0.5 + 0.5) * width).floor() - 1.0, width);
        let x1 = pixel(((right * 0.5 + 0.5) * width).ceil() + 1.0, width);
        let y0 = pixel(((0.5 - top * 0.5) * height).floor() - 1.0, height);
        let y1 = pixel(((0.5 - bottom * 0.5) * height).ceil() + 1.0, height);
        (x1 > x0 && y1 > y0).then_some([x0, y0, x1 - x0, y1 - y0])
    }
}

#[derive(Default)]
pub(crate) struct TerrainFrame {
    pub terrains: Vec<FrameTerrain>,
    pub shadow_patches: u32,
}

impl TerrainFrame {
    pub(crate) fn map_records(&self, map: usize) -> Option<Range<u32>> {
        self.terrains
            .iter()
            .filter_map(|terrain| terrain.maps.iter().find(|(drawn, _)| *drawn == map))
            .map(|(_, range)| range.clone())
            .filter(|range| !range.is_empty())
            .reduce(|all, range| all.start.min(range.start)..all.end.max(range.end))
    }
}
