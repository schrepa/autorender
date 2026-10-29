use engine_core::{Pyramid, RenderSnapshot, TileKey};
use engine_render::cached::Bound;
use glam::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct GroundChange {
    pub min: Vec3,
    pub max: Vec3,
    pub pyramid: Pyramid,
    pub level: u32,
}

pub struct CoverSources<'a> {
    pub heights: &'a wgpu::TextureView,
    pub control: &'a wgpu::TextureView,
    pub normals: &'a wgpu::TextureView,
    pub directory: &'a wgpu::Buffer,
    pub palette: &'a wgpu::Buffer,
    pub covers: &'a wgpu::Buffer,
    pub macro_maps: &'a wgpu::TextureView,
    pub generation: (u64, u64, u64),
    pub revision: u64,
}

pub struct TraceSources<'a> {
    pub arrays: Option<[Bound<'a>; 2]>,
    pub directory: Bound<'a>,
    pub palette: Bound<'a>,
    pub macro_maps: Bound<'a>,
    pub sky: Bound<'a>,
}

pub struct LiveGround<'a> {
    pub group: &'a wgpu::BindGroup,
    pub uniform: u32,
    pub terrain: u64,
}

pub struct SurfaceDraw<'a> {
    pub indices: &'a wgpu::Buffer,
    pub args: &'a wgpu::Buffer,
    pub offset: u64,
}

pub struct WaterDraws<'a> {
    pub heights: &'a wgpu::TextureView,
    pub views: &'a wgpu::Buffer,
    pub uniforms: &'a wgpu::Buffer,
    pub directory: &'a wgpu::Buffer,
    pub leaves: &'a wgpu::Buffer,
    pub trees: &'a wgpu::Buffer,
    pub surfaces: &'a wgpu::TextureView,
    pub info: &'a wgpu::TextureView,
    pub generation: [u64; 6],
    pub terrains: Vec<(u64, u32, u32, SurfaceDraw<'a>)>,
    pub targets: [f32; 2],
    pub blocks: [u32; 2],
}

pub trait HeightfieldFace {
    fn drawn(&self) -> bool;

    fn ground_changes(&self) -> &[GroundChange];

    fn cover_sources(&self) -> Option<CoverSources<'_>>;

    fn cover_terrain(&self, index: usize) -> (u32, u32, u32);

    fn cover_tallest(&self) -> f32;

    fn trace_sources(&self) -> TraceSources<'_>;

    fn ground(&self, terrain: u64) -> Option<LiveGround<'_>>;

    fn holds(&self, terrain: u64, key: TileKey, tile: *const ()) -> bool;

    fn water_draws(&self, snapshot: &RenderSnapshot) -> Option<WaterDraws<'_>>;

    fn layer_of(&self, terrain: u64, key: TileKey) -> Option<u32>;

    fn layers(&self) -> (u32, u64);

    fn highest(&self, terrain: u64) -> Option<f32>;
}
