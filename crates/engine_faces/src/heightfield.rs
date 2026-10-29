use bytemuck::{Pod, Zeroable};
use engine_core::{Pyramid, RenderSnapshot, TileKey};
use engine_render::bindings::Sample::{Float, Uint, Unfilterable};
use engine_render::bindings::Sampling::{Filtering, NonFiltering};
use engine_render::bindings::View::D2Array;
use engine_render::bindings::{Stages, Table, dynamic_uniform, read, sampler, texture, uniform_of};
use engine_render::cached::Bound;
use glam::Vec3;

const C: Stages = Stages::COMPUTE;
const F: Stages = Stages::FRAGMENT;
const V: Stages = Stages::VERTEX;
const VF: Stages = Stages::VERTEX_FRAGMENT;
const FC: Stages = Stages::FRAGMENT_COMPUTE;
const VFC: Stages = Stages::VERTEX_FRAGMENT_COMPUTE;

pub const HEIGHT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

pub const PALETTE_ENTRIES: usize = 16_384 / size_of::<PaletteEntry>();

pub const PALETTE_BYTES: usize = PALETTE_ENTRIES * size_of::<PaletteEntry>();

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub struct PaletteEntry as ["TerrainPaletteEntry", "PaletteEntry" in "cover::grow", "PaletteEntry" in "terrain::bounce_maps", "PaletteEntry" in "scatter::snap", "PaletteEntry" in "post::reflections"] {
        pub color: [f32; 4],
        pub params: [f32; 4],
        pub maps: [f32; 4],
        pub far: [f32; 4],
        pub relief: [f32; 4],
        pub canopy: [f32; 4],
        pub placed: [f32; 4],
        pub placed_size: [f32; 4],
        pub heights: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub struct TerrainInfo as ["TerrainInfo" in "scatter::snap", "TerrainInfo" in "post::reflections"] {
        pub origin: [f32; 4],
        pub levels: [u32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub struct CoverEntry as ["CoverSpec"] {
        pub shape: [f32; 4],
        pub color: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub struct ViewUniform as ["TerrainView"] {
        pub relative_view_projection: [[f32; 4]; 4],
        pub eye_offset: [f32; 4],
        pub inverse_relative_view_projection: [[f32; 4]; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub struct TerrainUniform as ["TerrainUniform"] {
        pub anchor: [f32; 4],
        pub eye: [f32; 4],
        pub surface: [u32; 4],
        pub origin: [f32; 4],
        pub directory: [u32; 4],
        pub relief: [f32; 4],
        pub cover: [f32; 4],
        pub macro_map: [u32; 4],
        pub steep: [f32; 4],
        pub pixel_error: [f32; 4],
        pub water: [u32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub struct SubdivisionUniform as ["TerrainSubdivision"] {
        pub square: [f32; 4],
        pub tree: [u32; 4],
        pub frame: [u32; 4],
        pub extent: [f32; 4],
        pub cutout: [f32; 4],
        pub planes: [[f32; 4]; 6],
    }
}

pub const TERRAIN: Table = Table {
    label: "engine/terrain-layout",
    bindings: &[
        texture("terrain_heights", Unfilterable, D2Array, VFC),
        dynamic_uniform("terrain_view", size_of::<ViewUniform>(), VF),
        dynamic_uniform("terrain", size_of::<TerrainUniform>(), VFC),
        read("terrain_holes", VFC),
        texture("terrain_normals", Uint, D2Array, VFC),
        sampler("terrain_sky_sampler", Filtering, F),
        read("terrain_directory", VFC),
        texture("terrain_control", Uint, D2Array, VFC),
        uniform_of("terrain_palette", PALETTE_BYTES, VFC),
        texture("terrain_albedo", Float, D2Array, F),
        texture("terrain_material_data", Float, D2Array, VFC),
        sampler("terrain_material_sampler", Filtering, VFC),
        texture("terrain_sky", Float, D2Array, F),
        sampler("terrain_control_sampler", NonFiltering, VFC),
        texture("terrain_macro", Float, D2Array, F),
        read("terrain_surface_leaves", V),
        texture("water_surface", Unfilterable, D2Array, FC),
        texture("water_info", Uint, D2Array, C),
        texture("terrain_errors", Unfilterable, D2Array, C),
        texture("water_errors", Unfilterable, D2Array, C),
    ],
};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CoverTerms {
    pub densest: f32,
    pub width: f32,
}

impl CoverTerms {
    pub fn span(self, pixels_per_meter: f32) -> f32 {
        self.width * pixels_per_meter
    }
}

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

#[derive(Clone)]
pub struct SurfaceDraw {
    pub indices: wgpu::Buffer,
    pub args: wgpu::Buffer,
    pub offset: u64,
}

impl SurfaceDraw {
    pub fn record(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed_indirect(&self.args, self.offset);
    }
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
    pub terrains: Vec<(u64, u32, u32, SurfaceDraw)>,
    pub targets: [f32; 2],
    pub blocks: [u32; 2],
    pub subdivided: [u32; 2],
}

pub trait HeightfieldFace {
    fn drawn(&self) -> bool;

    fn ground_changes(&self) -> &[GroundChange];

    fn cover_sources(&self) -> Option<CoverSources<'_>>;

    fn cover_terrain(&self, index: usize) -> (u32, u32, u32);

    fn cover_terms(&self, index: usize) -> CoverTerms;

    fn terrain_info(&self, index: usize, origin: Vec3, pyramid: &Pyramid) -> TerrainInfo;

    fn cover_tallest(&self) -> f32;

    fn trace_sources(&self) -> TraceSources<'_>;

    fn ground(&self, terrain: u64) -> Option<LiveGround<'_>>;

    fn holds(&self, terrain: u64, key: TileKey, tile: *const ()) -> bool;

    fn water_draws(&self, snapshot: &RenderSnapshot) -> Option<WaterDraws<'_>>;

    fn layer_of(&self, terrain: u64, key: TileKey) -> Option<u32>;

    fn layers(&self) -> (u32, u64);

    fn highest(&self, terrain: u64) -> Option<f32>;

    fn sky_layer(&self, index: usize) -> u32;
}
