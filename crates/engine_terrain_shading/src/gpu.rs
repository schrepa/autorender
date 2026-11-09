use bevy_ecs::prelude::*;
use engine_core::{GpuBits, GpuStruct, gpu_bits, gpu_consts, gpu_struct};
use engine_places_world::ground::PaletteEntry;

use crate::jobs::gpu::{BounceBin, BounceBlockWord, TILE_STRIDE};
use crate::jobs::{bounce, lighting};

gpu_consts! {
    pub const NOT_RESIDENT: u32 = 0xffffffff;
    pub const PATCH_GRID: u32 = 17;
    pub const PATCH_GRID_VERTICES: u32 = PATCH_GRID * PATCH_GRID;
    pub const SURFACE_MESHLET_LEVEL: u32 = 2;
    pub const SURFACE_MESHLET_VERTICES: u32 = 15;
    pub const SURFACE_MESHLET_INDICES: u32 = 48;
    pub const SURFACE_VERTEX_STRIDE: u32 = 16;
    pub const SURFACE_ERROR_MIPS: u32 = 8;
    pub const SUBDIVISION_GROUP: u32 = 64;
    pub const SUBDIVISION_ARGS: u32 = 16;
    pub const SUBDIVISION_LEAVES: u32 = 3;
    pub const SUBDIVISION_DRAW: u32 = 4;
    pub const SUBDIVISION_RESERVED: u32 = 9;
    pub const SUBDIVISION_REFUSED: u32 = 10;
    pub const SUBDIVISION_COUNTED: u32 = 8;
    pub const SUBDIVISION_MEASURE: u32 = 11;

    pub const HORIZON_DIRECTIONS: u32 = 16;
    pub const HORIZON_LAYERS: u32 = 9;
    pub const HORIZON_RECORD_VECTORS: u32 = 6;
    pub const HORIZON_RECORDS: u32 = 384;
    pub const BOUNCE_BINS: u32 = 26;
    pub const BOUNCE_PROBE_LAYERS: u32 = 2;
    pub const BOUNCE_RECORD_WORDS: u32 = 26;
    pub const BOUNCE_WINDOW_ROW: u32 = 4096;
    pub const CONTROL_GROUP: u32 = 8;
    pub const BOUNCE_GROUP: u32 = 8;
    pub const NORMALS_GROUP: u32 = 16;
    pub const PEAKS_GROUP: u32 = 16;
    pub const PEAKS_SIDE: u32 = 128;
    pub const ANCESTORS_GROUP: u32 = 16;
    pub const NORMAL_PLANE: u32 = 0;
    pub const CONTROL_PLANE: u32 = 1;
    pub const TILE_PLANES: u32 = 2;
    pub const RIVER_PLANE_FAR: u32 = 0xffff;
    pub const ARRIVAL_SECONDS: f32 = 0.4;
    pub const DEBUG_SHOWN_TILES: u32 = 1;
    pub const DEBUG_SHOWN_MATERIALS: u32 = 2;
    pub const DEBUG_SHOWN_SKY: u32 = 3;
    pub const DEBUG_SHOWN_DETAIL: u32 = 4;
    pub const TABLE_TERRAINS: u32 = 0;
    pub const TABLE_VIEWS: u32 = 1;
    pub const TABLE_FADES: u32 = 2;
    pub const TABLE_NOW: u32 = 3;
    pub const TERRAIN_WORDS: u32 = 44;
    pub const VIEW_WORDS: u32 = 36;
    pub const NO_PLACED: u32 = u32::MAX;
    pub const CANOPY_GROUP: u32 = 64;
    pub const DETAIL_ANISOTROPY: f32 = 16.0;
}

const _: () = assert!(
    DETAIL_ANISOTROPY.to_bits() == (engine_core::Sampler::ANISOTROPY as f32).to_bits(),
    "DETAIL_ANISOTROPY differs from Sampler::ANISOTROPY"
);

pub const HEIGHT_FORMAT: engine_render::wgpu::TextureFormat =
    engine_render::wgpu::TextureFormat::R32Float;

pub const PALETTE_BYTES: usize =
    engine_places_world::ground::PALETTE_ENTRIES as usize * size_of::<PaletteEntry>();

const _: () = assert!(
    SUBDIVISION_MEASURE > SUBDIVISION_DRAW + 4
        && SUBDIVISION_MEASURE != SUBDIVISION_RESERVED
        && SUBDIVISION_MEASURE != SUBDIVISION_REFUSED
        && SUBDIVISION_MEASURE < SUBDIVISION_ARGS,
    "subdivision_counters_keep_their_words"
);

const _: () = assert!(
    HORIZON_RECORDS == 64 * HORIZON_RECORD_VECTORS
        && HORIZON_RECORD_VECTORS == 4 + crate::settings::MAX_BOUNCE_LEVELS / 4,
    "horizon_records_hold_the_bounce_levels"
);

const _: () = assert!(
    HORIZON_DIRECTIONS as usize == lighting::DIRECTIONS
        && HORIZON_LAYERS as usize == lighting::HORIZON_LAYERS
        && BOUNCE_BINS as usize == bounce::BINS
        && BOUNCE_PROBE_LAYERS as usize == bounce::LAYERS
        && BOUNCE_RECORD_WORDS as usize == bounce::PROBE_BYTES / 4
        && BOUNCE_WINDOW_ROW == bounce::MAX_PROBES,
    "bake_shapes_match_their_shaders"
);

const _: () = assert!(
    TERRAIN_WORDS as usize * 4 == size_of::<TerrainUniform>()
        && VIEW_WORDS as usize * 4 == size_of::<ViewUniform>()
        && TABLE_TERRAINS < TABLE_VIEWS
        && TABLE_VIEWS < TABLE_FADES
        && TABLE_FADES < TABLE_NOW,
    "terrain_tables_hold_whole_records"
);

const _: () = assert!(
    RIVER_PLANE_FAR == engine_heightfield::tile::RIVER_EDGE_FAR as u32,
    "river_planes_mean_far_the_same"
);

const _: () = assert!(
    TILE_STRIDE == engine_heightfield::TILE_SAMPLES + 2
        && PEAKS_SIDE == engine_heightfield::TILE_QUADS / 2
        && SURFACE_ERROR_MIPS == engine_heightfield::TILE_QUADS.trailing_zeros()
        && PATCH_GRID_VERTICES == PATCH_GRID * PATCH_GRID
        && SURFACE_MESHLET_VERTICES
            == ((1 << SURFACE_MESHLET_LEVEL) + 1) * ((1 << SURFACE_MESHLET_LEVEL) + 2) / 2
        && SURFACE_MESHLET_INDICES == 3 << (2 * SURFACE_MESHLET_LEVEL)
        && SURFACE_VERTEX_STRIDE > SURFACE_MESHLET_VERTICES,
    "terrain_grid_constants_agree"
);

gpu_struct! {
    pub struct PatchRecord {
        pub first: [i32; 2],
        pub shape: [f32; 2],
        pub atlas: [u32; 2],
        pub holes: u32,
        pub terrain: u32,
    }
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PatchTexel {
    #[bits(16)]
    pub x: u32,
    #[bits(16)]
    pub z: u32,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShownTile {
    #[bits(16)]
    pub x: u16,
    #[bits(16)]
    pub z: u16,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShownMaterials {
    #[bits(8)]
    pub first: u8,
    #[bits(8)]
    pub second: u8,
    #[bits(8)]
    pub third: u8,
    #[bits(8)]
    pub fourth: u8,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RiverMaps {
    pub edges: bool,
    pub beds: bool,
}

pub(crate) const GENERATED: &[&str] = &[
    <PaletteEntry as GpuStruct>::WGSL,
    <PatchTexel as GpuBits>::WGSL,
    <BounceBlockWord as GpuBits>::WGSL,
    <BounceBin as GpuBits>::WGSL,
    <ShownTile as GpuBits>::WGSL,
    <ShownMaterials as GpuBits>::WGSL,
    <RiverMaps as GpuBits>::WGSL,
    <engine_heightfield::tile::Control as GpuBits>::WGSL,
    <engine_places_rendering::shadows::SunShown as GpuBits>::WGSL,
];

gpu_struct! {
    #[derive(Resource)]
    pub struct HorizonsPiece {
        pub records: [[f32; 4]; 385],
        pub debug: [u32; 4],
        pub trace: [u32; 4],
    }
}

gpu_struct! {
    pub struct TraceTerrain {
        pub cutout: [f32; 4],
        pub maps: [u32; 4],
        pub reach: [f32; 4],
    }
}

gpu_struct! {
    pub struct BounceFrame {
        pub bins: [[f32; 4]; 26],
    }
}

gpu_struct! {
    pub struct BounceTerrain {
        pub place: [f32; 4],
        pub grid: [u32; 4],
        pub layers: [u32; 4],
        pub sources: [u32; 4],
        pub level: [u32; 4],
        pub kept: [u32; 4],
        pub finer: [u32; 4],
    }
}

gpu_struct! {
    pub struct ViewUniform {
        pub relative_view_projection: [[f32; 4]; 4],
        pub eye_offset: [f32; 4],
        pub inverse_relative_view_projection: [[f32; 4]; 4],
    }
}

gpu_struct! {
    pub struct TerrainUniform {
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
}

gpu_struct! {
    pub struct SubdivisionUniform {
        pub square: [f32; 4],
        pub tree: [u32; 4],
        pub frame: [u32; 4],
        pub extent: [f32; 4],
        pub cutout: [f32; 4],
        pub planes: [[f32; 4]; 6],
    }
}

gpu_struct! {
    pub struct NormalJob {
        pub layer: u32,
        pub spacing: u32,
        pub pad: [u32; 2],
    }
}

gpu_struct! {
    pub struct ControlJob {
        pub layer: u32,
        pub rivers: u32,
        pub first: [i32; 2],
        pub spacing: f32,
        pub terrain: u32,
        pub pad: [u32; 2],
    }
}

gpu_struct! {
    pub struct AncestorJob {
        pub layer: u32,
        pub held_layer: u32,
        pub shift: u32,
        pub plane: u32,
        pub first: [i32; 2],
        pub held_first: [i32; 2],
    }
}
