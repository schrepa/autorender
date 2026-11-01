use bevy_ecs::prelude::*;
use engine_core::{gpu_consts, gpu_struct};

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
    pub const SUBDIVISION_EPOCH: u32 = 12;
    pub const HORIZON_DIRECTIONS: u32 = 16;
    pub const HORIZON_LAYERS: u32 = 9;
    pub const HORIZON_RECORD_VECTORS: u32 = 6;
    pub const HORIZON_RECORDS: u32 = 384;
    pub const BOUNCE_BINS: u32 = 26;
    pub const BOUNCE_PROBE_LAYERS: u32 = 2;
    pub const BOUNCE_RECORD_WORDS: u32 = 26;
    pub const BOUNCE_LOW: f32 = 1.5;
    pub const BOUNCE_HIGH: f32 = 24.0;
    pub const BOUNCE_WINDOW_ROW: u32 = 4096;
    pub const BOUNCE_NEAREST: f32 = 0.5;
    pub const BOUNCE_STEPS: f32 = 16.0;
    pub const TILE_STRIDE: u32 = 259;
    pub const CONTROL_GROUP: u32 = 8;
    pub const BOUNCE_GROUP: u32 = 8;
    pub const NORMALS_GROUP: u32 = 16;
    pub const DEBUG_TILE_EDGE: u32 = 0x800;
    pub const DEBUG_PATCH_EDGE: u32 = 0x1000;
    pub const DEBUG_PATCH_SHIFT: u32 = 24;
}

pub const HEIGHT_FORMAT: engine_render::wgpu::TextureFormat = engine_render::wgpu::TextureFormat::R32Float;

pub const PALETTE_BYTES: usize = engine_places::ground::PALETTE_ENTRIES as usize * size_of::<PaletteEntry>();

const _: () = assert!(
    engine_places::ground::PALETTE_ENTRIES as usize * size_of::<PaletteEntry>() <= 16_384,
    "palette_fits_a_uniform_binding"
);

const _: () = assert!(
    SUBDIVISION_MEASURE > SUBDIVISION_DRAW + 4
        && SUBDIVISION_MEASURE != SUBDIVISION_RESERVED
        && SUBDIVISION_MEASURE != SUBDIVISION_REFUSED
        && SUBDIVISION_EPOCH > SUBDIVISION_MEASURE
        && SUBDIVISION_EPOCH < SUBDIVISION_ARGS,
    "subdivision_counters_keep_their_words"
);

const _: () = assert!(
    HORIZON_RECORDS == 64 * HORIZON_RECORD_VECTORS
        && HORIZON_RECORD_VECTORS == 4 + crate::settings::MAX_BOUNCE_LEVELS / 4,
    "horizon_records_hold_the_bounce_levels"
);

const _: () = assert!(
    HORIZON_DIRECTIONS as usize == crate::lighting::DIRECTIONS
        && HORIZON_LAYERS as usize == crate::lighting::HORIZON_LAYERS
        && BOUNCE_BINS as usize == crate::bounce::BINS
        && BOUNCE_PROBE_LAYERS as usize == crate::bounce::LAYERS
        && BOUNCE_RECORD_WORDS as usize == crate::bounce::PROBE_BYTES / 4
        && BOUNCE_WINDOW_ROW == crate::bounce::MAX_PROBES,
    "bake_shapes_match_their_shaders"
);

const _: () = assert!(
    TILE_STRIDE == engine_heightfield::TILE_SAMPLES + 2
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

gpu_struct! {
    #[derive(Resource)]
    pub struct HorizonsPiece {
        pub records: [[f32; 4]; 385],
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
    pub struct PaletteEntry {
        pub color: [f32; 4],
        pub params: [f32; 4],
        pub maps: [f32; 4],
        pub far: [f32; 4],
        pub relief: [f32; 4],
        pub heights: [f32; 4],
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
