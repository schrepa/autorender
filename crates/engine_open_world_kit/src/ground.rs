use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::prelude::{Entity, Resource};
use engine_heightfield::tile::{
    BIOME_WORDS, H_BIOMES, H_EROSION, H_FADE, H_HYDROLOGY, H_MATERIALS, H_PATHS, H_PROGRAM,
    H_READS, H_STACK, H_WET_BIOMES, MAX_MATERIALS, MAX_STACK, OP_ADD, OP_CONSTANT, OP_CURVATURE,
    OP_DEPOSIT, OP_END, OP_FACING, OP_FLOOD, OP_FLOW, OP_GRADIENT, OP_GRAIN, OP_HEIGHT, OP_INVERT,
    OP_LAKE, OP_MAX, OP_MIN, OP_MULTIPLY, OP_NOISE, OP_PATH, OP_POWER, OP_REMAP, OP_RIVER,
    OP_VALUE, OP_WETNESS, OP_WORN, Placed, Steep,
};
use engine_place::{
    Feed, GeometryHandle, List, MaterialHandle, Slot, Stamp, TextureHandle, UNIFORM_BINDING_BYTES,
    gpu_struct,
};
use engine_tiles::{Pyramid, TILE_QUADS, TileKey};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

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

pub const PALETTE_ENTRIES: u32 = (UNIFORM_BINDING_BYTES / size_of::<PaletteEntry>() as u64) as u32;

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceProjection {
    #[default]
    Top,
    Biplanar,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceMaterial {
    pub name: Arc<str>,
    pub material: Option<MaterialHandle>,
    pub color: [f32; 3],
    pub roughness: f32,
    pub tiling: f32,
    pub height_contrast: f32,
    pub projection: SurfaceProjection,
    pub depth: f32,
}

impl Default for SurfaceMaterial {
    fn default() -> Self {
        Self {
            name: Arc::from(""),
            material: None,
            color: [0.5; 3],
            roughness: 0.9,
            tiling: 4.0,
            height_contrast: 0.5,
            projection: SurfaceProjection::Top,
            depth: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceMaps {
    pub material: MaterialHandle,
    pub base_color: Option<TextureHandle>,
    pub normal: Option<TextureHandle>,
    pub metallic_roughness: Option<TextureHandle>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacedAsset {
    pub mesh: GeometryHandle,
    pub material: MaterialHandle,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlacedLayer {
    pub density: f32,
    pub scale: [f32; 2],
    pub stands_on: Option<[u64; 4]>,
    pub assets: Vec<(u16, f32)>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TileMaterials {
    #[default]
    Baked,
    Program,
}

#[derive(Clone, Debug)]
pub struct TileFace {
    pub key: TileKey,
    pub stamp: Stamp,
    pub bounds: (Vec3, Vec3),
    pub level: u8,
    pub spacing: f32,
    pub heights: Arc<[f32]>,
    pub holes: Option<Arc<[u64]>>,
    pub materials: TileMaterials,
    pub control: Option<Arc<[u32]>>,
    pub river_edges: Option<Arc<[u16]>>,
    pub river_beds: Option<Arc<[u16]>>,
    pub patch_bounds: Arc<[[f32; 2]]>,
    pub patch_holes: [u64; 4],
    pub ground_errors: Arc<[u16]>,
    pub stamps: Arc<[engine_heightfield::Stamp]>,
    pub placed: Arc<[Placed]>,
    pub current: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileChangeKind {
    Arrived,
    Replaced,
    Left,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileChange {
    pub key: TileKey,
    pub kind: TileChangeKind,
}

#[derive(Clone, Debug)]
pub struct TerrainEntry {
    pub terrain: Entity,
    pub origin: Vec3,
    pub pyramid: Pyramid,
    pub cutout: Option<(Vec2, Vec2)>,
    pub surroundings: bool,
    pub steep: Option<Steep>,
    pub may_hole: bool,
    pub reach: Arc<[f32]>,
    pub recipe: Option<[u8; 32]>,
    pub edits: Arc<[engine_heightfield::Stamp]>,
    pub solids: Arc<[Option<engine_heightfield::tile::Solid>]>,
    pub relief: u128,
    pub pinned_levels: u32,
    pub pixel_error: f32,
    pub level_errors: Arc<[f32]>,
    pub surface_program: bool,
    pub baked: Option<Arc<engine_heightfield::Layers>>,
    pub baking: bool,
    pub macro_map: Option<TextureHandle>,
    pub materials: Arc<[SurfaceMaterial]>,
    pub maps: Arc<[Option<SurfaceMaps>]>,
    pub placement: Arc<[PlacedAsset]>,
    pub layers: Arc<[PlacedLayer]>,
    pub tiles: Arc<[TileFace]>,
    pub changes: Arc<Feed<TileChange>>,
    pub sample_offset: Vec2,
    pub kept: Arc<[(TileKey, Stamp)]>,
}

#[derive(Resource, Clone)]
pub struct TerrainTiles {
    pub terrains: Arc<[TerrainEntry]>,
}

impl Default for TerrainTiles {
    fn default() -> Self {
        Self {
            terrains: Arc::from(Vec::new()),
        }
    }
}

impl Slot for TerrainTiles {
    const NAME: &'static str = "terrain_tiles";
}

#[derive(Resource, Default)]
pub struct GroundSurface;

impl Slot for GroundSurface {
    const NAME: &'static str = "ground_surface";
    const WGSL: Option<&'static str> = Some(engine_places::contract!(
        "places/ground_surface.wgsl",
        OP_CONSTANT,
        OP_HEIGHT,
        OP_GRADIENT,
        OP_CURVATURE,
        OP_FLOW,
        OP_WETNESS,
        OP_LAKE,
        OP_RIVER,
        OP_PATH,
        OP_NOISE,
        OP_MULTIPLY,
        OP_ADD,
        OP_MIN,
        OP_MAX,
        OP_INVERT,
        OP_REMAP,
        OP_GRAIN,
        OP_POWER,
        OP_FLOOD,
        OP_WORN,
        OP_DEPOSIT,
        OP_FACING,
        OP_VALUE,
        OP_END,
        H_PROGRAM,
        H_STACK,
        H_BIOMES,
        H_MATERIALS,
        H_READS,
        H_HYDROLOGY,
        H_PATHS,
        H_FADE,
        H_WET_BIOMES,
        H_EROSION,
        BIOME_WORDS,
        MAX_STACK,
        MAX_MATERIALS,
    ));
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct Heightfield {
    pub leaf_indices: Arc<[u32]>,
    pub subdivision_error: f32,
    pub water_leaves: [u32; 2],
}

impl Slot for Heightfield {
    const NAME: &'static str = "heightfield";
    const WGSL: Option<&'static str> = Some(include_str!("places/heightfield.wgsl"));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroundBake {
    Off,
    Asked,
    Made,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroundBakes {
    pub terrain: Entity,
    pub lighting: GroundBake,
    pub bounce: GroundBake,
}

#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct GroundBaking {
    pub terrains: Vec<GroundBakes>,
}

impl Slot for GroundBaking {
    const NAME: &'static str = "ground_baking";
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeldTile {
    pub terrain: Entity,
    pub key: TileKey,
    pub stamp: Stamp,
    pub layer: u32,
    pub top: f32,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct GroundTiles {
    pub held: Vec<HeldTile>,
    pub layers: u32,
    pub tables: Vec<(Entity, u32)>,
    pub palette: Vec<(Entity, Option<u32>)>,
}

impl Slot for GroundTiles {
    const NAME: &'static str = "ground_tiles";
    const WGSL: Option<&'static str> = Some(engine_places::contract!(
        "places/ground_tiles.wgsl",
        TILE_QUADS,
        PALETTE_ENTRIES,
    ));
}

#[derive(Resource, Default)]
pub struct Horizons;

impl Slot for Horizons {
    const NAME: &'static str = "horizons";
    const WGSL: Option<&'static str> = Some(include_str!("places/horizons.wgsl"));
}

#[derive(Resource, Default)]
pub struct Bounce;

impl Slot for Bounce {
    const NAME: &'static str = "bounce";
    const WGSL: Option<&'static str> = Some(include_str!("places/bounce.wgsl"));
}

#[derive(Clone, Debug, PartialEq)]
pub struct PooledTile {
    pub terrain: Entity,
    pub key: TileKey,
    pub stamp: Stamp,
    pub slots: Range<u32>,
    pub identity: u64,
    pub impostors: Vec<(Vec3, Vec3)>,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct PlacedTiles {
    pub tiles: Vec<PooledTile>,
    pub groups: Vec<([f32; 4], [f32; 3])>,
}

impl Slot for PlacedTiles {
    const NAME: &'static str = "placed_tiles";
    const WGSL: Option<&'static str> = Some(include_str!("places/placed_tiles.wgsl"));
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct Cover {
    pub density: f32,
    pub widths: Vec<(Entity, f32)>,
}

impl Cover {
    pub fn width_of(&self, terrain: Entity) -> f32 {
        self.widths
            .iter()
            .find(|(covered, _)| *covered == terrain)
            .map_or(0.0, |&(_, width)| width)
    }
}

impl Slot for Cover {
    const NAME: &'static str = "cover";
    const WGSL: Option<&'static str> = Some(include_str!("places/cover.wgsl"));
}

pub struct FarCover;

impl List for FarCover {
    const NAME: &'static str = "far_cover";
    type Member = ();
    const WGSL: Option<&'static str> = Some(engine_places::contract!(
        "places/far_cover.wgsl",
        PALETTE_ENTRIES
    ));
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct Foliage {
    pub ground_blend: f32,
}

impl Slot for Foliage {
    const NAME: &'static str = "foliage";
    const WGSL: Option<&'static str> = Some(include_str!("places/foliage.wgsl"));
}
