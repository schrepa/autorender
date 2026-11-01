use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::prelude::{Component, Entity, Resource};
use engine_math::{max, sin_cos};
use engine_place::{JobRef, List, MaterialHandle, MeshHandle, Slot, Stamp, TextureHandle};
use glam::{Mat4, Vec2, Vec3};
use serde::{Deserialize, Serialize};

pub use engine_heightfield::grid::{MAX_LEVEL, Pyramid, TILE_QUADS, TILE_SAMPLES, TileKey};

pub const PALETTE_ENTRIES: u32 = 113;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundPoint {
    pub height: f32,
    pub normal: Vec3,
    pub level: u32,
    pub hole: bool,
    pub material: Option<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundObstacle {
    pub center: Vec2,
    pub radius: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GroundRead<T> {
    Ready(T),
    Waiting,
    Off,
}

#[derive(Clone, PartialEq)]
pub struct GroundSource {
    pub ground: Entity,
    pub origin: Vec3,
    pub pyramid: Pyramid,
    pub identity: Arc<[u8]>,
}

impl std::fmt::Debug for GroundSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GroundSource")
            .field("ground", &self.ground)
            .field("origin", &self.origin)
            .field("pyramid", &self.pyramid)
            .field("identity", &self.identity.len())
            .finish()
    }
}

pub trait GroundQuery: Send + Sync + 'static {
    fn point(&self, at: Vec2) -> Option<GroundPoint>;

    fn obstacles(&self, ground: Entity) -> Vec<GroundObstacle>;

    fn source(&self, from: Vec2, to: Vec2) -> GroundRead<GroundSource>;

    fn grid(&self) -> Option<JobRef>;
}

struct NoGround;

impl GroundQuery for NoGround {
    fn point(&self, _: Vec2) -> Option<GroundPoint> {
        None
    }

    fn obstacles(&self, _: Entity) -> Vec<GroundObstacle> {
        Vec::new()
    }

    fn source(&self, _: Vec2, _: Vec2) -> GroundRead<GroundSource> {
        GroundRead::Off
    }

    fn grid(&self) -> Option<JobRef> {
        None
    }
}

#[derive(Resource, Clone)]
pub struct Ground(pub Arc<dyn GroundQuery>);

impl Ground {
    pub fn point(&self, at: Vec2) -> Option<GroundPoint> {
        self.0.point(at)
    }

    pub fn sample(&self, at: Vec2) -> Option<GroundPoint> {
        self.0.point(at).filter(|point| !point.hole)
    }
}

impl Default for Ground {
    fn default() -> Self {
        Self(Arc::new(NoGround))
    }
}

impl Slot for Ground {
    const NAME: &'static str = "ground";
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroundGrid {
    pub key: TileKey,
    pub spacing: f32,
    pub heights: Box<[f32]>,
    pub holes: Option<Box<[u64]>>,
    pub bridges: Arc<[GroundBridge]>,
    pub water: Option<Box<[f32]>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroundBridge {
    pub deck: Vec<Vec3>,
    pub width: f32,
    pub tunnel: bool,
}

engine_codec::record!(GroundGrid {
    key,
    spacing,
    heights,
    holes,
    bridges,
    water,
});

engine_codec::record!(GroundBridge {
    deck,
    width,
    tunnel,
});

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeightHeld(pub bool);

#[derive(
    Component, Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[require(GroundedAt)]
#[serde(default, deny_unknown_fields)]
pub struct Grounded {
    pub offset: f32,
}

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct GroundedAt {
    pub position: Vec2,
    pub offset: f32,
    pub revision: u64,
    pub level: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Steep {
    pub material: u8,
    pub gradient: [f32; 2],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
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
pub struct Placed {
    pub position: [f32; 3],
    pub yaw: f32,
    pub lean: [f32; 2],
    pub scale: f32,
    pub sink: f32,
    pub ground: [f32; 2],
    pub asset: u16,
    pub layer: u16,
    pub near_level: Option<u8>,
    pub rank: f32,
}

impl Placed {
    pub fn model(&self, origin: Vec3, base: f32) -> Mat4 {
        let [lean_x, lean_z] = self.lean;
        let up = Vec3::new(
            lean_x,
            max(1.0 - lean_x * lean_x - lean_z * lean_z, 0.01).sqrt(),
            lean_z,
        )
        .normalize();
        let (sin, cos) = sin_cos(self.yaw);
        let forward = Vec3::new(cos, 0.0, sin);
        let tangent = (forward - up * forward.dot(up)).normalize_or(Vec3::X);
        let across = tangent.cross(up);
        let [x, _, z] = self.position;
        Mat4::from_cols(
            (tangent * self.scale).extend(0.0),
            (up * self.scale).extend(0.0),
            (across * self.scale).extend(0.0),
            (origin + Vec3::new(x, base, z)).extend(1.0),
        )
    }
}

engine_codec::record!(Placed {
    position,
    yaw,
    lean,
    scale,
    sink,
    ground,
    asset,
    layer,
    near_level,
    rank,
});

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacedAsset {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlacedLayer {
    pub density: f32,
    pub scale: [f32; 2],
    pub stands_on: Option<[u64; 4]>,
    pub assets: Vec<(u16, f32)>,
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
    pub control: Option<Arc<[u32]>>,
    pub river_edges: Option<Arc<[u16]>>,
    pub river_beds: Option<Arc<[u16]>>,
    pub patch_bounds: Arc<[[f32; 2]]>,
    pub patch_holes: Arc<[u64]>,
    pub ground_errors: Arc<[u16]>,
    pub water_errors: Arc<[u16]>,
    pub errors: Arc<[f32]>,
    pub stamps: Arc<[engine_heightfield::Stamp]>,
    pub placed: Arc<[Placed]>,
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
    pub pinned_levels: u32,
    pub pixel_error: f32,
    pub level_errors: Arc<[f32]>,
    pub surface_program: bool,
    pub baked: Option<Arc<engine_heightfield::Layers>>,
    pub macro_map: Option<TextureHandle>,
    pub materials: Arc<[SurfaceMaterial]>,
    pub maps: Arc<[Option<SurfaceMaps>]>,
    pub placement: Arc<[PlacedAsset]>,
    pub layers: Arc<[PlacedLayer]>,
    pub tiles: Arc<[TileFace]>,
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
    const WGSL: Option<&'static str> = Some(include_str!("places/ground_surface.wgsl"));
}

#[derive(Resource, Default)]
pub struct Heightfield;

impl Slot for Heightfield {
    const NAME: &'static str = "heightfield";
    const WGSL: Option<&'static str> = Some(include_str!("places/heightfield.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::HEIGHTFIELD];
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
}

impl Slot for GroundTiles {
    const NAME: &'static str = "ground_tiles";
    const WGSL: Option<&'static str> = Some(contract!(
        "places/ground_tiles.wgsl",
        TILE_QUADS,
        PALETTE_ENTRIES,
    ));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::GROUND_TILES];
}

#[derive(Resource, Default)]
pub struct Horizons;

impl Slot for Horizons {
    const NAME: &'static str = "horizons";
    const WGSL: Option<&'static str> = Some(include_str!("places/horizons.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::HORIZONS];
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
}

impl Slot for PlacedTiles {
    const NAME: &'static str = "placed_tiles";
    const WGSL: Option<&'static str> = Some(include_str!("places/placed_tiles.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::PLACED_TILES];
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct Cover {
    pub enabled: bool,
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
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::COVER];
}

pub struct FarCover;

impl List for FarCover {
    const NAME: &'static str = "far_cover";
    type Member = ();
    const WGSL: Option<&'static str> = Some(include_str!("places/far_cover.wgsl"));
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct Occupancy {
    pub texel_share: f32,
}

impl Slot for Occupancy {
    const NAME: &'static str = "occupancy";
    const WGSL: Option<&'static str> = Some(include_str!("places/occupancy.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::OCCUPANCY];
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct Foliage {
    pub ground_blend: f32,
    pub crown_depth: f32,
}

impl Slot for Foliage {
    const NAME: &'static str = "foliage";
    const WGSL: Option<&'static str> = Some(include_str!("places/foliage.wgsl"));
}
