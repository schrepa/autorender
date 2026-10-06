//! Terrain (decision 051): heightfields whose tiles stream around the camera,
//! drawn in continuous level of detail and sampled by queries and collision.
//!
//! An entity's [`Terrain`] describes one: where it lies, its tile pyramid,
//! where its tiles come from (generated from a layer stack, or read from an
//! `ETR1` file by byte ranges) and its surface (decision 054): the rules
//! choosing each sample's materials and how those look. The [`Terrains`] resource
//! keeps every terrain's resident tiles. Each extraction streams toward the
//! camera: the coarsest levels stay resident everywhere, so the whole
//! terrain always draws; finer levels load nearest and ahead of the camera
//! first, and level-0 tiles load around the camera and [`TerrainFocus`]
//! entities for collision. Generating and decoding share a per-frame work
//! budget; past a tile budget or the device's capacity the least important
//! tiles drop and count as dropped. Frames list each terrain's resident
//! tiles and the patches the camera draws in [`RenderSnapshot::terrain`].
//!
//! Queries ([`Terrains::sample`], [`Terrains::height_at`] and the rest)
//! answer from the finest resident tile at a point, on the triangles
//! renderers and collision use, and return `None` where no tile is
//! resident.
//!
//! The [`TerrainSettings`] are inserted by default; without them, or when
//! disabled, terrains neither stream nor draw.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy_ecs::{
    prelude::{Component, Entity, Resource, With, World},
    query::QueryState,
};
use glam::{Mat4, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use engine_assets::{AssetLibrary, Material, MaterialHandle, MeshHandle};
use engine_terrain::{
    Baked, Biome, BiomeError, BiomeLayer, BiomeRules, Biomes, ErosionStats, FileHeader,
    FormatError, Generator, HydrologyStats, Layers, LodRanges, Mask, NoiseMask, Patch, Pyramid,
    Remap, Route, RouteOptions, Selection, TILE_STRIDE, TerrainBake, Tile, TileKey, TileSample,
    WantedTile, decode_tile,
};

use crate::environment::{InvalidSetting, check_budget, check_range};
use crate::{Camera, GlobalTransforms, RenderMaterial, RenderMesh, RenderSnapshot, RenderTexture};

/// Where a terrain's tiles come from.
#[derive(Clone, Debug, PartialEq)]
pub enum TerrainSource {
    /// Generated in the engine from a layer stack, a tile at a time.
    Generated(Arc<Layers>),
    /// Read from an `ETR1` file: its header and directory, and the URL
    /// hosts fetch its tiles' byte ranges from, or the whole file, which
    /// tiles then decode from without fetching.
    File {
        header: Arc<FileHeader>,
        url: Arc<str>,
        data: Option<Arc<[u8]>>,
    },
}

/// How a terrain's ground looks (decision 054): the rules choosing each
/// sample's materials (its biomes), and how each material looks, in the
/// rules' order.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainSurface {
    pub rules: Arc<BiomeRules>,
    pub materials: Vec<SurfaceMaterial>,
    /// What the rules' placement layers stand on the terrain (decision
    /// 057), in the rules' order of asset IDs.
    pub assets: Vec<PlacementAsset>,
}

/// An object placement layers stand on a terrain (decision 057): the mesh
/// it draws and the material it draws with, the coarser meshes it draws at
/// a distance, finest first (without them, the mesh's automatic levels of
/// detail, decision 047), how far it sways in the wind (decision 058):
/// meters at ten meters up in full wind, its bend growing with the square
/// of the height, 0 (rigid) to 10, and its objects' collider (decision
/// 064), if they collide.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacementAsset {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub lods: Vec<PlacementLod>,
    pub sway: f32,
    pub collider: Option<PlacedCollider>,
}

/// The collider of a placement asset's objects (decision 064), in the
/// asset's mesh space at scale 1: its primitive about `center`, and the
/// collision layers the objects belong to and collide with. Objects within
/// terrains' collision radius collide; those standing far away (decision
/// 061) never do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacedCollider {
    pub shape: PlacedShape,
    pub center: [f32; 3],
    pub memberships: u32,
    pub filter: u32,
}

/// A placed collider's primitive: a box's half extents or a sphere's
/// radius, each 0.0001 to 10000.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlacedShape {
    Cuboid { half_extents: [f32; 3] },
    Sphere { radius: f32 },
}

impl PlacedCollider {
    /// A collider of `shape` about `center` in the default layers: a
    /// member of the first, colliding with every one.
    pub fn new(shape: PlacedShape, center: [f32; 3]) -> Self {
        Self {
            shape,
            center,
            memberships: 1,
            filter: u32::MAX,
        }
    }

    /// Whether its dimensions are in range and its center finite.
    pub fn is_usable(&self) -> bool {
        let usable = |value: f32| value.is_finite() && (1.0e-4..=1.0e4).contains(&value);
        let dimensions = match self.shape {
            PlacedShape::Cuboid { half_extents } => half_extents.iter().all(|&half| usable(half)),
            PlacedShape::Sphere { radius } => usable(radius),
        };
        dimensions && self.center.iter().all(|value| value.is_finite())
    }
}

/// A coarser mesh of a placement asset and the largest deviation from the
/// asset's mesh it shows, in mesh units: it draws where that projects
/// within the pixel error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacementLod {
    pub mesh: MeshHandle,
    pub error: f32,
}

/// Coarser meshes a placement asset lists at most: renderers list each
/// object's levels in a fixed number of planes.
pub const MAX_PLACEMENT_LODS: usize = 4;

impl PlacementLod {
    /// Checks a placement asset's coarser meshes' `errors`, in order: at
    /// most [`MAX_PLACEMENT_LODS`], each finite, positive and greater than
    /// the last. Errors name the offending index (none for the count) and
    /// the rule.
    pub fn check(errors: &[f32]) -> Result<(), (Option<usize>, &'static str)> {
        if errors.len() > MAX_PLACEMENT_LODS {
            return Err((None, "lists at most 4 meshes"));
        }
        let mut last = 0.0;
        for (level, &error) in errors.iter().enumerate() {
            if !(error.is_finite() && error > last) {
                return Err((
                    Some(level),
                    "must be finite, positive and grow with each mesh",
                ));
            }
            last = error;
        }
        Ok(())
    }
}

/// How a surface material looks: an engine material's maps and factors,
/// or without one a linear `color` and `roughness`; the meters its
/// textures repeat over; how sharply its height map wins where it blends
/// with another (0 even, 1 sharpest); whether steep ground projects it
/// from the side; and the cover growing where it lies (decision 056).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceMaterial {
    pub material: Option<MaterialHandle>,
    pub color: [f32; 3],
    pub roughness: f32,
    pub tiling: f32,
    pub height_contrast: f32,
    pub projection: SurfaceProjection,
    pub cover: Option<crate::Cover>,
}

/// How a surface material's textures lie on the ground: from above, or
/// on steep ground also from the side, so cliffs don't stretch them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SurfaceProjection {
    #[default]
    Top,
    Biplanar,
}

impl SurfaceMaterial {
    /// A material of plain linear `color` and `roughness`, its textures (if
    /// it gains any) repeating every 4 meters.
    pub fn color(color: [f32; 3], roughness: f32) -> Self {
        Self {
            material: None,
            color,
            roughness,
            tiling: 4.0,
            height_contrast: 0.5,
            projection: SurfaceProjection::Top,
            cover: None,
        }
    }

    /// Whether every value is in range: color 0 to 1, roughness 0.03 to
    /// 1, tiling 0.01 to 1000 meters, height contrast 0 to 1, and its
    /// cover's.
    pub fn is_usable(&self) -> bool {
        self.cover.is_none_or(|cover| cover.validate().is_ok()) && self.values_usable()
    }

    fn values_usable(&self) -> bool {
        self.color
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
            && self.roughness.is_finite()
            && (0.03..=1.0).contains(&self.roughness)
            && self.tiling.is_finite()
            && (0.01..=1000.0).contains(&self.tiling)
            && self.height_contrast.is_finite()
            && (0.0..=1.0).contains(&self.height_contrast)
    }
}

impl TerrainSurface {
    /// The surface of `surface`'s rules with `materials`, one per
    /// material it names, in order.
    pub fn new(surface: &Biomes, materials: Vec<SurfaceMaterial>) -> Result<Self, BiomeError> {
        let rules = surface.compile()?;
        if materials.len() != surface.materials.len() {
            return Err(BiomeError {
                path: "materials".into(),
                rule: "needs one look per material".into(),
            });
        }
        Ok(Self {
            rules: Arc::new(rules),
            materials,
            assets: Vec::new(),
        })
    }

    /// The surface with `assets`, one per asset its rules name, in order
    /// (decision 057).
    pub fn with_assets(mut self, assets: Vec<PlacementAsset>) -> Result<Self, BiomeError> {
        let fail = |path: String, rule: &str| {
            Err(BiomeError {
                path,
                rule: rule.into(),
            })
        };
        if assets.len() != self.rules.authored().assets.len() {
            return fail("assets".into(), "needs one mesh and material per asset");
        }
        for (index, asset) in assets.iter().enumerate() {
            if !(asset.sway.is_finite() && (0.0..=10.0).contains(&asset.sway)) {
                return fail(format!("assets[{index}].sway"), "must be in [0, 10]");
            }
            if asset.collider.is_some_and(|collider| !collider.is_usable()) {
                return fail(
                    format!("assets[{index}].collider"),
                    "needs dimensions in [0.0001, 10000] and a finite center",
                );
            }
            let errors: Vec<f32> = asset.lods.iter().map(|lod| lod.error).collect();
            if let Err((level, rule)) = PlacementLod::check(&errors) {
                let path = match level {
                    Some(level) => format!("assets[{index}].lods[{level}].error"),
                    None => format!("assets[{index}].lods"),
                };
                return fail(path, rule);
            }
        }
        self.assets = assets;
        Ok(self)
    }

    /// Whether every material is usable and one exists per material and
    /// asset the rules name.
    pub fn is_usable(&self) -> bool {
        self.materials.len() == self.rules.authored().materials.len()
            && self.assets.len() == self.rules.authored().assets.len()
            && self.materials.iter().all(SurfaceMaterial::is_usable)
    }

    /// The default surface's rules as authored: grass, bare dirt where noise
    /// and slope open it up, rock on steep ground and snow above
    /// `snow_height` meters on gentler slopes.
    pub fn default_rules(snow_height: f32) -> Biomes {
        let noise = |wavelength: f32, seed: u32| {
            Mask::Noise(NoiseMask {
                wavelength,
                seed,
                octaves: 2,
            })
        };
        let scaled = |mask: Mask, by: f32| Mask::Multiply(vec![mask, Mask::Constant(by)]);
        let remap = |mask: Mask, from: f32, to: f32| {
            Mask::Remap(Remap {
                mask: Box::new(mask),
                from,
                to,
            })
        };
        let layer = |material: &str, mask: Mask| BiomeLayer {
            material: material.into(),
            mask,
            opacity: 1.0,
        };
        Biomes {
            assets: Vec::new(),
            placements: Vec::new(),
            materials: ["grass", "dirt", "rock", "snow"].map(String::from).to_vec(),
            biomes: vec![Biome {
                id: "ground".into(),
                fitness: Mask::one(),
                layers: vec![
                    layer("grass", Mask::one()),
                    layer(
                        "dirt",
                        remap(
                            Mask::Add(vec![
                                scaled(noise(60.0, 1), 0.7),
                                scaled(noise(9.0, 2), 0.3),
                                scaled(Mask::Slope([10.0, 40.0]), 0.5),
                            ]),
                            0.55,
                            0.75,
                        ),
                    ),
                    layer(
                        "rock",
                        remap(
                            Mask::Add(vec![Mask::Slope([25.0, 45.0]), scaled(noise(60.0, 1), 0.3)]),
                            0.55,
                            0.75,
                        ),
                    ),
                    layer(
                        "snow",
                        Mask::Multiply(vec![
                            remap(
                                Mask::Add(vec![
                                    Mask::Height([snow_height - 100.0, snow_height + 100.0]),
                                    scaled(noise(60.0, 3), 0.4),
                                    Mask::Constant(-0.2),
                                ]),
                                0.4,
                                0.6,
                            ),
                            Mask::Invert(Box::new(Mask::Slope([35.0, 50.0]))),
                        ]),
                    ),
                ],
            }],
        }
    }

    /// The default materials' looks, matching [`Self::default_rules`].
    pub fn default_materials() -> Vec<SurfaceMaterial> {
        vec![
            SurfaceMaterial::color([0.105, 0.135, 0.045], 0.92),
            SurfaceMaterial::color([0.21, 0.165, 0.11], 0.92),
            SurfaceMaterial::color([0.25, 0.235, 0.22], 0.82),
            SurfaceMaterial::color([0.82, 0.84, 0.88], 0.55),
        ]
    }
}

impl Default for TerrainSurface {
    fn default() -> Self {
        Self::new(&Self::default_rules(650.0), Self::default_materials())
            .expect("the default surface is valid")
    }
}

/// A heightfield terrain (decision 051) whose first corner (least x and z)
/// lies at `origin`, heights measured up from it. Its pyramid sets its
/// level-0 tiles and spacing; its source gives its tiles.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct Terrain {
    pub origin: Vec3,
    pub pyramid: Pyramid,
    pub source: TerrainSource,
    pub surface: TerrainSurface,
    /// Whether it collides where collision is installed.
    pub collision: bool,
}

impl Terrain {
    /// Whether `other` describes the same terrain: equal fields and the
    /// same shared source, so streaming keeps its tiles.
    fn same(&self, other: &Self) -> bool {
        let source = match (&self.source, &other.source) {
            (TerrainSource::Generated(a), TerrainSource::Generated(b)) => Arc::ptr_eq(a, b),
            (
                TerrainSource::File {
                    header: a, url: u, ..
                },
                TerrainSource::File {
                    header: b, url: v, ..
                },
            ) => Arc::ptr_eq(a, b) && u == v,
            _ => false,
        };
        source
            && self.origin == other.origin
            && self.pyramid == other.pyramid
            && (Arc::ptr_eq(&self.surface.rules, &other.surface.rules)
                || self.surface.rules == other.surface.rules)
            && self.surface.materials == other.surface.materials
            && self.collision == other.collision
    }

    /// A terrain generated from `layers`.
    pub fn generated(origin: Vec3, pyramid: Pyramid, layers: Layers) -> Self {
        Self {
            origin,
            pyramid,
            source: TerrainSource::Generated(Arc::new(layers)),
            surface: TerrainSurface::default(),
            collision: true,
        }
    }

    /// A terrain read from the `ETR1` file at `url` whose prefix parsed into
    /// `header`.
    pub fn from_file(origin: Vec3, header: FileHeader, url: impl Into<Arc<str>>) -> Self {
        Self {
            origin,
            pyramid: *header.pyramid(),
            source: TerrainSource::File {
                header: Arc::new(header),
                url: url.into(),
                data: None,
            },
            surface: TerrainSurface::default(),
            collision: true,
        }
    }

    /// A terrain read from a whole `ETR1` file in memory.
    pub fn from_file_bytes(origin: Vec3, bytes: Arc<[u8]>) -> Result<Self, FormatError> {
        let header = FileHeader::parse(&bytes)?;
        Ok(Self {
            origin,
            pyramid: *header.pyramid(),
            source: TerrainSource::File {
                header: Arc::new(header),
                url: Arc::from(""),
                data: Some(bytes),
            },
            surface: TerrainSurface::default(),
            collision: true,
        })
    }

    pub fn with_surface(mut self, surface: TerrainSurface) -> Self {
        self.surface = surface;
        self
    }

    pub fn with_collision(mut self, collision: bool) -> Self {
        self.collision = collision;
        self
    }

    /// Whether streaming can use this terrain.
    pub fn is_usable(&self) -> bool {
        self.origin.is_finite() && self.surface.is_usable()
    }
}

/// What terrain shows (decision 051): its material, or for development its
/// levels of detail, normals or tiles and patches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum TerrainView {
    #[default]
    Shaded,
    Levels,
    Normals,
    Tiles,
    /// Each surface material in its own color, blended as the ground is.
    Materials,
    /// The share of the sky the ground sees (decision 055), unlit, from
    /// black (none) to white (all of it).
    Sky,
}

/// Temporary terrain state for development (decision 051): an eye that
/// streaming and level of detail follow instead of the camera's, so agents
/// can look at the patches it selects from elsewhere, and what terrain
/// shows. Absent, terrain follows the camera, shaded.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct TerrainDebug {
    pub freeze: Option<Vec3>,
    pub view: TerrainView,
}

/// Keeps level-0 terrain tiles resident around its entity, as the camera
/// does, so collision and queries hold there.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TerrainFocus;

/// The size in pixels of the viewport frames render to, which terrain level
/// of detail measures pixels on; hosts set it each frame. Without it,
/// 1920x1080.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
        }
    }
}

/// Terrain settings (decision 051). Requires version 9.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneTerrainSettings")
)]
#[serde(default, deny_unknown_fields)]
pub struct TerrainSettings {
    pub enabled: bool,
    /// Pixels a level's samples span before the next coarser level takes
    /// over, 1 to 64; smaller draws more triangles.
    #[cfg_attr(feature = "schema", schemars(range(min = 1.0, max = 64.0)))]
    pub pixel_error: f32,
    /// The coarsest levels kept resident over the whole terrain, so it
    /// always draws whole, from 1 to 8.
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 8)))]
    pub pinned_levels: u32,
    /// An optional budget of resident tiles (about 270 KB each, twice that
    /// on the GPU); the least important beyond it drop.
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tiles: Option<u32>,
    /// Work generated or decoded per frame, in samples' worth (67,000 a
    /// decoded tile, about 500,000 a generated one).
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    pub samples_per_frame: u32,
    /// Meters around the camera and focus entities within which level-0
    /// tiles load for collision, 0 to 4096.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 4096.0)))]
    pub collision_radius: f32,
    /// Long sun shadows through each terrain's horizon map (decision 052),
    /// with shadows on; shadow maps then draw terrain near the camera only.
    /// Off, terrain casts into the shadow maps alone.
    pub sun_shadows: bool,
    /// Texels along a horizon map's sides, 64 to 4096 (8 bytes each).
    #[cfg_attr(feature = "schema", schemars(range(min = 64, max = 4096)))]
    pub shadow_resolution: u32,
    /// Shadow edge height per meter to the slope casting it (the sun's disc
    /// gives 0.0093).
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 0.2)))]
    pub shadow_softness: f32,
    /// Ambient light on terrain scaled by how much sky its ground sees
    /// (decision 055): each terrain's sky visibility map, built once its
    /// finest pinned level is resident, where valleys and hollows see less
    /// sky and take their own light back instead.
    pub sky_visibility: bool,
    /// Texels along a sky visibility map's sides, 64 to 4096 (4 bytes each).
    #[cfg_attr(feature = "schema", schemars(range(min = 64, max = 4096)))]
    pub sky_resolution: u32,
}

impl Default for TerrainSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            pixel_error: 6.0,
            pinned_levels: 2,
            max_tiles: None,
            samples_per_frame: 24_576,
            collision_radius: 192.0,
            sun_shadows: true,
            shadow_resolution: 1024,
            shadow_softness: 0.01,
            sky_visibility: true,
            sky_resolution: 1024,
        }
    }
}

impl TerrainSettings {
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("pixel_error", self.pixel_error, 1.0..=64.0)?;
        if !(1..=8).contains(&self.pinned_levels) {
            return Err(InvalidSetting::new("pinned_levels", "must be in [1, 8]"));
        }
        check_budget("max_tiles", self.max_tiles)?;
        if self.samples_per_frame == 0 {
            return Err(InvalidSetting::new(
                "samples_per_frame",
                "must be at least 1",
            ));
        }
        check_range("collision_radius", self.collision_radius, 0.0..=4096.0)?;
        if !(64..=4096).contains(&self.shadow_resolution) {
            return Err(InvalidSetting::new(
                "shadow_resolution",
                "must be in [64, 4096]",
            ));
        }
        check_range("shadow_softness", self.shadow_softness, 0.0..=0.2)?;
        if !(64..=4096).contains(&self.sky_resolution) {
            return Err(InvalidSetting::new(
                "sky_resolution",
                "must be in [64, 4096]",
            ));
        }
        Ok(())
    }

    /// Every value in range.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        let finite = |value: f32, fallback: f32| if value.is_finite() { value } else { fallback };
        Self {
            enabled: self.enabled,
            pixel_error: crate::math::clamp(
                finite(self.pixel_error, defaults.pixel_error),
                1.0,
                64.0,
            ),
            pinned_levels: self.pinned_levels.clamp(1, 8),
            max_tiles: self.max_tiles.map(|budget| budget.max(1)),
            samples_per_frame: self.samples_per_frame.max(1),
            collision_radius: crate::math::clamp(
                finite(self.collision_radius, defaults.collision_radius),
                0.0,
                4096.0,
            ),
            sun_shadows: self.sun_shadows,
            shadow_resolution: self.shadow_resolution.clamp(64, 4096),
            shadow_softness: crate::math::clamp(
                finite(self.shadow_softness, defaults.shadow_softness),
                0.0,
                0.2,
            ),
            sky_visibility: self.sky_visibility,
            sky_resolution: self.sky_resolution.clamp(64, 4096),
        }
    }
}

/// The surface of a terrain at a point: its terrain, world height, the
/// normal of its triangle, whether its cell is a hole, and the level of the
/// tile it came from (0 is the finest).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainSample {
    pub terrain: Entity,
    pub height: f32,
    pub normal: Vec3,
    pub hole: bool,
    pub level: u32,
}

/// A byte range of a terrain file a host should fetch and hand back with
/// [`Terrains::deliver`].
#[derive(Clone, Debug, PartialEq)]
pub struct TileRequest {
    pub terrain: Entity,
    pub key: TileKey,
    pub url: Arc<str>,
    pub offset: u64,
    pub length: u32,
}

/// Counters of the last streaming update, over every terrain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct TerrainStats {
    pub terrains: u32,
    /// Tiles resident now.
    pub resident_tiles: u32,
    /// Tiles being generated, fetched or decoded.
    pub loading_tiles: u32,
    /// Wanted tiles beyond the tile budget or the device's capacity.
    pub dropped_tiles: u32,
    /// Patches the camera draws, and of those how many stand in for finer
    /// levels not resident yet.
    pub patches: u32,
    pub fallback_patches: u32,
    /// Terrains whose bakes (decisions 066 and 068) are in progress: they
    /// load no tiles until done.
    pub baking: u32,
}

/// Frames a failed fetch waits before it is requested again.
const RETRY_AFTER: u64 = 120;
/// Ranges, as multiples of each level's, within which tiles load and stay:
/// beyond its range a level's tiles still shade the coarser level's
/// patches (their normals), so they load further than they draw.
const LOAD_REACH: f32 = 2.0;
const KEEP_REACH: f32 = 2.5;

/// One terrain's streaming state.
struct TerrainState {
    entity: Entity,
    terrain: Terrain,
    tiles: HashMap<TileKey, Arc<Tile>>,
    /// Generators of tiles in progress.
    jobs: HashMap<TileKey, Generator>,
    /// File tiles requested and not yet delivered.
    in_flight: HashSet<TileKey>,
    arrivals: Vec<(TileKey, Option<Vec<u8>>)>,
    failed: HashMap<TileKey, u64>,
    selection: Selection,
    /// Level-0 tiles collision wants.
    collision: Vec<TileKey>,
    /// Advances whenever the resident tiles change.
    revision: u64,
    /// The largest height difference measured between each level and the
    /// next coarser one ([`Tile::morph_error`]), what its patches morph by;
    /// zero until measured.
    errors: Vec<f32>,
    /// The ranges the last selection used.
    ranges: Vec<f32>,
    /// The layers tiles generate from once its bakes are done (decisions
    /// 066 and 068), the bake in progress, and what the last hydrology and
    /// erosion bakes found.
    baked: Option<Arc<Layers>>,
    bake: Option<TerrainBake>,
    hydrology: Option<HydrologyStats>,
    erosion: Option<ErosionStats>,
}

/// A terrain's bakes (decisions 066 and 068): whether one is in progress,
/// and what its hydrology and erosion bakes found once done.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TerrainBakes {
    pub baking: bool,
    pub hydrology: Option<HydrologyStats>,
    pub erosion: Option<ErosionStats>,
}

impl TerrainState {
    fn new(entity: Entity, terrain: Terrain) -> Self {
        let levels = terrain.pyramid.levels() as usize;
        Self {
            entity,
            terrain,
            tiles: HashMap::new(),
            jobs: HashMap::new(),
            in_flight: HashSet::new(),
            arrivals: Vec::new(),
            failed: HashMap::new(),
            selection: Selection::default(),
            collision: Vec::new(),
            revision: 1,
            errors: vec![0.0; levels],
            ranges: Vec::new(),
            baked: None,
            bake: None,
            hydrology: None,
            erosion: None,
        }
    }

    /// Gives decoded `tile` its surface's control values, within
    /// `budget`; it becomes resident once they are done.
    fn resurface(&mut self, tile: Tile, budget: &mut u64) {
        let key = tile.key();
        let mut job = Generator::resurface(
            &tile,
            &self.terrain.pyramid,
            self.terrain.surface.rules.clone(),
        );
        match job.advance(budget) {
            Some(tile) => self.insert(tile),
            None => {
                self.jobs.insert(key, job);
            }
        }
    }

    /// Advances `key`'s surface job within `budget`, if it has one,
    /// making its tile resident once done; whether it had one.
    fn surface_job(&mut self, key: TileKey, budget: &mut u64) -> bool {
        let Some(job) = self.jobs.get_mut(&key) else {
            return false;
        };
        if let Some(tile) = job.advance(budget) {
            self.jobs.remove(&key);
            self.insert(tile);
        }
        true
    }

    /// Makes `tile` resident, measuring how far it lies from its resident
    /// parent and its resident children from it.
    fn insert(&mut self, tile: Tile) {
        let key = tile.key();
        let pyramid = self.terrain.pyramid;
        let level = usize::from(key.level);
        if let Some(parent) = pyramid
            .parent(key)
            .and_then(|parent| self.tiles.get(&parent))
        {
            self.errors[level] = crate::math::max(self.errors[level], tile.morph_error(parent));
        }
        if let Some(finer) = level.checked_sub(1) {
            for (dx, dz) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let child = TileKey::new(finer as u8, key.x * 2 + dx, key.z * 2 + dz);
                if let Some(child) = self.tiles.get(&child) {
                    self.errors[finer] =
                        crate::math::max(self.errors[finer], child.morph_error(&tile));
                }
            }
        }
        self.tiles.insert(key, Arc::new(tile));
        self.revision += 1;
    }

    /// The finest resident tile holding terrain-local `local`, and its
    /// sample there.
    fn sample(&self, local: Vec2) -> Option<(u32, TileSample)> {
        let pyramid = &self.terrain.pyramid;
        (0..pyramid.levels()).find_map(|level| {
            let key = pyramid.tile_at(level, local)?;
            let tile = self.tiles.get(&key)?;
            let sample = tile.sample(local - pyramid.tile_origin(key))?;
            Some((level, sample))
        })
    }
}

/// Every terrain's resident tiles, streaming state and queries (decision
/// 051). Inserted by default; terrains stream during extraction.
#[derive(Resource, Default)]
pub struct Terrains {
    entries: Vec<TerrainState>,
    device_tiles: Option<u32>,
    frame: u64,
    stats: TerrainStats,
    requests: Vec<TileRequest>,
    // Scratch.
    wanted: Vec<WantedTile>,
}

impl Terrains {
    /// The surface under world point `(x, z)` of the first terrain with a
    /// resident tile there, from the finest such tile.
    pub fn sample(&self, x: f32, z: f32) -> Option<TerrainSample> {
        self.entries.iter().find_map(|state| {
            let origin = state.terrain.origin;
            let (level, sample) = state.sample(Vec2::new(x - origin.x, z - origin.z))?;
            Some(TerrainSample {
                terrain: state.entity,
                height: origin.y + sample.height,
                normal: sample.normal,
                hole: sample.hole,
                level,
            })
        })
    }

    /// The world height of the terrain under `(x, z)`; `None` where no tile
    /// is resident or the cell is a hole.
    pub fn height_at(&self, x: f32, z: f32) -> Option<f32> {
        self.sample(x, z)
            .filter(|sample| !sample.hole)
            .map(|sample| sample.height)
    }

    /// The surface normal under `(x, z)`, as [`Self::height_at`].
    pub fn normal_at(&self, x: f32, z: f32) -> Option<Vec3> {
        self.sample(x, z)
            .filter(|sample| !sample.hole)
            .map(|sample| sample.normal)
    }

    /// Whether the cell under `(x, z)` is a hole; `None` where no tile is
    /// resident.
    pub fn is_hole(&self, x: f32, z: f32) -> Option<bool> {
        self.sample(x, z).map(|sample| sample.hole)
    }

    /// Heights of a grid of `count` points from `min` to `max` (world x and
    /// z), row by row along z, into `out`: `None` where no tile is resident
    /// or the cell is a hole.
    pub fn region(&self, min: Vec2, max: Vec2, count: [u32; 2], out: &mut Vec<Option<f32>>) {
        out.clear();
        let steps = Vec2::new((count[0].max(2) - 1) as f32, (count[1].max(2) - 1) as f32);
        for row in 0..count[1] {
            for column in 0..count[0] {
                let point = min + (max - min) * Vec2::new(column as f32, row as f32) / steps;
                out.push(self.height_at(point.x, point.y));
            }
        }
    }

    /// Counters of the last streaming update.
    pub fn stats(&self) -> TerrainStats {
        self.stats
    }

    /// The terrain of `entity`, while it streams.
    pub fn terrain(&self, entity: Entity) -> Option<&Terrain> {
        self.entries
            .iter()
            .find(|state| state.entity == entity)
            .map(|state| &state.terrain)
    }

    /// The layers `entity`'s tiles generate from: its own, with their
    /// hydrology baked once it is (decision 066); `None` for terrains read
    /// from files.
    pub fn layers(&self, entity: Entity) -> Option<Arc<Layers>> {
        let state = self.entries.iter().find(|state| state.entity == entity)?;
        match &state.terrain.source {
            TerrainSource::Generated(layers) => {
                Some(state.baked.clone().unwrap_or_else(|| layers.clone()))
            }
            TerrainSource::File { .. } => None,
        }
    }

    /// The cheapest route over `entity`'s generated terrain from world
    /// `from` to `to` (x and z), as [`engine_terrain::route`] finds it over
    /// its layers with their bakes (decision 069): its points are
    /// terrain-local, as path layers name them. `None` for terrains read
    /// from files, before their bakes are done, and where no route keeps
    /// within the steepest grade. Its work grows with the square of the
    /// ends' distance: for authoring and loading, not every frame.
    pub fn route(
        &self,
        entity: Entity,
        from: Vec2,
        to: Vec2,
        options: &RouteOptions,
    ) -> Option<Route> {
        let state = self.entries.iter().find(|state| state.entity == entity)?;
        let TerrainSource::Generated(authored) = &state.terrain.source else {
            return None;
        };
        let layers = state.baked.as_ref().unwrap_or(authored);
        if layers.unbaked().is_some() {
            return None;
        }
        let origin = Vec2::new(state.terrain.origin.x, state.terrain.origin.z);
        engine_terrain::route(
            layers,
            &state.terrain.pyramid,
            from - origin,
            to - origin,
            options,
        )
    }

    /// `entity`'s bakes (decisions 066 and 068): whether one is in
    /// progress, and what its hydrology and erosion bakes found.
    pub fn bakes(&self, entity: Entity) -> TerrainBakes {
        self.entries
            .iter()
            .find(|state| state.entity == entity)
            .map_or_else(TerrainBakes::default, |state| TerrainBakes {
                baking: state.bake.is_some(),
                hydrology: state.hydrology,
                erosion: state.erosion,
            })
    }

    /// Each level's largest measured height difference from the next
    /// coarser one (meters; zero until measured) and the range it drew to
    /// at the last selection (meters from the eye), finest first, of
    /// `entity`'s terrain.
    pub fn level_detail(&self, entity: Entity) -> Option<(&[f32], &[f32])> {
        self.entries
            .iter()
            .find(|state| state.entity == entity)
            .map(|state| (state.errors.as_slice(), state.ranges.as_slice()))
    }

    /// Resident tiles of `entity`'s terrain.
    pub fn tiles(&self, entity: Entity) -> impl Iterator<Item = &Arc<Tile>> {
        self.entries
            .iter()
            .filter(move |state| state.entity == entity)
            .flat_map(|state| state.tiles.values())
    }

    /// Level-0 tiles resident around the camera and focus entities, of
    /// terrains that collide: the terrain, its origin and the tile.
    pub fn collision_tiles(&self) -> impl Iterator<Item = (Entity, Vec3, &Arc<Tile>)> {
        self.entries
            .iter()
            .filter(|state| state.terrain.collision)
            .flat_map(|state| {
                state.collision.iter().filter_map(move |key| {
                    let tile = state.tiles.get(key)?;
                    Some((state.entity, state.terrain.origin, tile))
                })
            })
    }

    /// Advances whenever any terrain's resident tiles change.
    pub fn revision(&self) -> u64 {
        self.entries
            .iter()
            .map(|state| state.revision)
            .fold(self.entries.len() as u64, u64::wrapping_add)
    }

    /// Tiles the device holds, which renderers report: residency never
    /// exceeds it. `None` for no limit.
    pub fn set_device_tiles(&mut self, tiles: Option<u32>) {
        self.device_tiles = tiles;
    }

    /// Moves the byte ranges hosts should fetch into `out`.
    pub fn take_requests(&mut self, out: &mut Vec<TileRequest>) {
        out.append(&mut self.requests);
    }

    /// Hands back a fetched tile's bytes, or `None` when the fetch failed:
    /// it is requested again later. Tiles no longer wanted are ignored.
    pub fn deliver(&mut self, terrain: Entity, key: TileKey, bytes: Option<Vec<u8>>) {
        if let Some(state) = self
            .entries
            .iter_mut()
            .find(|state| state.entity == terrain)
            && state.in_flight.remove(&key)
        {
            state.arrivals.push((key, bytes));
        }
    }

    /// Streams every terrain toward `eye` (the camera's, unless frozen)
    /// and focus points, within the settings' work and tile budgets, and
    /// selects the patches seen from `eye`.
    fn stream(
        &mut self,
        settings: &TerrainSettings,
        camera: &Camera,
        eye: Vec3,
        viewport: Viewport,
        focus: &[Vec3],
    ) {
        self.frame += 1;
        let mut stats = TerrainStats {
            terrains: self.entries.len() as u32,
            ..TerrainStats::default()
        };
        let forward = camera.target - eye;
        let forward = Vec2::new(forward.x, forward.z);
        let fov = camera.vertical_fov_radians;
        let fov = if fov.is_finite() && fov > 0.0 && fov < std::f32::consts::PI {
            fov
        } else {
            Camera::default().vertical_fov_radians
        };
        let mut budget = u64::from(settings.samples_per_frame);
        let limit = match (settings.max_tiles, self.device_tiles) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        // Budgets cover every terrain: each takes what is left.
        let mut remaining = limit.map_or(usize::MAX, |limit| limit as usize);
        let frame = self.frame;
        for state in &mut self.entries {
            if !state.terrain.is_usable() {
                continue;
            }
            let pyramid = state.terrain.pyramid;
            let origin = state.terrain.origin;
            let ranges =
                LodRanges::new(&pyramid, settings.pixel_error, viewport.height as f32, fov);
            state.ranges.clear();
            state
                .ranges
                .extend((0..pyramid.levels()).map(|level| ranges.range(level)));
            let local_eye = eye - origin;
            // What to load: pinned levels and the camera's tiles, then
            // collision's, most important first.
            self.wanted.clear();
            engine_terrain::wanted_tiles(
                &pyramid,
                &ranges,
                local_eye,
                forward,
                LOAD_REACH,
                settings.pinned_levels,
                &mut self.wanted,
            );
            state.collision.clear();
            if state.terrain.collision && settings.collision_radius > 0.0 {
                for point in focus {
                    collision_keys(
                        &pyramid,
                        *point - origin,
                        settings.collision_radius,
                        &mut state.collision,
                    );
                }
                state.collision.sort_unstable();
                state.collision.dedup();
            }
            let pinned_end = self
                .wanted
                .iter()
                .position(|tile| {
                    u32::from(tile.key.level) + settings.pinned_levels < pyramid.levels()
                })
                .unwrap_or(self.wanted.len());
            let mut order: Vec<TileKey> =
                Vec::with_capacity(self.wanted.len() + state.collision.len());
            let mut seen: HashSet<TileKey> = HashSet::with_capacity(order.capacity());
            for key in self.wanted[..pinned_end]
                .iter()
                .map(|tile| tile.key)
                .chain(state.collision.iter().copied())
                .chain(self.wanted[pinned_end..].iter().map(|tile| tile.key))
            {
                if seen.insert(key) {
                    order.push(key);
                }
            }
            let kept = order.len().min(remaining);
            stats.dropped_tiles += (order.len() - kept) as u32;
            remaining -= kept;
            let target: HashSet<TileKey> = order[..kept].iter().copied().collect();
            // What to keep: everything within a wider reach, unless over
            // budget.
            self.wanted.clear();
            engine_terrain::wanted_tiles(
                &pyramid,
                &ranges,
                local_eye,
                forward,
                KEEP_REACH,
                settings.pinned_levels,
                &mut self.wanted,
            );
            let keep: HashSet<TileKey> = self
                .wanted
                .iter()
                .map(|tile| tile.key)
                .chain(state.collision.iter().copied())
                .collect();
            let before = state.tiles.len();
            state.tiles.retain(|key, _| keep.contains(key));
            if state.tiles.len() > kept {
                // Over budget: drop resident tiles outside the target, finest
                // first.
                let mut extra: Vec<TileKey> = state
                    .tiles
                    .keys()
                    .filter(|key| !target.contains(key))
                    .copied()
                    .collect();
                extra.sort_unstable_by_key(|key| key.level);
                for key in extra.into_iter().take(state.tiles.len() - kept) {
                    state.tiles.remove(&key);
                }
            }
            if state.tiles.len() != before {
                state.revision += 1;
            }
            state.jobs.retain(|key, _| target.contains(key));
            // Load what is missing, most important first; disabled terrains
            // load nothing.
            let missing: Vec<TileKey> = order[..kept]
                .iter()
                .copied()
                .filter(|key| settings.enabled && !state.tiles.contains_key(key))
                .collect();
            match state.terrain.source.clone() {
                TerrainSource::Generated(authored) => {
                    // Bakes (decisions 066 and 068) run before any tile loads,
                    // each reading the layers below it, in stack order.
                    let layers = state.baked.clone().unwrap_or(authored);
                    if let Some(unbaked) = layers.unbaked() {
                        stats.baking += 1;
                        let bake = state.bake.get_or_insert_with(|| {
                            TerrainBake::new(&layers, unbaked.clone(), &pyramid)
                        });
                        let _profile = engine_profile::span("cpu", "terrain_bake");
                        if let Some(baked) = bake.advance(&pyramid, &mut budget) {
                            match &baked {
                                Baked::Hydrology(found) => state.hydrology = Some(found.stats),
                                Baked::Erosion(found) => state.erosion = Some(found.stats),
                                Baked::Path(_) => {}
                            }
                            let index = unbaked.0;
                            state.baked = Some(Arc::new(layers.with_baked(index, baked)));
                            state.bake = None;
                        }
                        continue;
                    }
                    // The coarsest level always draws: generate it whole at once.
                    let first = state.tiles.is_empty();
                    for key in missing {
                        let mut unlimited = u64::MAX;
                        let work = if first && u32::from(key.level) + 1 == pyramid.levels() {
                            &mut unlimited
                        } else {
                            &mut budget
                        };
                        if *work == 0 {
                            break;
                        }
                        let rules = &state.terrain.surface.rules;
                        let job = state.jobs.entry(key).or_insert_with(|| {
                            Generator::new(layers.clone(), &pyramid, key)
                                .with_surface(rules.clone())
                        });
                        if let Some(tile) = job.advance(work) {
                            state.jobs.remove(&key);
                            state.insert(tile);
                        }
                    }
                }
                TerrainSource::File {
                    header,
                    url: _,
                    data: Some(data),
                } => {
                    // The whole file is here: decode within the budget, then
                    // give each tile its surface.
                    for key in missing {
                        if budget == 0 {
                            break;
                        }
                        if state.surface_job(key, &mut budget) {
                            continue;
                        }
                        let Some(entry) = header.tile(key) else {
                            continue;
                        };
                        let range = usize::try_from(entry.offset).ok().and_then(|start| {
                            data.get(start..start.checked_add(entry.length as usize)?)
                        });
                        budget = budget.saturating_sub(u64::from(TILE_STRIDE * TILE_STRIDE));
                        match range.map(|bytes| decode_tile(&header, key, bytes)) {
                            Some(Ok(tile)) => state.resurface(tile, &mut budget),
                            _ => {
                                state.failed.insert(key, frame);
                            }
                        }
                    }
                }
                TerrainSource::File {
                    header,
                    url,
                    data: None,
                } => {
                    for &key in &missing {
                        if budget > 0 {
                            state.surface_job(key, &mut budget);
                        }
                        if state.in_flight.contains(&key)
                            || state.jobs.contains_key(&key)
                            || state.tiles.contains_key(&key)
                            || state
                                .failed
                                .get(&key)
                                .is_some_and(|&at| frame < at + RETRY_AFTER)
                        {
                            continue;
                        }
                        let Some(entry) = header.tile(key) else {
                            continue;
                        };
                        state.in_flight.insert(key);
                        self.requests.push(TileRequest {
                            terrain: state.entity,
                            key,
                            url: url.clone(),
                            offset: entry.offset,
                            length: entry.length,
                        });
                    }
                    let arrivals = std::mem::take(&mut state.arrivals);
                    let mut later = Vec::new();
                    for (key, bytes) in arrivals {
                        if budget == 0 || !target.contains(&key) {
                            if target.contains(&key) {
                                later.push((key, bytes));
                            }
                            continue;
                        }
                        budget = budget.saturating_sub(u64::from(TILE_STRIDE * TILE_STRIDE));
                        match bytes.map(|bytes| decode_tile(&header, key, &bytes)) {
                            Some(Ok(tile)) => {
                                state.failed.remove(&key);
                                state.resurface(tile, &mut budget);
                            }
                            _ => {
                                state.failed.insert(key, frame);
                            }
                        }
                    }
                    state.arrivals = later;
                }
            }
            stats.resident_tiles += state.tiles.len() as u32;
            stats.loading_tiles +=
                (state.jobs.len() + state.in_flight.len() + state.arrivals.len()) as u32;
            if settings.enabled {
                let tiles = &state.tiles;
                engine_terrain::select_patches(
                    &pyramid,
                    &ranges,
                    local_eye,
                    |key| tiles.get(&key).map(Arc::as_ref),
                    &mut state.selection,
                );
            } else {
                state.selection.patches.clear();
                state.selection.fallbacks = 0;
            }
            stats.patches += state.selection.patches.len() as u32;
            stats.fallback_patches += state.selection.fallbacks;
        }
        self.stats = stats;
    }
}

/// Appends the level-0 tiles within `radius` of terrain-local `point` to
/// `out`.
fn collision_keys(pyramid: &Pyramid, point: Vec3, radius: f32, out: &mut Vec<TileKey>) {
    let size = pyramid.tile_size(0);
    let [tiles_x, tiles_z] = pyramid.tiles(0);
    let low = ((Vec2::new(point.x, point.z) - radius) / size).floor();
    let high = ((Vec2::new(point.x, point.z) + radius) / size).floor();
    if high.x < 0.0 || high.y < 0.0 || low.x >= tiles_x as f32 || low.y >= tiles_z as f32 {
        return;
    }
    let first = [
        crate::math::max(low.x, 0.0) as u32,
        crate::math::max(low.y, 0.0) as u32,
    ];
    let last = [
        (high.x as u32).min(tiles_x - 1),
        (high.y as u32).min(tiles_z - 1),
    ];
    for z in first[1]..=last[1] {
        for x in first[0]..=last[0] {
            out.push(TileKey::new(0, x as u16, z as u16));
        }
    }
}

/// A surface material's engine material (decision 054): its factors, and
/// the base color, normal and metallic-roughness textures it names, those
/// loaded. Terrain reads height from the metallic-roughness map's alpha,
/// and occlusion from its red where the material's occlusion map is the
/// same texture.
#[derive(Clone, Debug)]
pub struct RenderSurfaceMaps {
    pub material: Material,
    pub base_color: Option<RenderTexture>,
    pub normal: Option<RenderTexture>,
    pub metallic_roughness: Option<RenderTexture>,
}

impl RenderSurfaceMaps {
    /// `material`'s maps from `assets`; each one named but missing counts
    /// in `missing`.
    fn of(material: Material, assets: &AssetLibrary, missing: &mut usize) -> Self {
        let mut texture = |map: Option<engine_assets::TextureRef>| {
            let handle = map?.texture;
            let found = assets.texture(handle).map(|texture| RenderTexture {
                handle,
                texture: Arc::clone(texture),
            });
            *missing += usize::from(found.is_none());
            found
        };
        Self {
            base_color: texture(material.base_color_texture()),
            normal: texture(material.normal_texture()),
            metallic_roughness: texture(material.metallic_roughness_texture()),
            material,
        }
    }
}

/// A placement asset of a frame (decision 057): its mesh, its material
/// and the material's textures, its coarser meshes with their errors, and
/// its sway (decision 058).
#[derive(Clone, Debug)]
pub struct RenderPlacementAsset {
    pub mesh: RenderMesh,
    pub material: RenderMaterial,
    pub textures: Vec<RenderTexture>,
    pub lods: Vec<(RenderMesh, f32)>,
    pub sway: f32,
}

impl RenderPlacementAsset {
    /// `asset` from `assets`, or `None` with what is missing counted in
    /// `missing`.
    fn of(asset: &PlacementAsset, assets: &AssetLibrary, missing: &mut usize) -> Option<Self> {
        let found = (|| {
            let mesh = assets.mesh(asset.mesh)?;
            let material = assets.material(asset.material)?;
            let lods = asset
                .lods
                .iter()
                .map(|lod| {
                    assets.mesh(lod.mesh).map(|mesh| {
                        (
                            RenderMesh {
                                handle: lod.mesh,
                                mesh: Arc::clone(mesh),
                            },
                            lod.error,
                        )
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            let textures = material
                .textures()
                .map(|handle| {
                    assets.texture(handle).map(|texture| RenderTexture {
                        handle,
                        texture: Arc::clone(texture),
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            Some(Self {
                mesh: RenderMesh {
                    handle: asset.mesh,
                    mesh: Arc::clone(mesh),
                },
                material: RenderMaterial {
                    handle: asset.material,
                    material: *material,
                    version: assets.material_version(asset.material).unwrap_or(0),
                },
                textures,
                lods,
                sway: asset.sway,
            })
        })();
        *missing += usize::from(found.is_none());
        found
    }
}

/// One terrain of a frame (decision 051).
#[derive(Clone, Debug)]
pub struct RenderTerrain {
    /// The terrain's entity bits, which renderers key its tiles by.
    pub id: u64,
    pub origin: Vec3,
    pub pyramid: Pyramid,
    /// How each of its surface's materials looks, by the index tiles'
    /// control values hold.
    pub surface: Vec<SurfaceMaterial>,
    /// Each surface material's engine material, where it names one that
    /// exists.
    pub maps: Vec<Option<RenderSurfaceMaps>>,
    /// What its placement layers stand on it (decision 057), by the asset
    /// indices its tiles' placed objects hold; `None` where the mesh or
    /// material is missing.
    pub assets: Vec<Option<RenderPlacementAsset>>,
    /// Its resident tiles: renderers upload those they lack and release
    /// the rest. The list changes only when `tiles_revision` advances.
    pub tiles: Arc<[Arc<Tile>]>,
    pub tiles_revision: u64,
    /// The patches the camera draws.
    pub patches: Vec<Patch>,
    /// Meters from the eye within which each level's tiles load, finest
    /// first: where renderers can shade with them.
    pub reach: Vec<f32>,
    /// Where its hydrology bake found water standing (decision 067).
    pub water: Option<Arc<engine_terrain::WaterGrid>>,
}

/// The frame's terrains (decision 051).
#[derive(Clone, Debug, Default)]
pub struct RenderTerrains {
    pub terrains: Vec<RenderTerrain>,
    /// The camera's view and projection without its translation: terrain
    /// draws camera-relative, from positions less the camera's.
    pub relative_view_projection: Mat4,
    pub stats: TerrainStats,
    /// What terrain shows ([`TerrainDebug`]).
    pub view: TerrainView,
}

/// Streams terrains and lists them in each frame.
pub(crate) struct TerrainExtraction {
    terrains: QueryState<(Entity, &'static Terrain)>,
    focus: QueryState<Entity, With<TerrainFocus>>,
    points: Vec<Vec3>,
}

impl TerrainExtraction {
    pub(crate) fn new(world: &mut World) -> Self {
        Self {
            terrains: world.query::<(Entity, &Terrain)>(),
            focus: world.query_filtered::<Entity, With<TerrainFocus>>(),
            points: Vec::new(),
        }
    }

    /// Follows added, changed and removed terrains, then streams them
    /// toward this frame's camera and focus entities.
    pub(crate) fn update(&mut self, world: &mut World) {
        let _profile = engine_profile::span("cpu", "terrain_stream");
        let settings = world
            .get_resource::<TerrainSettings>()
            .copied()
            .map(TerrainSettings::sanitized);
        let camera = *world.resource::<Camera>();
        let viewport = world
            .get_resource::<Viewport>()
            .copied()
            .unwrap_or_default();
        let eye = world
            .get_resource::<TerrainDebug>()
            .and_then(|debug| debug.freeze)
            .unwrap_or_else(|| camera.eye());
        self.points.clear();
        self.points.push(eye);
        if let Some(globals) = world.get_resource::<GlobalTransforms>() {
            for entity in self.focus.iter(world) {
                if let Some(point) = globals.get(entity).and_then(|global| global.translation()) {
                    self.points.push(point);
                }
            }
        }
        let present: Vec<(Entity, Terrain)> = self
            .terrains
            .iter(world)
            .map(|(entity, terrain)| (entity, terrain.clone()))
            .collect();
        let Some(mut terrains) = world.get_resource_mut::<Terrains>() else {
            return;
        };
        let terrains = terrains.as_mut();
        terrains
            .entries
            .retain(|state| present.iter().any(|(entity, _)| *entity == state.entity));
        for (entity, terrain) in present {
            match terrains
                .entries
                .iter_mut()
                .find(|state| state.entity == entity)
            {
                Some(state) => {
                    if !terrain.same(&state.terrain) {
                        *state = TerrainState::new(entity, terrain);
                    }
                }
                None => terrains.entries.push(TerrainState::new(entity, terrain)),
            }
        }
        let Some(settings) = settings else {
            for state in &mut terrains.entries {
                state.selection.patches.clear();
            }
            terrains.stats = TerrainStats {
                terrains: terrains.entries.len() as u32,
                ..TerrainStats::default()
            };
            return;
        };
        terrains.stream(&settings, &camera, eye, viewport, &self.points);
    }

    /// Lists every terrain's resident tiles and the camera's patches.
    pub(crate) fn extract(&mut self, world: &World, aspect: f32, snapshot: &mut RenderSnapshot) {
        let listing = &mut snapshot.terrain;
        let missing = &mut snapshot.missing_assets;
        let assets = world.get_resource::<AssetLibrary>();
        let Some(terrains) = world.get_resource::<Terrains>() else {
            listing.terrains.clear();
            listing.stats = TerrainStats::default();
            return;
        };
        let camera = world.resource::<Camera>();
        listing.relative_view_projection = camera.relative_view_projection(aspect);
        listing.stats = terrains.stats;
        listing.view = world
            .get_resource::<TerrainDebug>()
            .map_or(TerrainView::Shaded, |debug| debug.view);
        let enabled = world
            .get_resource::<TerrainSettings>()
            .is_some_and(|settings| settings.enabled);
        listing.terrains.truncate(terrains.entries.len());
        for (index, state) in terrains.entries.iter().enumerate() {
            if index == listing.terrains.len() {
                listing.terrains.push(RenderTerrain {
                    id: 0,
                    origin: Vec3::ZERO,
                    pyramid: state.terrain.pyramid,
                    surface: Vec::new(),
                    maps: Vec::new(),
                    assets: Vec::new(),
                    tiles: Arc::from(Vec::new()),
                    tiles_revision: 0,
                    patches: Vec::new(),
                    reach: Vec::new(),
                    water: None,
                });
            }
            let render = &mut listing.terrains[index];
            render.id = state.entity.to_bits();
            render.origin = state.terrain.origin;
            render.pyramid = state.terrain.pyramid;
            render.surface.clone_from(&state.terrain.surface.materials);
            render.assets.clear();
            for asset in &state.terrain.surface.assets {
                render.assets.push(
                    assets.and_then(|assets| RenderPlacementAsset::of(asset, assets, missing)),
                );
            }
            render.maps.clear();
            for look in &state.terrain.surface.materials {
                let material = look.material.and_then(|handle| {
                    let material =
                        assets.and_then(|assets| Some((assets, *assets.material(handle)?)));
                    *missing += usize::from(material.is_none());
                    material
                });
                render
                    .maps
                    .push(material.map(|(assets, material)| {
                        RenderSurfaceMaps::of(material, assets, missing)
                    }));
            }
            if render.tiles_revision != state.revision || render.id != state.entity.to_bits() {
                let mut tiles: Vec<Arc<Tile>> = state.tiles.values().cloned().collect();
                tiles.sort_unstable_by_key(|tile| tile.key());
                render.tiles = Arc::from(tiles);
                render.tiles_revision = state.revision;
            }
            render.patches.clear();
            if enabled {
                render.patches.extend_from_slice(&state.selection.patches);
            }
            render.water = state
                .baked
                .as_ref()
                .and_then(|layers| layers.hydrology())
                .map(|hydrology| Arc::clone(&hydrology.water));
            render.reach.clear();
            render
                .reach
                .extend(state.ranges.iter().map(|range| range * LOAD_REACH));
        }
    }
}
