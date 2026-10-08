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
//! entities for collision. Generating, decoding and baking share a frame's
//! work time (decision 099), larger until the camera's view first loads
//! whole than after; past a tile budget or the device's capacity the least
//! important tiles drop and count as dropped. Frames list each terrain's
//! resident tiles and the patches the camera draws in
//! [`RenderSnapshot::terrain`].
//!
//! Where the host runs workers (decision 110), terrains bake and generate
//! tiles on them instead: frames share each terrain's layers with them,
//! hand them the most important missing tiles and make the tiles they
//! finish resident, which takes frames little time.
//!
//! Tiles load in the order the view needs them (decision 112): the pinned
//! levels; the tiles the view draws and shows, with those under focus
//! entities; the tiles it draws out of sight, then collision's; those that
//! only shade what it shows; where the eye is headed; the rest. Distances
//! run to each tile's ground, so a high eye wants fine tiles only as near
//! as their ground is. Jobs carry that order as their priority, the first
//! tiers taking a third job per worker, which workers run first; jobs for
//! tiles no longer kept stop at once. Far tiles' objects standing far away
//! at their level are placed by jobs of their own after their tiles, so
//! the tiles under the view stop waiting for far forests.
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
use glam::{Mat4, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use engine_assets::{AssetLibrary, Material, MaterialHandle, MeshHandle};
use engine_terrain::jobs::{
    BAKE_JOB, BAND_JOB, FAR_JOB, PLACE_JOB, StackView, TILE_JOB, TerrainRecipe,
};
use engine_terrain::{
    Baked, Biome, BiomeError, BiomeLayer, BiomeRules, Biomes, Bridge, ErosionStage, ErosionStats,
    Fall, FileHeader, FormatError, Generator, HydrologyStage, HydrologyStats, Layer, Layers,
    LodRanges, Mask, NavOptions, NavPath, NoiseMask, Obstacle, PATCH_QUADS, PATCHES_PER_TILE,
    Patch, PathStage, Placing, Pyramid, Remap, Route, RouteOptions, Selection, Stamp, TILE_QUADS,
    TILE_STRIDE, TerrainBake, Tile, TileBand, TileKey, TileSample, WantedTile, WantedView,
    decode_tile, merge_placed,
};

use crate::environment::{InvalidSetting, check_budget, check_range};
use crate::{
    Camera, GlobalTransforms, IDLE_JOBS, JobId, Jobs, RenderMaterial, RenderMesh, RenderSnapshot,
    RenderTexture, URGENT_JOBS, VIEW_JOBS, content_hash, job_class, job_priority,
};

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
    /// The material its roads' bridges (decision 072) draw with; without
    /// one, a weathered stone.
    pub bridge: Option<MaterialHandle>,
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
    /// The meters its maps' heights span where relief (decision 082) sinks
    /// the ground between its stones and tufts: 0, flat, to 1. Top
    /// projections only.
    pub relief: f32,
    /// The meters tracks (decision 083) sink it at most: 0, rigid, to 2.
    pub deform: f32,
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
            relief: 0.0,
            deform: 0.0,
        }
    }

    /// Whether every value is in range: color 0 to 1, roughness 0.03 to
    /// 1, tiling 0.01 to 1000 meters, height contrast and relief 0 to 1,
    /// deform 0 to 2, and its cover's.
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
            && (0.0..=1.0).contains(&self.relief)
            && (0.0..=2.0).contains(&self.deform)
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
            bridge: None,
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
            placed: Vec::new(),
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
    /// Edits over its generated layers and their bakes (decision 080), in
    /// order: replacing them regenerates only the tiles under the edits
    /// added or taken away.
    pub edits: Arc<[TerrainEdit]>,
    /// Ground generated around it to the horizon (decision 094).
    pub surroundings: Option<Surroundings>,
}

/// Ground generated past a terrain's edges to the horizon (decision 094):
/// a coarser terrain around it from the same layers, its samples four
/// times as far apart as the terrain's finest, reaching about `reach`
/// meters past every edge (the terrain's inspection reports how far: its
/// tiles align with the terrain's sample grids). Its stamps, heightmaps,
/// bakes and roads fade into its noise over `blend` meters past the edges,
/// so the two meet where the terrain ends; it never collides, places
/// objects, grows blades or holds water. Terrains read from files have
/// none.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surroundings {
    pub reach: f32,
    pub blend: f32,
}

impl Surroundings {
    /// The most meters surroundings reach past a terrain's edges or blend
    /// over.
    pub const MAX_REACH: f32 = 1.0e6;

    /// Whether streaming can use them.
    pub fn is_usable(&self) -> bool {
        self.reach.is_finite()
            && self.reach > 0.0
            && self.reach <= Self::MAX_REACH
            && self.blend.is_finite()
            && (0.0..=Self::MAX_REACH).contains(&self.blend)
    }

    /// Where `terrain`'s surroundings lie: their origin, their pyramid and
    /// the rectangle of their ground (local) the terrain covers. Their
    /// tiles along a side are a power of two, as few as reach `reach` past
    /// the terrain's widest side while the terrain's edges stay on the
    /// edges of every level's patches, so the ground the terrain covers is
    /// left whole patches and both draw the same samples where they meet;
    /// `None` where no count fits.
    pub fn layout(&self, terrain: &Terrain) -> Option<(Vec3, Pyramid, (Vec2, Vec2))> {
        let base = terrain.pyramid;
        let extent = base.extent();
        let spacing = base.spacing(base.levels().saturating_sub(1).min(2));
        let tile = TILE_QUADS as f32 * spacing;
        let wanted = crate::math::max(extent.x, extent.y) + 2.0 * self.reach;
        let mut tiles = ((wanted / tile).ceil() as u32)
            .clamp(1, Pyramid::MAX_TILES)
            .next_power_of_two()
            .min(Pyramid::MAX_TILES);
        let whole = |length: f32, unit: f32| {
            let ratio = length / unit;
            ratio >= 1.0 && ratio == ratio.floor()
        };
        while tiles >= 1 {
            // The coarsest level's samples lie `tiles` times as far apart
            // as the finest's, its patches sixteen of them across.
            let patch = PATCH_QUADS as f32 * spacing * tiles as f32;
            let offset = (Vec2::splat(tiles as f32 * tile) - extent) * 0.5;
            if offset.x > 0.0
                && offset.y > 0.0
                && whole(offset.x, patch)
                && whole(offset.y, patch)
                && whole(extent.x, patch)
                && whole(extent.y, patch)
            {
                let pyramid = Pyramid::new([tiles, tiles], spacing).ok()?;
                let origin = terrain.origin - Vec3::new(offset.x, 0.0, offset.y);
                return Some((origin, pyramid, (offset, offset + extent)));
            }
            tiles /= 2;
        }
        None
    }
}

/// A terrain's surroundings as they stream (decision 094): how far they
/// reach past its edges along x and z, their level-0 tiles along a side and
/// sample spacing, and their resident tiles and the patches drawn.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SurroundingsStatus {
    pub reach: [f32; 2],
    pub tiles: u32,
    pub spacing: f32,
    pub resident_tiles: u32,
    pub patches: u32,
}

/// One edit of a generated terrain (decision 080): a stamp over its layers
/// and bakes (terrain-local), under an `id` that repeating it with leaves
/// alone.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainEdit {
    pub id: Arc<str>,
    pub stamp: Stamp,
}

impl Terrain {
    /// Whether `other` describes the same terrain: equal fields and the
    /// same shared source, so streaming keeps its tiles; edits apply over
    /// it tile by tile.
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
            edits: Arc::from(Vec::new()),
            surroundings: None,
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
            edits: Arc::from(Vec::new()),
            surroundings: None,
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
            edits: Arc::from(Vec::new()),
            surroundings: None,
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

    /// The terrain with ground generated around it to the horizon
    /// (decision 094).
    pub fn with_surroundings(mut self, surroundings: Surroundings) -> Self {
        self.surroundings = Some(surroundings);
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
    /// How much detail the ground shows (decision 090): green where all
    /// its maps show, yellow where their color alone does, red where their
    /// means stand in, blue where it shows as its cover's canopy.
    Detail,
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
    /// Work generated, decoded or baked between checks of the frame's work
    /// time, in samples' worth (67,000 a decoded tile, about 500,000 a
    /// generated one); a frame's whole work where hosts give terrain no
    /// clock.
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    pub samples_per_frame: u32,
    /// Milliseconds a frame works on terrain until the camera's view first
    /// loads whole (decision 099), and after, 0.1 to 1000 each: loading
    /// takes large slices of frames, play small ones.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.1, max = 1000.0)))]
    pub loading_work_ms: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.1, max = 1000.0)))]
    pub work_ms: f32,
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
    /// Meters from the camera within which the ground's relief (decision
    /// 082) shows, its view rays marching through its materials' heights,
    /// 0 (none) to 200; and the most steps a ray takes, 2 to 64.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 200.0)))]
    pub relief_distance: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 2, max = 64)))]
    pub relief_steps: u32,
    /// Ground generated around terrains with surroundings (decision 094)
    /// streams and draws; off, every terrain ends at its edges.
    pub surroundings: bool,
    /// Detail lowered while the camera moves fast (decision 115): off by
    /// default, as it lowers detail in motion on purpose.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub motion_detail: Option<MotionDetail>,
    /// Bytes of tiles kept in memory after the view leaves them (decision
    /// 116), about 600 KB each, to return at once when it comes back: the
    /// least recently used beyond drop, and 0 keeps none.
    pub cache_bytes: u64,
    /// Bytes of tiles becoming resident in a frame, which the GPU takes in
    /// that frame (decision 116), about 540 KB a tile: tiles beyond wait for
    /// the next, though each frame's first arrives whatever its size; at
    /// least 1.
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    pub upload_bytes: u32,
    /// Multiples of each level's range within which workers generate tiles
    /// ahead of the view into memory while nothing is missing (decision
    /// 116), 0 to 8; within the keeping reach (2.5) none.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 8.0)))]
    pub idle_reach: f32,
    /// The most bands of rows a tile under the view generates in at once
    /// on workers the other such tiles leave free (decision 117), 1 (whole
    /// tiles) to 16.
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 16)))]
    pub max_bands: u32,
    /// Tiles' surface materials from the GPU (decision 118): tiles generate
    /// without control values, which a compute pass gives each as it
    /// uploads, the rules' own but where GPU rounding crosses a threshold;
    /// gameplay reads the rules at the points it asks. Off, the CPU gives
    /// them.
    pub gpu_materials: bool,
}

/// How terrain loads while the camera moves fast (decision 115): the
/// finest levels' tiles in its view wait, more of them the faster it
/// moves, and tiles out of its view wait until it settles. The coarsest
/// pinned levels, collision's tiles and those under focus entities never
/// wait, and tiles resident stay.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneTerrainMotionDetail")
)]
#[serde(default, deny_unknown_fields)]
pub struct MotionDetail {
    /// Meters per second above which the camera moves fast, 0 to 100,000.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 100000.0)))]
    pub speed: f32,
    /// Meters per second at which the finest `levels` levels wait, from
    /// `speed` to 100,000; between, fewer wait, at least one.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 100000.0)))]
    pub fastest: f32,
    /// Finest levels whose tiles wait at `fastest`, 0 to 8.
    #[cfg_attr(feature = "schema", schemars(range(min = 0, max = 8)))]
    pub levels: u32,
    /// Seconds after the camera slows below `speed` before tiles out of its
    /// view load, 0 to 10.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 10.0)))]
    pub settle: f32,
}

impl Default for MotionDetail {
    fn default() -> Self {
        Self {
            speed: 30.0,
            fastest: 300.0,
            levels: 2,
            settle: 0.2,
        }
    }
}

impl MotionDetail {
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("motion_detail.speed", self.speed, 0.0..=100_000.0)?;
        check_range(
            "motion_detail.fastest",
            self.fastest,
            self.speed..=100_000.0,
        )?;
        if self.levels > 8 {
            return Err(InvalidSetting::new(
                "motion_detail.levels",
                "must be in [0, 8]",
            ));
        }
        check_range("motion_detail.settle", self.settle, 0.0..=10.0)
    }

    /// Every value in range.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        let finite = |value: f32, fallback: f32| if value.is_finite() { value } else { fallback };
        let speed = crate::math::clamp(finite(self.speed, defaults.speed), 0.0, 100_000.0);
        Self {
            speed,
            fastest: crate::math::clamp(finite(self.fastest, defaults.fastest), speed, 100_000.0),
            levels: self.levels.min(8),
            settle: crate::math::clamp(finite(self.settle, defaults.settle), 0.0, 10.0),
        }
    }

    /// The finest levels whose tiles wait at `speed` meters per second.
    pub fn waiting(&self, speed: f32) -> u32 {
        if speed <= self.speed || speed.is_nan() || self.levels == 0 {
            return 0;
        }
        let share = if self.fastest > self.speed {
            crate::math::clamp((speed - self.speed) / (self.fastest - self.speed), 0.0, 1.0)
        } else {
            1.0
        };
        ((share * self.levels as f32).ceil() as u32).clamp(1, self.levels)
    }
}

impl Default for TerrainSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            pixel_error: 6.0,
            pinned_levels: 2,
            max_tiles: None,
            samples_per_frame: 24_576,
            loading_work_ms: 50.0,
            work_ms: 4.0,
            collision_radius: 192.0,
            sun_shadows: true,
            shadow_resolution: 1024,
            shadow_softness: 0.01,
            sky_visibility: true,
            sky_resolution: 1024,
            relief_distance: 24.0,
            relief_steps: 8,
            surroundings: true,
            motion_detail: None,
            cache_bytes: 256 << 20,
            upload_bytes: 8 << 20,
            idle_reach: 3.0,
            max_bands: 4,
            gpu_materials: true,
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
        check_range("loading_work_ms", self.loading_work_ms, 0.1..=1000.0)?;
        check_range("work_ms", self.work_ms, 0.1..=1000.0)?;
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
        check_range("relief_distance", self.relief_distance, 0.0..=200.0)?;
        if !(2..=64).contains(&self.relief_steps) {
            return Err(InvalidSetting::new("relief_steps", "must be in [2, 64]"));
        }
        if let Some(detail) = &self.motion_detail {
            detail.validate()?;
        }
        if self.upload_bytes == 0 {
            return Err(InvalidSetting::new("upload_bytes", "must be at least 1"));
        }
        check_range("idle_reach", self.idle_reach, 0.0..=8.0)?;
        if !(1..=16).contains(&self.max_bands) {
            return Err(InvalidSetting::new("max_bands", "must be in [1, 16]"));
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
            loading_work_ms: crate::math::clamp(
                finite(self.loading_work_ms, defaults.loading_work_ms),
                0.1,
                1000.0,
            ),
            work_ms: crate::math::clamp(finite(self.work_ms, defaults.work_ms), 0.1, 1000.0),
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
            relief_distance: crate::math::clamp(
                finite(self.relief_distance, defaults.relief_distance),
                0.0,
                200.0,
            ),
            relief_steps: self.relief_steps.clamp(2, 64),
            surroundings: self.surroundings,
            motion_detail: self.motion_detail.map(MotionDetail::sanitized),
            cache_bytes: self.cache_bytes,
            upload_bytes: self.upload_bytes.max(1),
            idle_reach: crate::math::clamp(finite(self.idle_reach, defaults.idle_reach), 0.0, 8.0),
            max_bands: self.max_bands.clamp(1, 16),
            gpu_materials: self.gpu_materials,
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
    /// Tiles being generated, fetched or decoded, and every tile the
    /// camera's view and collision want that is not resident yet (decision
    /// 108).
    pub loading_tiles: u32,
    pub missing_tiles: u32,
    /// Of the resident tiles, those holding their shape alone so far
    /// (decision 114): heights and holes, drawn with coarser levels'
    /// materials while the rest loads.
    pub shaped_tiles: u32,
    /// Tiles the view wants that wait while the camera moves fast (decision
    /// 115).
    pub waiting_tiles: u32,
    /// Tiles kept in memory after the view left them (decision 116) and
    /// their bytes, tiles that returned from there since terrains started,
    /// and tiles generating ahead of the view.
    pub cached_tiles: u32,
    pub cached_bytes: u64,
    pub restored_tiles: u64,
    pub idle_tiles: u32,
    /// Tiles that generated in bands of rows on several workers since
    /// terrains started (decision 117).
    pub banded_tiles: u64,
    /// Wanted tiles beyond the tile budget or the device's capacity.
    pub dropped_tiles: u32,
    /// Patches the camera draws, and of those how many stand in for finer
    /// levels not resident yet.
    pub patches: u32,
    pub fallback_patches: u32,
    /// Terrains whose bakes (decisions 066 and 068) are in progress: they
    /// load no tiles until done.
    pub baking: u32,
    /// Bridges terrains' roads span (decision 072), tunnels they bore
    /// (decision 073), and waterfalls their rivers drop over (decision 075).
    pub bridges: u32,
    pub tunnels: u32,
    pub falls: u32,
    /// Of the resident tiles and patches, those of terrains' surroundings
    /// (decision 094).
    pub surroundings_tiles: u32,
    pub surroundings_patches: u32,
    /// Whether the camera's view has yet to load whole since terrains
    /// arrived or [`Terrains::begin_loading`] (decision 099): frames work
    /// on terrain for the settings' `loading_work_ms`, then `work_ms`.
    pub loading: bool,
}

/// A frame's terrain work (decision 099): slices of samples' worth handed
/// out while its time lasts, or a single slice without a clock.
struct Work {
    slice: u64,
    clock: Option<fn() -> f64>,
    end: f64,
    /// Bytes tiles may still bring the GPU this frame (decision 116).
    upload: std::cell::Cell<u64>,
    /// The most bands a tile under the view splits into (decision 117).
    bands: usize,
}

impl Work {
    /// Refills `budget` with a slice once spent, while the frame's time
    /// lasts; whether it did.
    fn refill(&self, budget: &mut u64) -> bool {
        if *budget > 0 {
            return false;
        }
        match self.clock {
            Some(clock) if clock() < self.end => {
                *budget = self.slice;
                true
            }
            _ => false,
        }
    }

    /// Whether the work's time is up; never without a clock.
    fn expired(&self) -> bool {
        self.clock.is_some_and(|clock| clock() >= self.end)
    }

    /// Whether tiles may still become resident within the frame's upload
    /// budget (decision 116).
    fn uploads(&self) -> bool {
        self.upload.get() > 0
    }

    /// Spends `bytes` of the frame's upload budget.
    fn upload(&self, bytes: u64) {
        self.upload.set(self.upload.get().saturating_sub(bytes));
    }
}

/// Loads `missing` tiles of `state`, most important first, within `work`'s
/// time from slices of `budget`: generated after the terrain's bakes,
/// decoded, or requested and decoded once delivered; `target` is every tile
/// it loads and `keep` every tile it keeps. Frames load what they find
/// missing, and hosts what is left between frames (decision 108). Where
/// `slots` has workers, they bake and generate instead (decision 110), each
/// tile at its job priority (decision 112). Returns whether a bake is still
/// in progress, which nothing loads behind.
#[allow(clippy::too_many_arguments)]
fn load_tiles(
    state: &mut TerrainState,
    missing: &[(TileKey, u32)],
    (target, keep): (&HashSet<TileKey>, &HashSet<TileKey>),
    work: &Work,
    budget: &mut u64,
    frame: u64,
    requests: &mut Vec<TileRequest>,
    mut slots: Option<&mut JobSlots<'_>>,
) -> bool {
    let pyramid = state.terrain.pyramid;
    match state.terrain.source.clone() {
        TerrainSource::Generated(authored) => {
            // Bakes (decisions 066 and 068) run before any tile loads,
            // each reading the layers below it, in stack order: on a
            // worker where workers run (decision 110), unless one failed
            // there.
            let layers = state.baked.clone().unwrap_or(authored);
            if let Some(unbaked) = layers.unbaked() {
                if let Some(slots) = slots.as_deref_mut()
                    && slots.jobs.workers() > 0
                    && !state.bake_inline
                {
                    bake_on_worker(state, slots.jobs, &layers, &pyramid);
                    return true;
                }
                // A bake handed to workers that stopped runs here.
                if let Some(id) = state.bake_job.take()
                    && let Some(slots) = slots.as_deref_mut()
                {
                    slots.jobs.cancel(id);
                }
                let bake = state
                    .bake
                    .get_or_insert_with(|| TerrainBake::new(&layers, unbaked.clone(), &pyramid));
                let _profile = engine_profile::span("cpu", "terrain_bake");
                let baked = loop {
                    if let Some(baked) = bake.advance(&pyramid, budget) {
                        break Some(baked);
                    }
                    if !work.refill(budget) {
                        break None;
                    }
                };
                if let Some(baked) = baked {
                    match &baked {
                        Baked::Hydrology(found) => state.hydrology = Some(found.stats),
                        Baked::Erosion(found) => state.erosion = Some(found.stats),
                        Baked::Path(_) => {}
                    }
                    let index = unbaked.0;
                    finish_bake(state, Arc::new(layers.with_baked(index, baked)));
                }
                return true;
            }
            // Edits (decision 080) stand over every bake.
            let layers = state.generation(layers);
            // Workers generate tiles where they run (decision 110), unless
            // they failed it again and again.
            if let Some(slots) = slots.as_deref_mut()
                && slots.jobs.workers() > 0
                && state.recipe.is_some()
                && state.worker_failures < WORKER_FAILURES
            {
                generate_on_workers(state, missing, (target, keep), work, slots, frame);
                return false;
            }
            // Tiles handed to workers that stopped generate here, whole.
            for (key, (id, _, _, _)) in state.dispatched.drain() {
                if let Some(slots) = slots.as_deref_mut() {
                    slots.jobs.cancel(id);
                }
                state.failed.remove(&key);
            }
            for (key, (id, _)) in state.far_jobs.drain() {
                if let Some(slots) = slots.as_deref_mut() {
                    slots.jobs.cancel(id);
                }
                state.far_wanted.insert(key, 0);
            }
            let banded = state
                .banded
                .drain()
                .flat_map(|(_, banding)| banding.jobs.into_iter().map(|(id, _)| id));
            let placing = state.placements.drain().map(|(_, (id, _, _, _))| id);
            for id in banded.chain(placing).collect::<Vec<_>>() {
                if let Some(slots) = slots.as_deref_mut() {
                    slots.jobs.cancel(id);
                }
            }
            for key in std::mem::take(&mut state.far_wanted).into_keys() {
                state.tiles.remove(&key);
                state.revision += 1;
            }
            // The coarsest level always draws: generate it whole at once.
            let first = state.tiles.is_empty();
            for &(key, _) in missing {
                let whole = first && u32::from(key.level) + 1 == pyramid.levels();
                if !whole && *budget == 0 && !work.refill(budget) {
                    break;
                }
                let rules = &state.terrain.surface.rules;
                // Tiles not resident hand out their shape first (decision
                // 114).
                let staged = !state.tiles.contains_key(&key);
                let materials = state.materials;
                let job = state.jobs.entry(key).or_insert_with(|| {
                    Generator::new(layers.clone(), &pyramid, key)
                        .with_surface(rules.clone())
                        .with_staged(staged)
                        .with_materials(materials)
                });
                let tile = if whole {
                    let mut unlimited = u64::MAX;
                    job.advance(&mut unlimited)
                } else {
                    loop {
                        if let Some(tile) = job.advance(budget) {
                            break Some(tile);
                        }
                        if !work.refill(budget) {
                            break None;
                        }
                    }
                };
                if let Some(tile) = tile {
                    state.jobs.remove(&key);
                    state.stale.remove(&key);
                    state.shaped.remove(&key);
                    state.insert(tile);
                } else if let Some(shape) = state.jobs.get_mut(&key).and_then(Generator::take_shape)
                {
                    state.shape(shape);
                }
            }
        }
        TerrainSource::File {
            header,
            url: _,
            data: Some(data),
        } => {
            // The whole file is here: decode within the frame's work,
            // then give each tile its surface.
            for &(key, _) in missing {
                if *budget == 0 && !work.refill(budget) {
                    break;
                }
                if state.surface_job(key, budget) {
                    continue;
                }
                let Some(entry) = header.tile(key) else {
                    continue;
                };
                let range = usize::try_from(entry.offset)
                    .ok()
                    .and_then(|start| data.get(start..start.checked_add(entry.length as usize)?));
                *budget = budget.saturating_sub(u64::from(TILE_STRIDE * TILE_STRIDE));
                match range.map(|bytes| decode_tile(&header, key, bytes)) {
                    Some(Ok(tile)) => state.resurface(tile, budget),
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
            for &(key, _) in missing {
                if *budget > 0 || work.refill(budget) {
                    state.surface_job(key, budget);
                }
                if state.in_flight.contains(&key)
                    || state.jobs.contains_key(&key)
                    || (state.tiles.contains_key(&key) && !state.shaped.contains(&key))
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
                requests.push(TileRequest {
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
                if *budget == 0 {
                    work.refill(budget);
                }
                if *budget == 0 || !target.contains(&key) {
                    if target.contains(&key) {
                        later.push((key, bytes));
                    }
                    continue;
                }
                *budget = budget.saturating_sub(u64::from(TILE_STRIDE * TILE_STRIDE));
                match bytes.map(|bytes| decode_tile(&header, key, &bytes)) {
                    Some(Ok(tile)) => {
                        state.failed.remove(&key);
                        state.resurface(tile, budget);
                    }
                    _ => {
                        state.failed.insert(key, frame);
                    }
                }
            }
            state.arrivals = later;
        }
    }
    false
}

/// Workers a frame hands terrain jobs to (decision 110): the jobs, and how
/// many more it may hand out, of any priority and urgent (decision 112).
struct JobSlots<'a> {
    jobs: &'a mut Jobs,
    slots: usize,
    urgent: usize,
}

impl JobSlots<'_> {
    /// The frame's jobs: two in flight per worker, so each has its next
    /// ready, and a third for urgent jobs, which workers take up first;
    /// less those still running.
    fn of(jobs: &mut Jobs) -> JobSlots<'_> {
        let workers = jobs.workers() as usize;
        let running = jobs.running();
        JobSlots {
            slots: (workers * 2).saturating_sub(running),
            urgent: (workers * 3).saturating_sub(running),
            jobs,
        }
    }

    /// Takes a slot for a job of `priority`; whether one was free.
    fn take(&mut self, priority: u32) -> bool {
        let free = if job_class(priority) == URGENT_JOBS {
            self.urgent
        } else {
            self.slots
        };
        if free == 0 {
            return false;
        }
        self.slots = self.slots.saturating_sub(1);
        self.urgent = self.urgent.saturating_sub(1);
        true
    }

    /// Gives back the slot of a job cancelled while running.
    fn release(&mut self) {
        self.slots += 1;
        self.urgent += 1;
    }
}

/// A terrain's recipe as workers hold it (decision 110): its key, the
/// stack it shares (the layers its tiles generate from, which surroundings
/// see from afar) and its revision, which its tiles' jobs carry.
struct SharedRecipe {
    key: String,
    stack: Arc<Layers>,
    revision: u64,
    materials: bool,
}

/// Makes `baked` the layers `state`'s tiles generate from: once every bake
/// is done, with its roads' bridges (decision 072; collision follows the
/// revision) and its rivers' waterfalls (decision 075).
fn finish_bake(state: &mut TerrainState, baked: Arc<Layers>) {
    if baked.unbaked().is_none() {
        state.bridges = baked
            .layers()
            .iter()
            .filter_map(|layer| match layer {
                Layer::Path(PathStage {
                    baked: Some(profile),
                    ..
                }) => Some(profile.bridges()),
                _ => None,
            })
            .flatten()
            .collect::<Vec<_>>()
            .into();
        state.falls = baked
            .hydrology()
            .map_or_else(|| Arc::from(Vec::new()), |water| water.falls.clone());
        state.revision += 1;
    }
    state.baked = Some(baked);
    state.bake = None;
}

/// Bakes `layers` over `pyramid` on a worker (decision 110): hands it every
/// bake at once, then makes what it baked the layers tiles generate from. A
/// bake that failed there runs in place instead.
fn bake_on_worker(state: &mut TerrainState, jobs: &mut Jobs, layers: &Layers, pyramid: &Pyramid) {
    let Some(id) = state.bake_job else {
        let input = engine_terrain::jobs::bake_input(layers, pyramid);
        let priority = job_priority(URGENT_JOBS, 0);
        // Hosts may keep the baked layers (decision 116), by the input's
        // hash.
        let identity = content_hash(&input).to_le_bytes().to_vec();
        state.bake_job = Some(jobs.submit_cacheable(BAKE_JOB, input, priority, identity));
        return;
    };
    let Some(output) = jobs.take(id) else {
        return;
    };
    state.bake_job = None;
    let Some(baked) = output
        .ok()
        .and_then(|bytes| engine_terrain::jobs::bake_output(&bytes).ok())
    else {
        state.bake_inline = true;
        return;
    };
    for layer in baked.layers() {
        match layer {
            Layer::Hydrology(HydrologyStage {
                baked: Some(found), ..
            }) => state.hydrology = Some(found.stats),
            Layer::Erosion(ErosionStage {
                baked: Some(found), ..
            }) => state.erosion = Some(found.stats),
            _ => {}
        }
    }
    finish_bake(state, Arc::new(baked));
}

/// Generates `state`'s missing tiles on workers (decision 110), most
/// important first, while the frame may hand out jobs; makes the tiles
/// workers finished resident, the first at once and the rest within
/// `work`'s time, when still wanted and generated from the layers tiles
/// generate from now. Tiles that failed are asked for again later. Jobs
/// for tiles `keep` no longer holds, or from layers since replaced, stop
/// (decision 112), and tiles of levels with objects standing far away
/// arrive without them: jobs of their own place those after, merged into
/// the tile in the order one pass places them.
fn generate_on_workers(
    state: &mut TerrainState,
    missing: &[(TileKey, u32)],
    (target, keep): (&HashSet<TileKey>, &HashSet<TileKey>),
    work: &Work,
    slots: &mut JobSlots<'_>,
    frame: u64,
) {
    let Some((recipe, revision)) = state
        .recipe
        .as_ref()
        .map(|recipe| (recipe.key.clone(), recipe.revision))
    else {
        return;
    };
    // Tiles generating here stop: workers generate them.
    state.jobs.clear();
    // Work no longer wanted stops, so its workers take up what is.
    state.dispatched.retain(|key, &mut (id, made, _, _)| {
        let wanted = made == revision && keep.contains(key);
        if !wanted {
            slots.jobs.cancel(id);
        }
        wanted
    });
    let tiles = &state.tiles;
    state.far_jobs.retain(|key, (id, tile)| {
        let wanted =
            keep.contains(key) && tiles.get(key).is_some_and(|held| Arc::ptr_eq(held, tile));
        if !wanted {
            slots.jobs.cancel(*id);
        }
        wanted
    });
    state
        .far_wanted
        .retain(|key, _| keep.contains(key) && tiles.contains_key(key));
    state.banded.retain(|key, banding| {
        let wanted = banding.made == revision && keep.contains(key);
        if !wanted {
            for &(id, ref band) in &banding.jobs {
                if band.is_none() {
                    slots.jobs.cancel(id);
                }
            }
        }
        wanted
    });
    state.placements.retain(|key, &mut (id, made, _, _)| {
        let wanted = made == revision && keep.contains(key) && tiles.contains_key(key);
        if !wanted {
            slots.jobs.cancel(id);
        }
        wanted
    });
    let handed: Vec<(TileKey, (JobId, u64, u32, Placing))> = state
        .dispatched
        .iter()
        .map(|(&key, &job)| (key, job))
        .collect();
    let rules = state.terrain.surface.rules.clone();
    let mut inserted = false;
    for (key, (id, made, priority, placing)) in handed {
        if inserted && (work.expired() || !work.uploads()) {
            break;
        }
        let Some(output) = slots.jobs.take(id) else {
            continue;
        };
        state.dispatched.remove(&key);
        let tile = output
            .ok()
            .and_then(|bytes| engine_terrain::jobs::tile_output(&bytes).ok());
        match tile {
            Some(tile) if made == revision && target.contains(&key) && tile.key() == key => {
                // After its shape a tile uploads its control values alone
                // (decision 116).
                work.upload(if state.shaped.contains(&key) {
                    TILE_ARRAY_BYTES
                } else {
                    2 * TILE_ARRAY_BYTES
                });
                state.stale.remove(&key);
                state.shaped.remove(&key);
                state.failed.remove(&key);
                state.far_jobs.remove(&key);
                state.far_wanted.remove(&key);
                if placing == Placing::Near {
                    state.far_wanted.insert(key, priority);
                }
                state.insert(tile);
                inserted = true;
            }
            Some(_) => {}
            None => {
                state.failed.insert(key, frame);
                state.worker_failures += 1;
            }
        }
    }
    // Tiles generating in bands (decision 117) join once every band is
    // done: resident at once with their materials, their objects placed by
    // a job after.
    let joining: Vec<TileKey> = state.banded.keys().copied().collect();
    for key in joining {
        if inserted && (work.expired() || !work.uploads()) {
            break;
        }
        let Some(banding) = state.banded.get_mut(&key) else {
            continue;
        };
        let mut failed = false;
        for (id, band) in &mut banding.jobs {
            if band.is_none()
                && let Some(output) = slots.jobs.take(*id)
            {
                match output
                    .ok()
                    .and_then(|bytes| engine_terrain::jobs::band_output(&bytes).ok())
                {
                    Some(done) => *band = Some(done),
                    None => failed = true,
                }
            }
        }
        if !failed && banding.jobs.iter().any(|(_, band)| band.is_none()) {
            continue;
        }
        let Some(banding) = state.banded.remove(&key) else {
            continue;
        };
        let (mut bands, mut running) = (Vec::new(), Vec::new());
        for (id, band) in banding.jobs {
            match band {
                Some(band) => bands.push(band),
                None => running.push(id),
            }
        }
        let spacing = state.terrain.pyramid.spacing(u32::from(key.level));
        let tile = (!failed)
            .then(|| engine_terrain::merge_bands(key, spacing, &bands))
            .flatten();
        let Some(tile) = tile else {
            for id in running {
                slots.jobs.cancel(id);
                slots.release();
            }
            state.failed.insert(key, frame);
            state.worker_failures += 1;
            continue;
        };
        if banding.made != revision || !target.contains(&key) {
            continue;
        }
        work.upload(2 * TILE_ARRAY_BYTES);
        state.banded_tiles += 1;
        state.stale.remove(&key);
        state.failed.remove(&key);
        if rules.places(key.level, banding.placing) {
            let input = engine_terrain::jobs::place_input(&recipe, &tile, banding.placing);
            let identity = engine_terrain::jobs::place_identity(&recipe, key, banding.placing);
            let id = slots
                .jobs
                .submit_cacheable(PLACE_JOB, input, banding.priority, identity);
            state
                .placements
                .insert(key, (id, revision, banding.priority, banding.placing));
            state.shaped.insert(key);
        } else {
            state.shaped.remove(&key);
        }
        state.insert(tile);
        inserted = true;
    }
    // Then their objects (decision 117).
    let placed: Vec<(TileKey, (JobId, u64, u32, Placing))> = state
        .placements
        .iter()
        .map(|(&key, &job)| (key, job))
        .collect();
    for (key, (id, made, priority, placing)) in placed {
        if inserted && work.expired() {
            break;
        }
        let Some(output) = slots.jobs.take(id) else {
            continue;
        };
        state.placements.remove(&key);
        let Some(tile) = state.tiles.get(&key).cloned() else {
            continue;
        };
        match output
            .ok()
            .and_then(|bytes| engine_terrain::jobs::far_output(&bytes).ok())
        {
            Some(objects) if made == revision && target.contains(&key) => {
                state.shaped.remove(&key);
                if placing == Placing::Near {
                    state.far_wanted.insert(key, priority);
                }
                state.insert(Tile::clone(&tile).with_placed(objects));
                inserted = true;
            }
            Some(_) => {}
            None => {
                state.failed.insert(key, frame);
                state.worker_failures += 1;
            }
        }
    }
    // Tiles still generating hand out their shape first (decision 114): it
    // draws, with coarser levels' materials, and collides until the whole
    // tile arrives.
    let shaping: Vec<(TileKey, JobId, u64)> = state
        .dispatched
        .iter()
        .map(|(&key, &(id, made, _, _))| (key, id, made))
        .collect();
    for (key, id, made) in shaping {
        if inserted && (work.expired() || !work.uploads()) {
            break;
        }
        let Some(part) = slots.jobs.take_part(id) else {
            continue;
        };
        if let Ok(shape) = engine_terrain::jobs::tile_output(&part)
            && made == revision
            && target.contains(&key)
            && shape.key() == key
            && state.shape(shape)
        {
            // Its heights, and its ancestor's control values for cover.
            work.upload(2 * TILE_ARRAY_BYTES);
            inserted = true;
        }
    }
    let placing: Vec<(TileKey, JobId)> = state
        .far_jobs
        .iter()
        .map(|(&key, &(id, _))| (key, id))
        .collect();
    for (key, id) in placing {
        if inserted && work.expired() {
            break;
        }
        let Some(output) = slots.jobs.take(id) else {
            continue;
        };
        let Some((_, tile)) = state.far_jobs.remove(&key) else {
            continue;
        };
        match output
            .ok()
            .and_then(|bytes| engine_terrain::jobs::far_output(&bytes).ok())
        {
            Some(far) => {
                let placed = merge_placed(tile.placed(), &far);
                state.insert(Tile::clone(&tile).with_placed(placed));
                inserted = true;
            }
            None => {
                state.far_wanted.insert(key, job_priority(VIEW_JOBS, 0));
                state.worker_failures += 1;
            }
        }
    }
    // Tiles generated ahead of need go to memory (decision 116); with tiles
    // missing, or out of the idle reach, their jobs stop.
    let idle: Vec<(TileKey, JobId, u64)> = state
        .idle
        .iter()
        .map(|(&key, &(id, made))| (key, id, made))
        .collect();
    for (key, id, made) in idle {
        if let Some(output) = slots.jobs.take(id) {
            state.idle.remove(&key);
            if let Some(tile) = output
                .ok()
                .and_then(|bytes| engine_terrain::jobs::tile_output(&bytes).ok())
                && made == revision
                && tile.key() == key
                && !state.tiles.contains_key(&key)
            {
                state.cache_tile(key, Arc::new(tile), frame);
            }
        } else if !missing.is_empty() || made != revision || !state.ahead.contains(&key) {
            slots.jobs.cancel(id);
            slots.release();
            state.idle.remove(&key);
        }
    }
    // Then what is missing, most urgent first: tiles, and the objects
    // standing far away on tiles that arrived without them, ranked after
    // the urgent tiles where their tiles ranked.
    let mut wanted: Vec<(u32, TileKey, bool)> = missing
        .iter()
        .filter(|(key, _)| {
            !state.dispatched.contains_key(key)
                && !state.banded.contains_key(key)
                && !state.placements.contains_key(key)
                && !state
                    .failed
                    .get(key)
                    .is_some_and(|&at| frame < at + RETRY_AFTER)
        })
        .map(|&(key, priority)| (priority, key, false))
        .collect();
    wanted.extend(
        state
            .far_wanted
            .iter()
            .filter(|(key, _)| !state.far_jobs.contains_key(key))
            .map(|(&key, &priority)| (job_priority(VIEW_JOBS, priority & 0xff_ffff), key, true)),
    );
    wanted.sort_by_key(|&(priority, key, far)| (priority, far, key));
    // Tiles under the view split into bands of rows across the slots the
    // urgent tiles left share (decision 117): each arrives sooner, though
    // workers do a little more in all.
    let mut urgent_left = wanted
        .iter()
        .filter(|&&(priority, _, far)| !far && job_class(priority) == URGENT_JOBS)
        .count();
    for (priority, key, far) in wanted {
        if !slots.take(priority) {
            break;
        }
        if far {
            let Some(tile) = state.tiles.get(&key).cloned() else {
                continue;
            };
            let input = engine_terrain::jobs::far_input(&recipe, &tile);
            let identity = engine_terrain::jobs::far_identity(&recipe, key);
            let id = slots
                .jobs
                .submit_cacheable(FAR_JOB, input, priority, identity);
            state.far_wanted.remove(&key);
            state.far_jobs.insert(key, (id, tile));
        } else {
            // A tile replacing one resident, as after an edit, comes whole,
            // so its objects standing far away never leave the view; one not
            // resident hands out its shape first (decision 114).
            let resident = state.tiles.contains_key(&key);
            let placing = if rules.places(key.level, Placing::Far)
                && (!resident || state.shaped.contains(&key))
            {
                Placing::Near
            } else {
                Placing::All
            };
            let urgent = job_class(priority) == URGENT_JOBS;
            urgent_left = urgent_left.saturating_sub(usize::from(urgent));
            let bands = if urgent && !resident {
                std::cmp::min(1 + slots.urgent / (urgent_left + 1), work.bands)
            } else {
                1
            };
            if bands > 1 {
                let size = (TILE_STRIDE as usize).div_ceil(bands);
                let mut jobs = Vec::with_capacity(bands);
                for band in 0..bands {
                    if band > 0 && !slots.take(priority) {
                        break;
                    }
                    let end = std::cmp::min((band + 1) * size, TILE_STRIDE as usize);
                    let rows = (band * size) as u32..end as u32;
                    let input = engine_terrain::jobs::band_input(&recipe, key, rows);
                    let id = slots
                        .jobs
                        .submit_cacheable(BAND_JOB, input.clone(), priority, input);
                    jobs.push((id, None));
                }
                // Unless every band found a slot, the tile generates whole.
                let covered = jobs.len() * size;
                if covered < TILE_STRIDE as usize {
                    for (id, _) in jobs.drain(..) {
                        slots.jobs.cancel(id);
                        slots.release();
                    }
                } else {
                    state.banded.insert(
                        key,
                        Banding {
                            made: revision,
                            priority,
                            placing,
                            jobs,
                        },
                    );
                    continue;
                }
            }
            let input = engine_terrain::jobs::tile_input(&recipe, key, placing, !resident);
            // Staged or not, the same tile (decision 116).
            let identity = engine_terrain::jobs::tile_input(&recipe, key, placing, false);
            let id = slots
                .jobs
                .submit_cacheable(TILE_JOB, input, priority, identity);
            state
                .dispatched
                .insert(key, (id, revision, priority, placing));
        }
    }
    // With nothing else to hand out, a job a worker generates the nearest
    // tiles within the idle reach (decision 116), whole, into memory.
    if missing.is_empty()
        && state.dispatched.is_empty()
        && state.banded.is_empty()
        && state.placements.is_empty()
        && state.far_wanted.is_empty()
        && state.far_jobs.is_empty()
    {
        let workers = slots.jobs.workers() as usize;
        for index in 0..state.ahead.len() {
            if state.idle.len() >= workers {
                break;
            }
            let key = state.ahead[index];
            if state.idle.contains_key(&key) {
                continue;
            }
            let priority = job_priority(IDLE_JOBS, index as u32);
            if !slots.take(priority) {
                break;
            }
            let input = engine_terrain::jobs::tile_input(&recipe, key, Placing::All, false);
            let id = slots
                .jobs
                .submit_cacheable(TILE_JOB, input.clone(), priority, input);
            state.idle.insert(key, (id, revision));
        }
    }
}

/// Shares the recipe workers generate `state`'s tiles from (decision 110)
/// once its bakes are done, and again whenever the layers its tiles
/// generate from change: their stack, shared once for every terrain seeing
/// it, and how its terrain sees it. Shares are named by their content
/// (decision 116), so jobs reading them are the same bytes in every
/// session; `shares` counts recipes for their revisions, and the key of
/// the recipe replaced goes to `released`.
fn share_recipe(
    state: &mut TerrainState,
    jobs: &mut Jobs,
    (stacks, shares): (&mut Vec<(Arc<Layers>, String)>, &mut u64),
    released: &mut Vec<String>,
) {
    let TerrainSource::Generated(authored) = &state.terrain.source else {
        return;
    };
    let layers = state.baked.clone().unwrap_or_else(|| authored.clone());
    if layers.unbaked().is_some() {
        return;
    }
    // Surroundings (decision 094) see their terrain's stack from afar.
    let (stack, view) = match &state.around {
        Some(around) => (
            around.base.clone(),
            Some(StackView {
                offset: -around.cutout.0,
                low: Vec2::ZERO,
                high: around.cutout.1 - around.cutout.0,
                blend: around.surroundings.blend,
            }),
        ),
        None => (state.generation(layers), None),
    };
    if state.recipe.as_ref().is_some_and(|recipe| {
        Arc::ptr_eq(&recipe.stack, &stack) && recipe.materials == state.materials
    }) {
        return;
    }
    let shared = stacks
        .iter()
        .find(|(shared, _)| Arc::ptr_eq(shared, &stack))
        .map(|(_, key)| key.clone());
    let stack_key = match shared {
        Some(key) => key,
        None => {
            let bytes = engine_terrain::jobs::share_stack(&stack);
            let key = format!("stack-{:032x}", content_hash(&bytes));
            if !jobs.is_shared(&key) {
                jobs.share(key.clone(), bytes);
            }
            stacks.push((stack.clone(), key.clone()));
            key
        }
    };
    *shares += 1;
    let recipe = TerrainRecipe {
        stack: stack_key,
        view,
        pyramid: state.terrain.pyramid,
        surface: state.terrain.surface.rules.authored().clone(),
        materials: state.materials,
    };
    let bytes = engine_terrain::jobs::share_recipe(&recipe);
    let key = format!("recipe-{:032x}", content_hash(&bytes));
    if !jobs.is_shared(&key) {
        jobs.share(key.clone(), bytes);
    }
    let shared = SharedRecipe {
        key,
        stack,
        revision: *shares,
        materials: state.materials,
    };
    if let Some(old) = state.recipe.replace(shared) {
        released.push(old.key);
    }
}

/// Pushes `key` onto `order` after those of its ancestors not yet there,
/// coarsest first (decision 108): a tile draws only within its parent, so a
/// finer one loaded first would wait unused, while what stands on the
/// ground (blades, placed objects) already followed its finer heights.
fn push_with_ancestors(
    pyramid: &engine_terrain::Pyramid,
    key: TileKey,
    order: &mut Vec<TileKey>,
    seen: &mut HashSet<TileKey>,
) {
    if seen.contains(&key) {
        return;
    }
    let mut chain = vec![key];
    let mut child = key;
    while let Some(parent) = pyramid.parent(child) {
        if seen.contains(&parent) {
            break;
        }
        chain.push(parent);
        child = parent;
    }
    for key in chain.into_iter().rev() {
        seen.insert(key);
        order.push(key);
    }
}

/// Frames a failed fetch waits before it is requested again.
const RETRY_AFTER: u64 = 120;
/// Tiles failing on workers (decision 110) after which a terrain generates
/// its own, as when the workers' module does not match the engine's.
const WORKER_FAILURES: u32 = 8;
/// Ranges, as multiples of each level's, within which tiles load and stay:
/// beyond its range a level's tiles still shade the coarser level's
/// patches (their normals), so they load further than they draw.
const LOAD_REACH: f32 = 2.0;
const KEEP_REACH: f32 = 2.5;
/// Tiles also load around where the eye is headed (decision 100): its
/// velocity carried this many seconds on, at most this many meters; the
/// eye moving farther in a frame jumped there.
const AHEAD_SECONDS: f32 = 2.0;
const AHEAD_MAX: f32 = 1000.0;
const JUMP: f32 = 100.0;

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
    /// Its roads' bridges (decision 072), terrain-local, once every bake
    /// is done, and its rivers' waterfalls (decision 075).
    bridges: Arc<[Bridge]>,
    falls: Arc<[Fall]>,
    /// The layers its tiles generate from with its edits over them
    /// (decision 080), and the baked layers they were made over; and the
    /// resident tiles under edits since, which draw until regenerated.
    edited: Option<(Arc<Layers>, Arc<Layers>)>,
    stale: HashSet<TileKey>,
    /// Resident tiles holding their shape alone so far (decision 114):
    /// heights and holes, drawn with coarser levels' materials and collided
    /// with while the rest loads.
    shaped: HashSet<TileKey>,
    /// For a terrain's surroundings (decision 094), what they were drawn
    /// from.
    around: Option<Around>,
    /// The tiles the last frame found missing, most important first with
    /// their jobs' priorities (decision 112), every tile it loads and every
    /// tile it keeps: what loads between frames (decision 108).
    missing: Vec<(TileKey, u32)>,
    target: HashSet<TileKey>,
    keep: HashSet<TileKey>,
    /// Workers' share (decision 110): the recipe they generate its tiles
    /// from, the tiles generating there (each one's job, the recipe's
    /// revision it generates from and its priority) and its bakes running
    /// there, unless they failed there; and the tiles that failed there.
    recipe: Option<SharedRecipe>,
    dispatched: HashMap<TileKey, (JobId, u64, u32, Placing)>,
    /// Resident tiles without their objects standing far away (decision
    /// 112), with their tiles' priorities, and the jobs placing those, each
    /// with the tile it places on.
    far_wanted: HashMap<TileKey, u32>,
    far_jobs: HashMap<TileKey, (JobId, Arc<Tile>)>,
    bake_job: Option<JobId>,
    bake_inline: bool,
    worker_failures: u32,
    /// Tiles that left whole (decision 116), kept in memory to return at
    /// once while the layers they generated from stay, and their bytes.
    cache: HashMap<TileKey, CachedTile>,
    cached_bytes: u64,
    /// Tiles generating ahead of the view on workers while nothing is
    /// missing (decision 116), for memory: each one's job and the recipe's
    /// revision it generates from; and the tiles to generate so, nearest
    /// first.
    idle: HashMap<TileKey, (JobId, u64)>,
    ahead: Vec<TileKey>,
    /// Tiles generating in bands of rows on several workers (decision
    /// 117); tiles their bands made whose objects a job places, each with
    /// that job, the recipe's revision, its priority and its placing; and
    /// how many tiles generated so.
    banded: HashMap<TileKey, Banding>,
    placements: HashMap<TileKey, (JobId, u64, u32, Placing)>,
    banded_tiles: u64,
    /// The layers its tiles generate from and its surface's data for the
    /// GPU from them (decision 118), `None` where the program needs more
    /// than the GPU's interpreter holds; and whether its tiles hold control
    /// values, as without GPU materials.
    surface: Option<(Arc<Layers>, Option<Arc<engine_terrain::gpu::SurfaceData>>)>,
    materials: bool,
}

/// A tile generating in bands of rows (decision 117): the recipe's revision
/// and the priority it generates at, the objects placed after, and its
/// bands' jobs, each with its band once done.
struct Banding {
    made: u64,
    priority: u32,
    placing: Placing,
    jobs: Vec<(JobId, Option<TileBand>)>,
}

/// A tile kept in memory after it left the view (decision 116): its bytes
/// there and the frame it left.
struct CachedTile {
    tile: Arc<Tile>,
    bytes: u64,
    left: u64,
}

/// Bytes of one of a tile's arrays on the GPU, heights or control values:
/// what a tile arriving uploads, once or twice (decision 116).
const TILE_ARRAY_BYTES: u64 = (TILE_STRIDE as u64) * (TILE_STRIDE as u64) * 4;

/// Bytes `tile` holds in memory (decision 116).
fn memory_bytes(tile: &Tile) -> u64 {
    let arrays = tile.heights().len() + tile.control().map_or(0, <[u32]>::len);
    let holes = tile.hole_bits().map_or(0, <[u64]>::len) * 8;
    (arrays * 4 + holes + std::mem::size_of_val(tile.placed())) as u64
}

/// What a terrain's surroundings (decision 094) were drawn from: the layers
/// its tiles generate from and its settings for them; and the rectangle of
/// their own ground (local) the terrain covers.
struct Around {
    base: Arc<Layers>,
    surroundings: Surroundings,
    cutout: (Vec2, Vec2),
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
            bridges: Arc::from(Vec::new()),
            falls: Arc::from(Vec::new()),
            edited: None,
            stale: HashSet::new(),
            shaped: HashSet::new(),
            around: None,
            missing: Vec::new(),
            target: HashSet::new(),
            keep: HashSet::new(),
            recipe: None,
            dispatched: HashMap::new(),
            far_wanted: HashMap::new(),
            far_jobs: HashMap::new(),
            bake_job: None,
            bake_inline: false,
            worker_failures: 0,
            cache: HashMap::new(),
            cached_bytes: 0,
            idle: HashMap::new(),
            ahead: Vec::new(),
            banded: HashMap::new(),
            placements: HashMap::new(),
            banded_tiles: 0,
            surface: None,
            materials: true,
        }
    }

    /// Whether its tiles hold control values, as `gpu` (the settings' GPU
    /// materials, decision 118) and its surface allow: generated terrains
    /// whose surface the GPU's interpreter holds leave them to the GPU.
    /// Tiles resident when that changes regenerate, drawn until they do.
    fn update_materials(&mut self, gpu: bool) {
        let TerrainSource::Generated(authored) = &self.terrain.source else {
            self.materials = true;
            return;
        };
        let layers = self.baked.clone().unwrap_or_else(|| authored.clone());
        let materials = if gpu && layers.unbaked().is_none() {
            let layers = self.generation(layers);
            let current = self
                .surface
                .as_ref()
                .is_some_and(|(held, _)| Arc::ptr_eq(held, &layers));
            if !current {
                let data =
                    engine_terrain::gpu::SurfaceData::new(&self.terrain.surface.rules, &layers)
                        .map(Arc::new);
                self.surface = Some((layers, data));
            }
            self.surface.as_ref().is_none_or(|(_, data)| data.is_none())
        } else {
            true
        };
        if materials != self.materials {
            self.materials = materials;
            let resident: Vec<TileKey> = self.tiles.keys().copied().collect();
            self.stale.extend(resident);
            let kept: Vec<TileKey> = self.cache.keys().copied().collect();
            for key in kept {
                self.uncache(key);
            }
            self.revision += 1;
        }
    }

    /// Its workers' jobs and recipe, for the next frame with workers to
    /// drop (decision 110) once the state is gone.
    fn retire(&self, retired: &mut Retired) {
        retired
            .jobs
            .extend(self.dispatched.values().map(|&(id, _, _, _)| id));
        retired
            .jobs
            .extend(self.far_jobs.values().map(|&(id, _)| id));
        retired.jobs.extend(self.bake_job);
        retired.jobs.extend(self.idle.values().map(|&(id, _)| id));
        retired.jobs.extend(
            self.banded
                .values()
                .flat_map(|banding| banding.jobs.iter().map(|&(id, _)| id)),
        );
        retired
            .jobs
            .extend(self.placements.values().map(|&(id, _, _, _)| id));
        retired
            .recipes
            .extend(self.recipe.as_ref().map(|recipe| recipe.key.clone()));
    }

    /// The id renderers key its tiles by: its entity's bits, the top bit
    /// flipped for surroundings (decision 094).
    fn render_id(&self) -> u64 {
        self.entity.to_bits() ^ (u64::from(self.around.is_some()) << 63)
    }

    /// Whether terrain-local `local` lies in the ground surroundings
    /// (decision 094) leave to the terrain they surround.
    fn cut_out(&self, local: Vec2) -> bool {
        self.around.as_ref().is_some_and(|around| {
            let (low, high) = around.cutout;
            local.x > low.x && local.y > low.y && local.x < high.x && local.y < high.y
        })
    }

    /// Takes `edits` in place of its own (decision 080): the resident tiles
    /// under any edit added or taken away, and two of their samples beyond,
    /// regenerate over the new stack, drawn as they were until then.
    fn edit(&mut self, edits: Arc<[TerrainEdit]>) {
        let mut reach: Option<(Vec2, Vec2)> = None;
        let changed = edits
            .iter()
            .filter(|edit| !self.terrain.edits.contains(edit))
            .chain(
                self.terrain
                    .edits
                    .iter()
                    .filter(|edit| !edits.contains(edit)),
            );
        for edit in changed {
            let center = Vec2::from(edit.stamp.center);
            let radius = Vec2::splat(edit.stamp.radius + edit.stamp.falloff);
            let (low, high) = (center - radius, center + radius);
            reach = Some(reach.map_or((low, high), |(min, max)| (min.min(low), max.max(high))));
        }
        self.terrain.edits = edits;
        self.edited = None;
        let Some((low, high)) = reach else {
            return;
        };
        let pyramid = self.terrain.pyramid;
        let under = |key: &TileKey| {
            let spacing = pyramid.spacing(u32::from(key.level));
            let size = TILE_QUADS as f32 * spacing;
            let first =
                Vec2::new(f32::from(key.x), f32::from(key.z)) * size - Vec2::splat(2.0 * spacing);
            let last = first + Vec2::splat(size + 4.0 * spacing);
            first.x <= high.x && first.y <= high.y && last.x >= low.x && last.y >= low.y
        };
        let stale: Vec<TileKey> = self
            .tiles
            .keys()
            .filter(|key| under(key))
            .copied()
            .collect();
        self.stale.extend(stale);
        self.jobs.retain(|key, _| !under(key));
        // Tiles kept in memory under the edits are no longer the ground
        // (decision 116).
        let gone: Vec<TileKey> = self
            .cache
            .keys()
            .filter(|key| under(key))
            .copied()
            .collect();
        for key in gone {
            self.uncache(key);
        }
        self.revision += 1;
    }

    /// The layers its tiles generate from (decision 080): `base`, its layers
    /// with their bakes, with its edits over them.
    fn generation(&mut self, base: Arc<Layers>) -> Arc<Layers> {
        if self.terrain.edits.is_empty() {
            return base;
        }
        if let Some((over, edited)) = &self.edited
            && Arc::ptr_eq(over, &base)
        {
            return edited.clone();
        }
        let stamps: Vec<Stamp> = self.terrain.edits.iter().map(|edit| edit.stamp).collect();
        match base.with_stamps(&stamps) {
            Ok(edited) => {
                let edited = Arc::new(edited);
                self.edited = Some((base, edited.clone()));
                edited
            }
            // Edits are validated as they are made.
            Err(_) => base,
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
                // Its shape draws meanwhile (decision 114).
                self.shape(tile);
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
            self.shaped.remove(&key);
            self.insert(tile);
        }
        true
    }

    /// Whether `key` still loads: not resident, resident under edits since
    /// (decision 080), or resident with its shape alone (decision 114).
    fn loads(&self, key: &TileKey) -> bool {
        !self.tiles.contains_key(key) || self.stale.contains(key) || self.shaped.contains(key)
    }

    /// Makes `shape`, a tile's heights and holes alone, resident until the
    /// whole tile arrives (decision 114), unless its key is resident or no
    /// coarser resident tile holds materials to draw it with; whether it
    /// did.
    fn shape(&mut self, shape: Tile) -> bool {
        let key = shape.key();
        if self.tiles.contains_key(&key) {
            return false;
        }
        let pyramid = self.terrain.pyramid;
        let mut ancestor = pyramid.parent(key);
        // The GPU gives shapes their materials (decision 118).
        if !self.materials {
            self.shaped.insert(key);
            self.insert(shape);
            return true;
        }
        loop {
            let Some(at) = ancestor else {
                return false;
            };
            if self
                .tiles
                .get(&at)
                .is_some_and(|tile| tile.control().is_some())
            {
                break;
            }
            ancestor = pyramid.parent(at);
        }
        self.shaped.insert(key);
        self.insert(shape);
        true
    }

    /// Makes `tile` resident, measuring how far it lies from its resident
    /// parent and its resident children from it.
    fn insert(&mut self, tile: Tile) {
        self.insert_shared(Arc::new(tile));
    }

    /// [`Self::insert`] for a tile already shared, as one kept in memory
    /// (decision 116).
    fn insert_shared(&mut self, tile: Arc<Tile>) {
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
        self.tiles.insert(key, tile);
        self.revision += 1;
    }

    /// Keeps `tile`, leaving residency at frame `frame`, in memory
    /// (decision 116) where it is whole: not its shape alone, not under
    /// edits since and not waiting for its objects standing far away.
    fn cache_leaving(&mut self, key: TileKey, tile: Arc<Tile>, frame: u64) {
        if !(self.shaped.contains(&key)
            || self.stale.contains(&key)
            || self.far_wanted.contains_key(&key)
            || self.far_jobs.contains_key(&key))
        {
            self.cache_tile(key, tile, frame);
        }
    }

    /// Keeps `tile` in memory (decision 116) as of frame `frame`.
    fn cache_tile(&mut self, key: TileKey, tile: Arc<Tile>, frame: u64) {
        let bytes = memory_bytes(&tile);
        self.uncache(key);
        self.cached_bytes += bytes;
        self.cache.insert(
            key,
            CachedTile {
                tile,
                bytes,
                left: frame,
            },
        );
    }

    /// Drops `key`'s tile from memory, if kept; it.
    fn uncache(&mut self, key: TileKey) -> Option<Arc<Tile>> {
        let cached = self.cache.remove(&key)?;
        self.cached_bytes -= cached.bytes;
        Some(cached.tile)
    }

    /// Makes the tiles of `missing` kept in memory resident again
    /// (decision 116), most important first, within `work`'s upload
    /// budget; how many.
    fn restore(&mut self, missing: &[(TileKey, u32)], work: &Work) -> u64 {
        let mut restored = 0;
        for &(key, _) in missing {
            if self.cache.is_empty() {
                break;
            }
            if self.tiles.contains_key(&key) || !self.cache.contains_key(&key) {
                continue;
            }
            if restored > 0 && !work.uploads() {
                break;
            }
            if let Some(tile) = self.uncache(key) {
                work.upload(2 * TILE_ARRAY_BYTES);
                self.insert_shared(tile);
                restored += 1;
            }
        }
        restored
    }

    /// The finest resident tile holding terrain-local `local`, and its
    /// sample there.
    fn sample(&self, local: Vec2) -> Option<(u32, TileSample)> {
        if self.cut_out(local) {
            return None;
        }
        let pyramid = &self.terrain.pyramid;
        (0..pyramid.levels()).find_map(|level| {
            let key = pyramid.tile_at(level, local)?;
            let tile = self.tiles.get(&key)?;
            let sample = tile.sample(local - pyramid.tile_origin(key))?;
            Some((level, sample))
        })
    }

    /// The meters tracks sink the ground at terrain-local `local` at most
    /// (decision 083): its nearest sample's materials' blend, from the
    /// finest resident tile.
    fn deform(&self, local: Vec2) -> Option<f32> {
        if self.cut_out(local) {
            return None;
        }
        let pyramid = &self.terrain.pyramid;
        (0..pyramid.levels()).find_map(|level| {
            let key = pyramid.tile_at(level, local)?;
            let tile = self.tiles.get(&key)?;
            let at = (local - pyramid.tile_origin(key)) / tile.spacing();
            let (x, z) = (at.x.round() as i32, at.y.round() as i32);
            // Tiles whose materials the GPU gives (decision 118): the rules'
            // at the sample.
            let control = match (tile.control_at(x, z), &self.surface) {
                (Some(control), _) => control,
                (None, Some((layers, _))) => {
                    let stored =
                        |sample: i32| (sample.clamp(-1, TILE_STRIDE as i32 - 2) + 1) as usize;
                    engine_terrain::gpu::sample_control(
                        tile,
                        (layers, &self.terrain.surface.rules),
                        (stored(x), stored(z)),
                    )
                }
                (None, None) => return None,
            };
            let materials = &self.terrain.surface.materials;
            let deform = |index: u8| {
                materials
                    .get(usize::from(index))
                    .map_or(0.0, |material| material.deform)
            };
            let share = f32::from(control.blend) / 255.0;
            Some(deform(control.a) + (deform(control.b) - deform(control.a)) * share)
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
    /// Milliseconds now, from the host (decision 099); without one, a
    /// frame works a slice.
    clock: Option<fn() -> f64>,
    /// The eye last streamed toward and its velocity (decision 100), the
    /// last frame's seconds, and the seconds since the eye last moved faster
    /// than the settings' motion detail speed (decision 115), `None` when
    /// it has not since they asked for it.
    last_eye: Option<Vec3>,
    velocity: Vec3,
    seconds: f32,
    fast_ago: Option<f32>,
    /// Whether the camera's view has loaded whole since terrains arrived.
    loaded: bool,
    /// Workers' share (decision 110): how many ran at the last frame, the
    /// stacks shared with them and keys handed out so far, and the jobs and
    /// recipes of terrains gone, to drop.
    workers: u32,
    stacks: Vec<(Arc<Layers>, String)>,
    shares: u64,
    retired: Retired,
    /// Tiles returned from memory since terrains started (decision 116);
    /// and bytes of the frame's upload budget left for work between
    /// frames, and spent there since the last frame.
    restored: u64,
    upload_left: u64,
    between: u64,
}

/// The workers' jobs and recipes of terrains gone (decision 110).
#[derive(Default)]
struct Retired {
    jobs: Vec<JobId>,
    recipes: Vec<String>,
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

    /// The meters tracks (decision 083) sink the ground under `(x, z)` at
    /// most, from its surface materials' [`SurfaceMaterial::deform`]; `None`
    /// where no tile with a surface is resident.
    pub fn deform_at(&self, x: f32, z: f32) -> Option<f32> {
        self.entries.iter().find_map(|state| {
            let origin = state.terrain.origin;
            state.deform(Vec2::new(x - origin.x, z - origin.z))
        })
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

    /// `entity`'s surroundings (decision 094) while they stream: they start
    /// once its bakes are done.
    pub fn surroundings(&self, entity: Entity) -> Option<SurroundingsStatus> {
        let state = self
            .entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_some())?;
        let around = state.around.as_ref()?;
        Some(SurroundingsStatus {
            reach: around.cutout.0.to_array(),
            tiles: state.terrain.pyramid.base_tiles()[0],
            spacing: state.terrain.pyramid.spacing(0),
            resident_tiles: state.tiles.len() as u32,
            patches: state.selection.patches.len() as u32,
        })
    }

    /// Follows each terrain's surroundings (decision 094): drawn, once its
    /// bakes are done, from the layers its tiles generate from, and dropped
    /// with it or when those or its settings for them change. They stream
    /// after every terrain, which take the frame's work first.
    fn sync_surroundings(&mut self) {
        let mut wanted = Vec::new();
        for state in &mut self.entries {
            if state.around.is_some() {
                continue;
            }
            let Some(surroundings) = state.terrain.surroundings.filter(Surroundings::is_usable)
            else {
                continue;
            };
            let TerrainSource::Generated(authored) = &state.terrain.source else {
                continue;
            };
            let layers = state.baked.clone().unwrap_or_else(|| authored.clone());
            if layers.unbaked().is_some() {
                continue;
            }
            let base = state.generation(layers);
            if let Some(layout) = surroundings.layout(&state.terrain) {
                let surface = state.terrain.surface.clone();
                wanted.push((state.entity, base, surroundings, layout, surface));
            }
        }
        let retired = &mut self.retired;
        self.entries.retain(|state| {
            let Some(around) = &state.around else {
                return true;
            };
            let kept = wanted
                .iter()
                .any(|(entity, base, surroundings, (_, _, cutout), _)| {
                    *entity == state.entity
                        && Arc::ptr_eq(base, &around.base)
                        && *surroundings == around.surroundings
                        && *cutout == around.cutout
                });
            if !kept {
                state.retire(retired);
            }
            kept
        });
        for (entity, base, surroundings, (origin, pyramid, cutout), surface) in wanted {
            if self
                .entries
                .iter()
                .any(|state| state.entity == entity && state.around.is_some())
            {
                continue;
            }
            // Their ground reads the terrain's layers where it lies in the
            // terrain's, whose own ground spans the cutout.
            let layers = base.surroundings(
                -cutout.0,
                (Vec2::ZERO, cutout.1 - cutout.0),
                surroundings.blend,
            );
            let terrain = Terrain {
                origin,
                pyramid,
                source: TerrainSource::Generated(Arc::new(layers)),
                surface: TerrainSurface {
                    rules: Arc::new(surface.rules.without_placements()),
                    materials: surface.materials,
                    assets: Vec::new(),
                    bridge: None,
                },
                collision: false,
                edits: Arc::from(Vec::new()),
                surroundings: None,
            };
            let mut state = TerrainState::new(entity, terrain);
            state.around = Some(Around {
                base,
                surroundings,
                cutout,
            });
            self.entries.push(state);
        }
    }

    /// The terrain of `entity`, while it streams.
    pub fn terrain(&self, entity: Entity) -> Option<&Terrain> {
        self.entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())
            .map(|state| &state.terrain)
    }

    /// The layers `entity`'s tiles generate from: its own, with their
    /// hydrology baked once it is (decision 066) and its edits over them
    /// (decision 080); `None` for terrains read from files.
    pub fn layers(&self, entity: Entity) -> Option<Arc<Layers>> {
        let state = self
            .entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())?;
        let base = match &state.terrain.source {
            TerrainSource::Generated(layers) => {
                state.baked.clone().unwrap_or_else(|| layers.clone())
            }
            TerrainSource::File { .. } => return None,
        };
        if state.terrain.edits.is_empty() {
            return Some(base);
        }
        if let Some((over, edited)) = &state.edited
            && Arc::ptr_eq(over, &base)
        {
            return Some(edited.clone());
        }
        let stamps: Vec<Stamp> = state.terrain.edits.iter().map(|edit| edit.stamp).collect();
        Some(base.with_stamps(&stamps).map_or(base, Arc::new))
    }

    /// The shortest walk over `entity`'s generated terrain from world
    /// `from` to `to` (x and z) as `options` walk (decision 081): around the
    /// placed objects that collide on its resident tiles, their footprints
    /// widened by the agent's radius, its points world and following the
    /// ground. `None` for terrains read from files and where no walk
    /// reaches. Its work grows with the area around the ends: for planning,
    /// not every frame.
    pub fn walk(
        &self,
        entity: Entity,
        from: Vec2,
        to: Vec2,
        options: &NavOptions,
    ) -> Option<NavPath> {
        let state = self
            .entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())?;
        let layers = self.layers(entity)?;
        let origin = state.terrain.origin;
        let corner = Vec2::new(origin.x, origin.z);
        let mut path = engine_terrain::walk(
            &layers,
            &state.terrain.pyramid,
            from - corner,
            to - corner,
            &Self::obstacles(state),
            options,
        )?;
        for point in &mut path.points {
            *point = [
                point[0] + origin.x,
                point[1] + origin.y,
                point[2] + origin.z,
            ];
        }
        Some(path)
    }

    /// A walk over the generated terrain under world `from` and `to` (x
    /// and z) as `options` walk, to search in budgeted steps (decision
    /// 089): the terrain, its origin, which the search's terrain-local
    /// points are from, and the search. `None` unless both ends lie on one
    /// generated terrain.
    pub fn walk_search(
        &self,
        from: Vec2,
        to: Vec2,
        options: &NavOptions,
    ) -> Option<(Entity, Vec3, engine_terrain::WalkSearch)> {
        let state = self.walked(from, to)?;
        let layers = self.layers(state.entity)?;
        let origin = state.terrain.origin;
        let corner = Vec2::new(origin.x, origin.z);
        let search = engine_terrain::WalkSearch::new(
            layers,
            &state.terrain.pyramid,
            (from - corner, to - corner),
            Self::obstacles(state),
            options,
        );
        Some((state.entity, origin, search))
    }

    /// The input of a walk job (decision 110) over the generated terrain
    /// under world `from` and `to` (x and z) as `options` walk, once workers
    /// hold its recipe for the layers its tiles generate from now: the
    /// terrain, its origin, which the walk's terrain-local points are from,
    /// and the input. `None` otherwise, as for [`Self::walk_search`].
    pub fn walk_job(
        &self,
        from: Vec2,
        to: Vec2,
        options: &NavOptions,
    ) -> Option<(Entity, Vec3, Vec<u8>)> {
        let state = self.walked(from, to)?;
        let (entity, origin) = (state.entity, state.terrain.origin);
        let recipe = state.recipe.as_ref()?;
        let layers = self.layers(entity)?;
        if !Arc::ptr_eq(&recipe.stack, &layers) {
            return None;
        }
        let corner = Vec2::new(origin.x, origin.z);
        let input = engine_terrain::jobs::walk_input(
            &recipe.key,
            (from - corner, to - corner),
            &Self::obstacles(state),
            options,
        );
        Some((entity, origin, input))
    }

    /// The terrain both world `from` and `to` (x and z) lie on, which walks
    /// between them cross: never surroundings (decision 094), which no one
    /// walks.
    fn walked(&self, from: Vec2, to: Vec2) -> Option<&TerrainState> {
        self.entries.iter().find(|state| {
            if state.around.is_some() {
                return false;
            }
            let pyramid = &state.terrain.pyramid;
            let [x, z] = pyramid.tiles(0);
            let size = Vec2::new(x as f32, z as f32)
                * engine_terrain::TILE_QUADS as f32
                * pyramid.spacing(0);
            let corner = Vec2::new(state.terrain.origin.x, state.terrain.origin.z);
            [from, to].iter().all(|&point| {
                let local = point - corner;
                local.cmpge(Vec2::ZERO).all() && local.cmple(size).all()
            })
        })
    }

    /// The footprints of the placed objects that collide on `state`'s
    /// resident tiles.
    fn obstacles(state: &TerrainState) -> Vec<Obstacle> {
        let assets = &state.terrain.surface.assets;
        state
            .tiles
            .values()
            .flat_map(|tile| tile.placed().iter())
            .filter_map(|placed| {
                let collider = assets.get(usize::from(placed.asset))?.collider?;
                let reach = match collider.shape {
                    PlacedShape::Sphere { radius } => radius,
                    PlacedShape::Cuboid { half_extents } => {
                        crate::math::max(half_extents[0], half_extents[2])
                    }
                };
                Some(Obstacle {
                    center: Vec2::new(placed.position[0], placed.position[2]),
                    radius: reach * placed.scale,
                })
            })
            .collect()
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
        let state = self
            .entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())?;
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
            .find(|state| state.entity == entity && state.around.is_none())
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
            .find(|state| state.entity == entity && state.around.is_none())
            .map(|state| (state.errors.as_slice(), state.ranges.as_slice()))
    }

    /// Resident tiles of `entity`'s terrain.
    pub fn tiles(&self, entity: Entity) -> impl Iterator<Item = &Arc<Tile>> {
        self.entries
            .iter()
            .filter(move |state| state.entity == entity && state.around.is_none())
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
    /// The bridges of `entity`'s roads (decision 072), terrain-local, once
    /// its bakes are done.
    pub fn bridges(&self, entity: Entity) -> &[Bridge] {
        self.entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())
            .map_or(&[], |state| &state.bridges)
    }

    /// Every terrain's bridges, its origin and its bridge material.
    pub(crate) fn bridge_sets(&self) -> impl Iterator<Item = crate::bridges::BridgeSet> + '_ {
        self.entries.iter().map(|state| {
            (
                state.entity,
                state.bridges.clone(),
                state.terrain.origin,
                state.terrain.surface.bridge,
            )
        })
    }

    /// The waterfalls of `entity`'s rivers (decision 075), terrain-local,
    /// once its bakes are done.
    pub fn falls(&self, entity: Entity) -> &[Fall] {
        self.entries
            .iter()
            .find(|state| state.entity == entity && state.around.is_none())
            .map_or(&[], |state| &state.falls)
    }

    /// Every terrain's waterfalls and its origin.
    pub(crate) fn fall_sets(&self) -> impl Iterator<Item = (Entity, Arc<[Fall]>, Vec3)> + '_ {
        self.entries
            .iter()
            .map(|state| (state.entity, state.falls.clone(), state.terrain.origin))
    }

    /// Colliding terrains' bridges, with their origins (decision 072).
    pub fn colliding_bridges(&self) -> impl Iterator<Item = (Entity, Vec3, &[Bridge])> {
        self.entries
            .iter()
            .filter(|state| state.terrain.collision && !state.bridges.is_empty())
            .map(|state| (state.entity, state.terrain.origin, &*state.bridges))
    }

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

    /// The clock frames time their terrain work by (decision 099):
    /// milliseconds, from any origin. Hosts without one work a slice of
    /// `samples_per_frame` a frame.
    pub fn set_clock(&mut self, clock: fn() -> f64) {
        self.clock = Some(clock);
    }

    /// Whether the camera's view has yet to load whole (decision 099);
    /// never without terrains.
    pub fn loading(&self) -> bool {
        !self.loaded && self.entries.iter().any(|state| state.around.is_none())
    }

    /// Works on terrain for the settings' `loading_work_ms` a frame again
    /// until the camera's view loads whole, as after the camera jumps far.
    pub fn begin_loading(&mut self) {
        self.loaded = false;
    }

    /// Whether the last frame found tiles to load: hosts then load them in
    /// time frames leave free (decision 108).
    pub fn streaming(&self) -> bool {
        self.entries.iter().any(|state| !state.missing.is_empty())
    }

    /// Loads, for `milliseconds`, the tiles the last frame found missing,
    /// most important first, between frames (decision 108): frames leave
    /// time free while the GPU draws them, which terrain otherwise only
    /// takes a few milliseconds of. Patches are selected anew with the next
    /// frame. Returns whether tiles are still to load; nothing without a
    /// clock (`set_clock`).
    pub fn work_between_frames(
        &mut self,
        settings: &TerrainSettings,
        milliseconds: f64,
        jobs: Option<&mut Jobs>,
    ) -> bool {
        let Some(clock) = self.clock else {
            return false;
        };
        let mut slots = jobs.map(JobSlots::of);
        // Smaller slices than frames take, so work ends close to its time.
        let work = Work {
            slice: u64::from(settings.samples_per_frame / 4).max(1),
            clock: Some(clock),
            end: clock() + milliseconds,
            upload: std::cell::Cell::new(self.upload_left),
            bands: settings.max_bands as usize,
        };
        let mut budget = work.slice;
        let frame = self.frame;
        let requests = &mut self.requests;
        let mut remaining = false;
        for state in &mut self.entries {
            if !state.terrain.is_usable() || !settings.enabled {
                continue;
            }
            let mut missing: Vec<(TileKey, u32)> = std::mem::take(&mut state.missing)
                .into_iter()
                .filter(|(key, _)| state.loads(key))
                .collect();
            // Tiles kept in memory return at once (decision 116).
            if !state.cache.is_empty() {
                self.restored += state.restore(&missing, &work);
                missing.retain(|(key, _)| state.loads(key));
            }
            if missing.is_empty() && state.far_wanted.is_empty() && state.far_jobs.is_empty() {
                continue;
            }
            let target = std::mem::take(&mut state.target);
            let keep = std::mem::take(&mut state.keep);
            let baking = load_tiles(
                state,
                &missing,
                (&target, &keep),
                &work,
                &mut budget,
                frame,
                requests,
                slots.as_mut(),
            );
            state.missing = missing
                .into_iter()
                .filter(|(key, _)| state.loads(key))
                .collect();
            state.target = target;
            state.keep = keep;
            // Tiles waiting on their fetch load once delivered.
            let waiting = matches!(state.terrain.source, TerrainSource::File { data: None, .. })
                && state.arrivals.is_empty();
            remaining |= baking
                || (!state.missing.is_empty() && !waiting)
                || !state.dispatched.is_empty()
                || !state.far_wanted.is_empty()
                || !state.far_jobs.is_empty();
            if budget == 0 && !work.refill(&mut budget) {
                break;
            }
        }
        // What arrived uploads with the next frame (decision 116).
        self.between += self.upload_left - work.upload.get();
        self.upload_left = work.upload.get();
        remaining
    }

    /// Where `eye`, moved since the last frame `seconds` ago, is headed
    /// (decision 100): its velocity, smoothed over frames, carried
    /// `AHEAD_SECONDS` on, at most `AHEAD_MAX` meters; the eye itself after
    /// a jump.
    fn heading(&mut self, eye: Vec3, seconds: f32) -> Vec3 {
        let moved = self.last_eye.map_or(Vec3::ZERO, |last| eye - last);
        self.last_eye = Some(eye);
        self.seconds = if seconds.is_finite() {
            crate::math::max(seconds, 0.0)
        } else {
            0.0
        };
        if moved.length() > JUMP {
            // A jump loads the view again at the loading budget (decision
            // 108), as the first view loads, unless workers load it
            // (decision 110).
            self.velocity = Vec3::ZERO;
            if self.workers == 0 {
                self.loaded = false;
            }
        } else if seconds > 0.0 {
            self.velocity = self.velocity.lerp(moved / seconds, 0.25);
        }
        if !self.velocity.is_finite() {
            self.velocity = Vec3::ZERO;
        }
        eye + (self.velocity * AHEAD_SECONDS).clamp_length_max(AHEAD_MAX)
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
            .find(|state| state.entity == terrain && state.around.is_none())
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
        (eye, ahead): (Vec3, Vec3),
        viewport: Viewport,
        focus: &[Vec3],
        jobs: Option<&mut Jobs>,
    ) {
        self.frame += 1;
        self.sync_surroundings();
        // Jobs of terrains gone are dropped, and so are their recipes
        // (decision 110).
        let mut slots = jobs.map(JobSlots::of);
        let retired = std::mem::take(&mut self.retired);
        self.workers = slots.as_ref().map_or(0, |slots| slots.jobs.workers());
        let workers = self.workers;
        // Keys no recipe may hold any longer, unshared at the end unless
        // one does (decision 116).
        let mut released = Vec::new();
        if let Some(slots) = slots.as_mut() {
            for id in retired.jobs {
                slots.jobs.cancel(id);
            }
            released.extend(retired.recipes);
        }
        let mut stats = TerrainStats {
            terrains: self
                .entries
                .iter()
                .filter(|state| state.around.is_none())
                .count() as u32,
            bridges: self
                .entries
                .iter()
                .flat_map(|state| state.bridges.iter())
                .filter(|bridge| !bridge.tunnel)
                .count() as u32,
            tunnels: self
                .entries
                .iter()
                .flat_map(|state| state.bridges.iter())
                .filter(|bridge| bridge.tunnel)
                .count() as u32,
            falls: self
                .entries
                .iter()
                .map(|state| state.falls.len() as u32)
                .sum(),
            ..TerrainStats::default()
        };
        let forward = camera.target - camera.eye();
        let forward = Vec2::new(forward.x, forward.z);
        let planes = ViewPlanes::new(camera, eye, viewport);
        let fov = camera.vertical_fov_radians;
        let fov = if fov.is_finite() && fov > 0.0 && fov < std::f32::consts::PI {
            fov
        } else {
            Camera::default().vertical_fov_radians
        };
        // The frame's work (decision 099): slices until its time is up.
        let work = Work {
            slice: u64::from(settings.samples_per_frame),
            clock: self.clock,
            end: self.clock.map_or(0.0, |clock| {
                clock()
                    + if self.loaded {
                        f64::from(settings.work_ms)
                    } else {
                        f64::from(settings.loading_work_ms)
                    }
            }),
            // Tiles arriving between frames upload with this frame's.
            upload: std::cell::Cell::new(
                u64::from(settings.upload_bytes).saturating_sub(std::mem::take(&mut self.between)),
            ),
            bands: settings.max_bands as usize,
        };
        let mut budget = work.slice;
        // Speed-aware detail (decision 115): the finest levels waiting at
        // the eye's speed, and whether it has yet to settle.
        let (waiting, unsettled) = match settings.motion_detail {
            Some(detail) => {
                let speed = self.velocity.length();
                self.fast_ago = if speed > detail.speed {
                    Some(0.0)
                } else {
                    self.fast_ago.map(|ago| ago + self.seconds)
                };
                (
                    detail.waiting(speed),
                    self.fast_ago.is_some_and(|ago| ago < detail.settle),
                )
            }
            None => {
                self.fast_ago = None;
                (0, false)
            }
        };
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
            // Surroundings (decision 094) leave the ground they surround to
            // their terrain, and stream while settings show them.
            let cutout = state.around.as_ref().map(|around| around.cutout);
            if cutout.is_some() && !settings.surroundings {
                state.selection.patches.clear();
                state.selection.fallbacks = 0;
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
            // Collision's tiles load within its radius of the camera and
            // focus entities and stay, colliding, a quarter farther, as the
            // camera's tiles do: focus moving to and fro across a tile's
            // edge must not regenerate it each time.
            state.collision.clear();
            let mut loading = Vec::new();
            let mut under = Vec::new();
            if state.terrain.collision && settings.collision_radius > 0.0 {
                let held = settings.collision_radius * KEEP_REACH / LOAD_REACH;
                for point in focus {
                    let local = *point - origin;
                    collision_keys(&pyramid, local, settings.collision_radius, &mut loading);
                    collision_keys(&pyramid, local, held, &mut state.collision);
                    under.extend(pyramid.tile_at(0, Vec2::new(local.x, local.z)));
                }
                loading.sort_unstable();
                loading.dedup();
                state.collision.sort_unstable();
                state.collision.dedup();
            }
            // What the view wants (decision 112), the camera's tiles within
            // their reach of the eye and where it is headed (decision 100),
            // and what it keeps, within a wider reach; distances run to each
            // tile's ground.
            // Ahead of need (decision 116), while nothing was missing, nor
            // generating, at the last frame and memory has room.
            let idle = workers > 0
                && settings.idle_reach > KEEP_REACH
                && state.cached_bytes < settings.cache_bytes
                && state.missing.is_empty()
                && state.dispatched.is_empty()
                && state.banded.is_empty()
                && state.placements.is_empty()
                && state.far_wanted.is_empty()
                && state.far_jobs.is_empty();
            let (wanted, headed, held, around) = {
                let tiles = &state.tiles;
                let errors = &state.errors;
                let ground = |key: TileKey| ground_bounds(tiles, &pyramid, errors, key);
                let seen = |low: Vec3, high: Vec3| planes.sees(low + origin, high + origin);
                let view = |eye: Vec3| WantedView {
                    eye,
                    forward,
                    ground: &ground,
                    seen: &seen,
                };
                let want = |eye: Vec3, reach: f32| {
                    let mut out = Vec::new();
                    engine_terrain::wanted_tiles(
                        &pyramid,
                        &ranges,
                        &view(eye),
                        reach,
                        settings.pinned_levels,
                        cutout,
                        &mut out,
                    );
                    out
                };
                (
                    want(local_eye, LOAD_REACH),
                    if ahead == eye {
                        Vec::new()
                    } else {
                        want(ahead - origin, LOAD_REACH)
                    },
                    want(local_eye, KEEP_REACH),
                    if idle {
                        want(local_eye, settings.idle_reach)
                    } else {
                        Vec::new()
                    },
                )
            };
            // What to load, most important first, each tile after its
            // ancestors (decision 112): the pinned levels; the tiles the
            // view draws and shows, then those under focus entities, which
            // jobs take up first; the tiles it draws out of sight, then
            // collision's; those only shading what it shows; where the eye
            // is headed; the rest.
            let tier = |pick: fn(&WantedTile) -> bool| {
                wanted
                    .iter()
                    .filter(move |tile| pick(tile))
                    .map(|tile| tile.key)
            };
            let mut order: Vec<TileKey> = Vec::with_capacity(wanted.len() + loading.len());
            let mut pushed: HashSet<TileKey> = HashSet::with_capacity(order.capacity());
            for key in tier(|tile| tile.score < 0.0)
                .chain(tier(|tile| tile.score >= 0.0 && tile.drawn && tile.seen))
                .chain(under.iter().copied())
            {
                push_with_ancestors(&pyramid, key, &mut order, &mut pushed);
            }
            let urgent = order.len();
            for key in tier(|tile| tile.score >= 0.0 && tile.drawn && !tile.seen)
                .chain(loading.iter().copied())
                .chain(tier(|tile| tile.score >= 0.0 && !tile.drawn && tile.seen))
                .chain(headed.iter().map(|tile| tile.key))
                .chain(tier(|tile| tile.score >= 0.0 && !tile.drawn && !tile.seen))
            {
                push_with_ancestors(&pyramid, key, &mut order, &mut pushed);
            }
            // Over a tile budget, the pinned levels and collision's tiles
            // stay first, then the rest in order.
            let target: HashSet<TileKey> = if order.len() <= remaining {
                order.iter().copied().collect()
            } else {
                let mut first = Vec::new();
                let mut held = HashSet::new();
                for key in tier(|tile| tile.score < 0.0).chain(loading.iter().copied()) {
                    push_with_ancestors(&pyramid, key, &mut first, &mut held);
                }
                first
                    .iter()
                    .copied()
                    .chain(order.iter().copied().filter(|key| !held.contains(key)))
                    .take(remaining)
                    .collect()
            };
            let kept = target.len();
            stats.dropped_tiles += (order.len() - kept) as u32;
            remaining -= kept;
            // What to keep: everything within a wider reach, unless over
            // budget.
            let keep: HashSet<TileKey> = held
                .iter()
                .map(|tile| tile.key)
                .chain(state.collision.iter().copied())
                .chain(target.iter().copied())
                .collect();
            let before = state.tiles.len();
            // Tiles leaving stay in memory (decision 116).
            let leaving: Vec<TileKey> = state
                .tiles
                .keys()
                .filter(|key| !keep.contains(key))
                .copied()
                .collect();
            for key in leaving {
                if let Some(tile) = state.tiles.remove(&key) {
                    state.cache_leaving(key, tile, frame);
                }
            }
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
                    if let Some(tile) = state.tiles.remove(&key) {
                        state.cache_leaving(key, tile, frame);
                    }
                }
            }
            if state.tiles.len() != before {
                state.revision += 1;
            }
            state.jobs.retain(|key, _| target.contains(key));
            // Load what is missing, most important first; disabled terrains
            // load nothing.
            // Tiles under edits since they loaded (decision 080) regenerate
            // in their turn and draw until they do.
            state.stale.retain(|key| state.tiles.contains_key(key));
            state.shaped.retain(|key| state.tiles.contains_key(key));
            // While the eye moves fast (decision 115), the finest levels'
            // tiles wait, and until it settles those it does not show; the
            // pinned levels, collision's tiles and those under focus
            // entities, with their ancestors, never wait.
            let mut never = HashSet::new();
            let mut shown = HashSet::new();
            if waiting > 0 || unsettled {
                let mut chain = Vec::new();
                for key in tier(|tile| tile.score < 0.0)
                    .chain(loading.iter().copied())
                    .chain(under.iter().copied())
                {
                    push_with_ancestors(&pyramid, key, &mut chain, &mut never);
                }
                if unsettled {
                    chain.clear();
                    for key in tier(|tile| tile.seen) {
                        push_with_ancestors(&pyramid, key, &mut chain, &mut shown);
                    }
                }
            }
            let waits = |key: &TileKey| {
                (waiting > 0 || unsettled)
                    && !never.contains(key)
                    && (u32::from(key.level) < waiting || (unsettled && !shown.contains(key)))
            };
            stats.waiting_tiles += order
                .iter()
                .filter(|key| target.contains(key) && state.loads(key) && waits(key))
                .count() as u32;
            let mut missing: Vec<(TileKey, u32)> = order
                .iter()
                .enumerate()
                .filter(|(_, key)| {
                    settings.enabled && target.contains(key) && state.loads(key) && !waits(key)
                })
                .map(|(index, &key)| {
                    let class = if index < urgent {
                        URGENT_JOBS
                    } else {
                        VIEW_JOBS
                    };
                    (key, job_priority(class, index as u32))
                })
                .collect();
            // Tiles kept in memory return at once (decision 116).
            if !state.cache.is_empty() {
                self.restored += state.restore(&missing, &work);
                missing.retain(|(key, _)| state.loads(key));
            }
            // What workers generate ahead of need: the nearest tiles within
            // the idle reach not resident, kept or kept in memory.
            state.ahead.clear();
            state
                .ahead
                .extend(around.iter().map(|tile| tile.key).filter(|key| {
                    !keep.contains(key)
                        && !state.tiles.contains_key(key)
                        && !state.cache.contains_key(key)
                }));
            // What the frame found missing loads between frames too
            // (decision 108).
            state.missing.clone_from(&missing);
            state.target.clone_from(&target);
            state.keep.clone_from(&keep);
            // GPU materials (decision 118) before the recipe that carries
            // them.
            state.update_materials(settings.gpu_materials);
            // Workers generate from its recipe (decision 110).
            if let Some(slots) = slots.as_mut()
                && slots.jobs.workers() > 0
            {
                share_recipe(
                    state,
                    slots.jobs,
                    (&mut self.stacks, &mut self.shares),
                    &mut released,
                );
            }
            if load_tiles(
                state,
                &missing,
                (&target, &keep),
                &work,
                &mut budget,
                frame,
                &mut self.requests,
                slots.as_mut(),
            ) {
                stats.baking += 1;
                continue;
            }
            let missing = std::mem::take(&mut state.missing);
            state.missing = missing
                .into_iter()
                .filter(|(key, _)| state.loads(key))
                .collect();
            stats.resident_tiles += state.tiles.len() as u32;
            stats.missing_tiles += state.missing.len() as u32;
            stats.shaped_tiles += state.shaped.len() as u32;
            stats.cached_tiles += state.cache.len() as u32;
            stats.cached_bytes += state.cached_bytes;
            stats.idle_tiles += state.idle.len() as u32;
            stats.loading_tiles += (state.jobs.len()
                + state.in_flight.len()
                + state.arrivals.len()
                + state.dispatched.len()
                + state.far_wanted.len()
                + state.far_jobs.len()
                + state.banded.len()
                + state.placements.len()) as u32;
            stats.banded_tiles += state.banded_tiles;
            if settings.enabled {
                let tiles = &state.tiles;
                engine_terrain::select_patches(
                    &pyramid,
                    &ranges,
                    local_eye,
                    |key| tiles.get(&key).map(Arc::as_ref),
                    cutout,
                    &mut state.selection,
                );
            } else {
                state.selection.patches.clear();
                state.selection.fallbacks = 0;
            }
            stats.patches += state.selection.patches.len() as u32;
            stats.fallback_patches += state.selection.fallbacks;
            if cutout.is_some() {
                stats.surroundings_tiles += state.tiles.len() as u32;
                stats.surroundings_patches += state.selection.patches.len() as u32;
            }
        }
        // Tiles kept in memory beyond its budget go, those that left first
        // (decision 116).
        let cached: u64 = self.entries.iter().map(|state| state.cached_bytes).sum();
        if cached > settings.cache_bytes {
            let mut oldest: Vec<(u64, usize, TileKey)> = self
                .entries
                .iter()
                .enumerate()
                .flat_map(|(index, state)| {
                    state
                        .cache
                        .iter()
                        .map(move |(key, kept)| (kept.left, index, *key))
                })
                .collect();
            oldest.sort_unstable();
            let mut over = cached - settings.cache_bytes;
            for (_, index, key) in oldest {
                if over == 0 {
                    break;
                }
                let state = &mut self.entries[index];
                let before = state.cached_bytes;
                state.uncache(key);
                over = over.saturating_sub(before - state.cached_bytes);
            }
        }
        stats.restored_tiles = self.restored;
        stats.cached_bytes = self.entries.iter().map(|state| state.cached_bytes).sum();
        stats.cached_tiles = self
            .entries
            .iter()
            .map(|state| state.cache.len() as u32)
            .sum();
        self.upload_left = work.upload.get();
        // Stacks no terrain's recipe shares any longer leave the workers
        // (decision 110); keys named by content (decision 116) leave once
        // no recipe or stack holds them.
        if let Some(slots) = slots.as_mut() {
            let entries = &self.entries;
            self.stacks.retain(|(stack, key)| {
                let held = entries.iter().any(|state| {
                    state
                        .recipe
                        .as_ref()
                        .is_some_and(|recipe| Arc::ptr_eq(&recipe.stack, stack))
                });
                if !held {
                    released.push(key.clone());
                }
                held
            });
            for key in released {
                let held = self.stacks.iter().any(|(_, shared)| *shared == key)
                    || self.entries.iter().any(|state| {
                        state
                            .recipe
                            .as_ref()
                            .is_some_and(|recipe| recipe.key == key)
                    });
                if !held {
                    slots.jobs.unshare(key);
                }
            }
        }
        // The camera's view has loaded whole once nothing bakes, loads or
        // stands in for a finer level (decision 099).
        if !self.loaded
            && stats.terrains > 0
            && stats.baking == 0
            && stats.loading_tiles == 0
            && stats.fallback_patches == 0
        {
            self.loaded = true;
        }
        stats.loading = self.loading();
        self.stats = stats;
    }
}

/// The lowest and highest ground tile `key` can hold (decision 112),
/// terrain-local, as distances to it take them: a resident tile's own,
/// widened by how far its level measured from the next coarser (its
/// patches morph toward that); else its nearest resident ancestor's over
/// its ground, widened by how far the levels between measured apart and by
/// four of the ancestor's spacings, as finer levels add detail. `None`
/// without either.
fn ground_bounds(
    tiles: &HashMap<TileKey, Arc<Tile>>,
    pyramid: &Pyramid,
    errors: &[f32],
    key: TileKey,
) -> Option<[f32; 2]> {
    let error = |level: u8| errors.get(usize::from(level)).copied().unwrap_or(0.0);
    if let Some(tile) = tiles.get(&key) {
        let [low, high] = tile.bounds();
        return Some([low - error(key.level), high + error(key.level)]);
    }
    let size = pyramid.tile_size(u32::from(key.level));
    let low = Vec2::new(f32::from(key.x), f32::from(key.z)) * size;
    let high = low + size;
    let mut ancestor = key;
    let mut margin = 0.0;
    while let Some(parent) = pyramid.parent(ancestor) {
        margin += error(ancestor.level);
        ancestor = parent;
        let Some(tile) = tiles.get(&ancestor) else {
            continue;
        };
        let level = u32::from(ancestor.level);
        let spacing = pyramid.spacing(level);
        let patch = PATCH_QUADS as f32 * spacing;
        let corner =
            Vec2::new(f32::from(ancestor.x), f32::from(ancestor.z)) * pyramid.tile_size(level);
        let last = (PATCHES_PER_TILE - 1) as f32;
        let first = ((low - corner) / patch)
            .floor()
            .clamp(Vec2::ZERO, Vec2::splat(last));
        let end = ((high - corner) / patch - 1.0)
            .ceil()
            .clamp(Vec2::ZERO, Vec2::splat(last));
        let (mut bottom, mut top) = (f32::INFINITY, f32::NEG_INFINITY);
        for z in first.y as u32..=end.y as u32 {
            for x in first.x as u32..=end.x as u32 {
                let [patch_low, patch_high] = tile.patch_bounds(x, z);
                bottom = crate::math::min(bottom, patch_low);
                top = crate::math::max(top, patch_high);
            }
        }
        let margin = margin + 4.0 * spacing;
        return Some([bottom - margin, top + margin]);
    }
    None
}

/// The view's side planes and near plane, inward, in world space (decision
/// 112): which boxes show in it, however far away.
struct ViewPlanes(Option<[Vec4; 5]>);

impl ViewPlanes {
    /// `camera`'s view from `eye` (its own, unless frozen) on `viewport`.
    fn new(camera: &Camera, eye: Vec3, viewport: Viewport) -> Self {
        let aspect = viewport.width as f32 / viewport.height.max(1) as f32;
        let from = Camera {
            position: eye,
            target: eye + (camera.target - camera.eye()),
            ..*camera
        };
        let rows = from.view_projection(aspect).transpose();
        let planes = [
            rows.w_axis + rows.x_axis,
            rows.w_axis - rows.x_axis,
            rows.w_axis + rows.y_axis,
            rows.w_axis - rows.y_axis,
            // Reversed depth (decision 042): the near plane is depth 1.
            rows.w_axis - rows.z_axis,
        ];
        Self(
            planes
                .iter()
                .all(|plane| plane.is_finite())
                .then_some(planes),
        )
    }

    /// Whether any of the box from `low` to `high` lies inside every plane;
    /// always without planes.
    fn sees(&self, low: Vec3, high: Vec3) -> bool {
        let Some(planes) = &self.0 else {
            return true;
        };
        planes.iter().all(|plane| {
            let corner = Vec3::new(
                if plane.x >= 0.0 { high.x } else { low.x },
                if plane.y >= 0.0 { high.y } else { low.y },
                if plane.z >= 0.0 { high.z } else { low.z },
            );
            plane.truncate().dot(corner) + plane.w >= 0.0
        })
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
    /// How each placement layer's objects thin with distance (decision
    /// 100), by the layer indices its tiles' placed objects hold: the level
    /// of the tiles holding them, the level they also stand far away at and
    /// the share standing there (their own level and 0 without).
    pub thinning: Vec<(u8, u8, f32)>,
    /// Its resident tiles: renderers upload those they lack and release
    /// the rest. The list changes only when `tiles_revision` advances.
    pub tiles: Arc<[Arc<Tile>]>,
    pub tiles_revision: u64,
    /// The patches the camera draws.
    pub patches: Vec<Patch>,
    /// Meters from the eye within which each level's tiles load, finest
    /// first: where renderers can shade with them.
    pub reach: Vec<f32>,
    /// Meters from the eye up to which each level draws, finest first; each
    /// but the coarsest morphs into the next over its last quarter
    /// ([`LodRanges`]): where renderers find the ground as drawn.
    pub ranges: Vec<f32>,
    /// Where its hydrology bake found lakes standing (decision 067), and
    /// its rivers (decision 092).
    pub water: Option<Arc<engine_terrain::WaterGrid>>,
    pub rivers: Option<Arc<engine_terrain::Rivers>>,
    /// Whether it is a terrain's surroundings (decision 094), which grow
    /// no blades: their grassy ground shows as canopy at every distance.
    pub surroundings: bool,
    /// Whether its tiles may hold holes and whether a hydrology bake will
    /// give it water: renderers create what draws them before it shows,
    /// while it loads (decision 099).
    pub may_hole: bool,
    pub hydrology: bool,
    /// Its surface for the GPU to give its tiles control values (decision
    /// 118), which they then hold none of; `None` where they hold them.
    pub gpu_surface: Option<Arc<engine_terrain::gpu::SurfaceData>>,
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

    /// Loads tiles the last frame found missing for `milliseconds` between
    /// frames (decision 108); whether tiles are still to load.
    pub(crate) fn work(&mut self, world: &mut World, milliseconds: f64) -> bool {
        let Some(settings) = world
            .get_resource::<TerrainSettings>()
            .copied()
            .map(TerrainSettings::sanitized)
        else {
            return false;
        };
        let mut jobs = world.remove_resource::<Jobs>();
        let remaining = world
            .get_resource_mut::<Terrains>()
            .is_some_and(|mut terrains| {
                terrains.work_between_frames(&settings, milliseconds, jobs.as_mut())
            });
        if let Some(jobs) = jobs {
            world.insert_resource(jobs);
        }
        remaining
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
        let seconds = world
            .get_resource::<crate::FrameTime>()
            .map_or(0.0, |time| time.delta_seconds);
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
        // Workers' jobs (decision 110) stream with the terrains.
        let mut jobs = world.remove_resource::<Jobs>();
        let Some(mut terrains) = world.get_resource_mut::<Terrains>() else {
            if let Some(jobs) = jobs {
                world.insert_resource(jobs);
            }
            return;
        };
        let terrains = terrains.as_mut();
        let retired = &mut terrains.retired;
        terrains.entries.retain(|state| {
            let kept = present.iter().any(|(entity, _)| *entity == state.entity);
            if !kept {
                state.retire(retired);
            }
            kept
        });
        for (entity, terrain) in present {
            match terrains
                .entries
                .iter_mut()
                .find(|state| state.entity == entity && state.around.is_none())
            {
                Some(state) => {
                    if !terrain.same(&state.terrain) {
                        state.retire(&mut terrains.retired);
                        *state = TerrainState::new(entity, terrain);
                        terrains.loaded = false;
                    } else {
                        if !Arc::ptr_eq(&terrain.edits, &state.terrain.edits) {
                            state.edit(terrain.edits.clone());
                        }
                        // Surroundings (decision 094) follow on their own.
                        state.terrain.surroundings = terrain.surroundings;
                    }
                }
                None => {
                    terrains.entries.push(TerrainState::new(entity, terrain));
                    terrains.loaded = false;
                }
            }
        }
        let Some(settings) = settings else {
            for state in &mut terrains.entries {
                state.selection.patches.clear();
            }
            terrains.stats = TerrainStats {
                terrains: terrains
                    .entries
                    .iter()
                    .filter(|state| state.around.is_none())
                    .count() as u32,
                ..TerrainStats::default()
            };
            // Terrains that do not stream have nothing to load.
            terrains.loaded = true;
            if let Some(jobs) = jobs {
                world.insert_resource(jobs);
            }
            return;
        };
        // Where the eye is headed (decision 100), from its motion since the
        // last frame.
        let ahead = terrains.heading(eye, seconds);
        terrains.stream(
            &settings,
            &camera,
            (eye, ahead),
            viewport,
            &self.points,
            jobs.as_mut(),
        );
        if let Some(jobs) = jobs {
            world.insert_resource(jobs);
        }
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
                    thinning: Vec::new(),
                    tiles: Arc::from(Vec::new()),
                    tiles_revision: 0,
                    patches: Vec::new(),
                    reach: Vec::new(),
                    ranges: Vec::new(),
                    water: None,
                    rivers: None,
                    surroundings: false,
                    may_hole: false,
                    hydrology: false,
                    gpu_surface: None,
                });
            }
            let render = &mut listing.terrains[index];
            render.id = state.render_id();
            render.surroundings = state.around.is_some();
            (render.may_hole, render.hydrology) = match &state.terrain.source {
                TerrainSource::Generated(layers) => (layers.may_hole(), layers.has_hydrology()),
                TerrainSource::File { .. } => (false, false),
            };
            render.origin = state.terrain.origin;
            render.pyramid = state.terrain.pyramid;
            render.surface.clone_from(&state.terrain.surface.materials);
            render.assets.clear();
            for asset in &state.terrain.surface.assets {
                render.assets.push(
                    assets.and_then(|assets| RenderPlacementAsset::of(asset, assets, missing)),
                );
            }
            render.thinning.clear();
            render.thinning.extend(
                state
                    .terrain
                    .surface
                    .rules
                    .authored()
                    .placements
                    .iter()
                    .map(|layer| match layer.far {
                        Some(far) => (layer.level, far.level, far.share),
                        None => (layer.level, layer.level, 0.0),
                    }),
            );
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
            if render.tiles_revision != state.revision || render.id != state.render_id() {
                let mut tiles: Vec<Arc<Tile>> = state.tiles.values().cloned().collect();
                tiles.sort_unstable_by_key(|tile| tile.key());
                render.tiles = Arc::from(tiles);
                render.tiles_revision = state.revision;
            }
            render.patches.clear();
            if enabled {
                render.patches.extend_from_slice(&state.selection.patches);
            }
            let hydrology = state.baked.as_ref().and_then(|layers| layers.hydrology());
            render.water = hydrology.map(|hydrology| Arc::clone(&hydrology.water));
            render.rivers = hydrology.map(|hydrology| Arc::clone(&hydrology.rivers));
            render.gpu_surface = if state.materials {
                None
            } else {
                state.surface.as_ref().and_then(|(_, data)| data.clone())
            };
            render.reach.clear();
            render
                .reach
                .extend(state.ranges.iter().map(|range| range * LOAD_REACH));
            render.ranges.clone_from(&state.ranges);
        }
    }
}
