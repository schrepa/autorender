//! Scene version 9 terrain (decision 051): an entity field describing a
//! heightfield terrain, generated from a layer stack or read from an `ETR1`
//! file beside the scene, and its surface (decision 054): the materials its
//! ground is made of and the biomes deciding where each lies.

use std::sync::Arc;

use engine_core::{
    Biome, Biomes, Blend, Cover, ErosionLayer, ErosionStage, HeightImage, HeightmapLayer,
    HydrologyLayer, HydrologyStage, Layer, Layers, MaterialHandle, MeshHandle, NoiseLayer,
    PathLayer, PathStage, PlacedCollider, PlacedShape, PlacementAsset, PlacementLayer,
    PlacementLod, Pyramid, Stamp, SurfaceMaterial, SurfaceProjection, Terrain, TerrainSource,
    TerrainSurface,
    glam::{Vec2, Vec3},
};
use serde::{Deserialize, Serialize};

use crate::{SceneError, SceneFiles, SceneMaterial, SceneMesh, files::valid_uri};

/// A heightfield terrain on this entity (decision 051). Requires version 9.
///
/// Its first corner (least x and z) lies at `origin` in world space,
/// whatever the entity's transform; heights rise from the origin's y. Its
/// heights come from `layers`, applied in order to every sample (later
/// layers win), or from an `ETR1` `file` beside the scene, whose tiles
/// stream by byte range. Level-0 tiles span 256 samples `spacing` meters
/// apart; coarser levels halve the tiles up to one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTerrain {
    pub origin: [f32; 3],
    /// Level-0 tiles along x and z, 1 to 4096 each. Required with layers;
    /// a file's own when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiles: Option<[u32; 2]>,
    /// Meters between level-0 samples, 1/64 to 1024. Required with layers;
    /// a file's own when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spacing: Option<f32>,
    /// The layers generating its heights, in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<SceneTerrainLayer>,
    /// An `ETR1` terrain file beside the scene, instead of layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// Its materials and biomes. Omitted, the engine's default: grass,
    /// bare dirt, rock on steep ground and snow above 650 meters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<SceneTerrainSurface>,
    /// Whether it collides where collision is installed. Omitted, true.
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub collision: bool,
}

fn yes() -> bool {
    true
}

fn is_true(value: &bool) -> bool {
    *value
}

/// A terrain's surface (decision 054): up to 255 materials, which biome
/// layers name by ID, and up to 255 biomes. Later biomes paint over earlier
/// ones where they fit; the first covers wherever none of the others do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainSurface {
    pub materials: Vec<SceneTerrainMaterial>,
    pub biomes: Vec<Biome>,
    /// What placement layers stand on the terrain (decision 057).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assets: Vec<SceneTerrainAsset>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub placements: Vec<PlacementLayer>,
}

/// An object placement layers stand on a terrain (decision 057): a `mesh`
/// and a `material` of this document, by ID, up to four coarser meshes it
/// draws at a distance, finest first (otherwise the mesh's automatic levels
/// of detail), and how far it sways in the wind (decision 058): meters at
/// ten meters up in full wind, 0 (rigid, the default) to 10.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainAsset {
    pub id: String,
    pub mesh: String,
    pub material: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lods: Vec<SceneTerrainLod>,
    #[serde(default, skip_serializing_if = "is_zero")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 10.0)))]
    pub sway: f32,
    /// Its objects' collider (decision 064); without one they never
    /// collide.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collider: Option<SceneTerrainCollider>,
}

fn is_zero(value: &f32) -> bool {
    *value == 0.0
}

/// The collider of a placement asset's objects (decision 064), in its
/// mesh's space at scale 1: its shape about `center` (the origin by
/// default), and the collision layers its objects belong to (the first by
/// default) and collide with (every one by default). Objects within
/// terrains' collision radius collide; those standing far away never do.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainCollider {
    pub shape: crate::SceneColliderShape,
    #[serde(default, skip_serializing_if = "is_origin")]
    pub center: [f32; 3],
    #[serde(default = "first_layer")]
    pub memberships: u32,
    #[serde(default = "every_layer")]
    pub filter: u32,
}

fn is_origin(center: &[f32; 3]) -> bool {
    *center == [0.0; 3]
}

fn first_layer() -> u32 {
    1
}

fn every_layer() -> u32 {
    u32::MAX
}

impl SceneTerrainCollider {
    /// The engine's collider.
    pub(crate) fn runtime(self) -> PlacedCollider {
        PlacedCollider {
            shape: match self.shape {
                crate::SceneColliderShape::Cuboid { half_extents } => {
                    PlacedShape::Cuboid { half_extents }
                }
                crate::SceneColliderShape::Sphere { radius } => PlacedShape::Sphere { radius },
            },
            center: self.center,
            memberships: self.memberships,
            filter: self.filter,
        }
    }

    /// The authored collider of an engine one.
    pub(crate) fn authored(collider: PlacedCollider) -> Self {
        Self {
            shape: match collider.shape {
                PlacedShape::Cuboid { half_extents } => {
                    crate::SceneColliderShape::Cuboid { half_extents }
                }
                PlacedShape::Sphere { radius } => crate::SceneColliderShape::Sphere { radius },
            },
            center: collider.center,
            memberships: collider.memberships,
            filter: collider.filter,
        }
    }
}

/// A coarser mesh of a placement asset, by ID, and the largest deviation
/// from the asset's mesh it shows, in mesh units; errors grow with each.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainLod {
    pub mesh: String,
    pub error: f32,
}

/// How one surface material looks: a `material` of this document whose
/// maps and factors shade it, or a linear `color` and `roughness`; the
/// meters its textures repeat over (0.01 to 1000); how sharply its height
/// map wins where it blends with another (0 to 1); whether steep ground
/// projects it from the side; and the cover growing where it lies
/// (decision 056).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainMaterial {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    #[serde(default = "gray")]
    pub color: [f32; 3],
    #[serde(default = "rough")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.03, max = 1.0)))]
    pub roughness: f32,
    #[serde(default = "four_meters")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.01, max = 1000.0)))]
    pub tiling: f32,
    #[serde(default = "half")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub height_contrast: f32,
    #[serde(default)]
    pub projection: SurfaceProjection,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cover: Option<Cover>,
}

fn gray() -> [f32; 3] {
    [0.5; 3]
}

fn rough() -> f32 {
    0.9
}

fn four_meters() -> f32 {
    4.0
}

fn half() -> f32 {
    0.5
}

/// A validated terrain whose surface materials name this document's
/// materials by index, which instantiation turns into handles.
#[derive(Clone, Debug)]
pub(crate) struct PreparedTerrain {
    pub terrain: Terrain,
    /// Each surface material's document material, if it names one.
    pub materials: Vec<Option<usize>>,
    /// Each placement asset's document mesh and material, and its coarser
    /// meshes with their errors.
    pub assets: Vec<PreparedAsset>,
}

/// A placement asset's document mesh, material, coarser meshes with their
/// errors, sway and collider.
pub(crate) type PreparedAsset = (usize, usize, Vec<(usize, f32)>, f32, Option<PlacedCollider>);

impl PreparedTerrain {
    /// The component, its surface materials and placement assets resolved
    /// through the document's `materials` and `meshes` handles.
    pub(crate) fn runtime(&self, materials: &[MaterialHandle], meshes: &[MeshHandle]) -> Terrain {
        let mut terrain = self.terrain.clone();
        for (look, index) in terrain.surface.materials.iter_mut().zip(&self.materials) {
            look.material = index.map(|index| materials[index]);
        }
        terrain.surface.assets = self
            .assets
            .iter()
            .map(|(mesh, material, lods, sway, collider)| PlacementAsset {
                mesh: meshes[*mesh],
                material: materials[*material],
                lods: lods
                    .iter()
                    .map(|&(mesh, error)| PlacementLod {
                        mesh: meshes[mesh],
                        error,
                    })
                    .collect(),
                sway: *sway,
                collider: *collider,
            })
            .collect();
        terrain
    }
}

/// One layer of a terrain's stack: procedural noise with erosion, a stamp
/// (raise, lower, flatten or cut holes), an imported heightmap, an erosion
/// bake (decision 068) of the layers before it, where rivers wear valleys
/// and slopes creep, a hydrology bake (decision 066) of them, which
/// carves channels where water gathers, fills lakes and rivers with water
/// (decision 067) and feeds the `flow`, `wetness` and `lake` masks, or a
/// path (decision 069): a road leveling the layers before it along its
/// line, which feeds the `path` mask (the `terrain_route` request finds
/// routes for them). Terrains load no tiles until their bakes are done.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SceneTerrainLayer {
    Noise(NoiseLayer),
    Stamp(Stamp),
    Heightmap(SceneHeightmap),
    Hydrology(HydrologyLayer),
    Erosion(ErosionLayer),
    Path(PathLayer),
}

/// An imported heightmap: a file of little-endian 16-bit samples, row by row
/// along z (`.r16`), `width` by `height` of them, spanning `size` meters
/// from terrain-local `min` (x and z). Samples 0 to 65535 map to heights
/// `low` to `high`; the layer fades in over `falloff` meters inside its
/// edges and combines by `blend`. `source` and `attribution` (up to 1024
/// characters each) record where its heights came from and the credit
/// their data requires, as the `terrain_import` tool writes them (decision
/// 070).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneHeightmap {
    pub uri: String,
    #[cfg_attr(feature = "schema", schemars(range(min = 2, max = 16384)))]
    pub width: u32,
    #[cfg_attr(feature = "schema", schemars(range(min = 2, max = 16384)))]
    pub height: u32,
    pub min: [f32; 2],
    pub size: [f32; 2],
    pub low: f32,
    pub high: f32,
    #[serde(default)]
    pub falloff: f32,
    #[serde(default)]
    pub blend: Blend,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(length(max = 1024)))]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(length(max = 1024)))]
    pub attribution: Option<String>,
}

/// The largest heightmap side, in samples.
const MAX_HEIGHTMAP: u32 = 16_384;
/// The most characters a heightmap's source or attribution holds.
const MAX_CREDIT: usize = 1024;

impl SceneTerrain {
    /// The files it references: its heightmaps (buffers) and its terrain
    /// file.
    pub(crate) fn files(&self) -> impl Iterator<Item = (&String, bool)> {
        self.layers
            .iter()
            .filter_map(|layer| match layer {
                SceneTerrainLayer::Heightmap(map) => Some((&map.uri, false)),
                _ => None,
            })
            .chain(self.file.iter().map(|uri| (uri, true)))
    }

    /// The surface at authored path `path`, its materials naming the
    /// document's `materials` and its assets its `meshes` and materials:
    /// the component's surface (assets left to instantiation), each surface
    /// material's document material index and each asset's mesh and
    /// material indices.
    #[allow(clippy::type_complexity)]
    fn surface(
        &self,
        path: &str,
        materials: &[SceneMaterial],
        meshes: &[SceneMesh],
    ) -> Result<(TerrainSurface, Vec<Option<usize>>, Vec<PreparedAsset>), SceneError> {
        let invalid = |field: String, rule: String| {
            SceneError::new("invalid_terrain", format!("{path}.surface.{field}"), rule)
        };
        let Some(surface) = &self.surface else {
            let looks = TerrainSurface::default();
            let count = looks.materials.len();
            return Ok((looks, vec![None; count], Vec::new()));
        };
        let mut looks = Vec::with_capacity(surface.materials.len());
        let mut indices = Vec::with_capacity(surface.materials.len());
        for (index, material) in surface.materials.iter().enumerate() {
            let at = format!("materials[{index}]");
            let look = SurfaceMaterial {
                material: None,
                color: material.color,
                roughness: material.roughness,
                tiling: material.tiling,
                height_contrast: material.height_contrast,
                projection: material.projection,
                cover: material.cover,
            };
            if let Some(Err(error)) = material.cover.as_ref().map(Cover::validate) {
                return Err(invalid(
                    format!("{at}.cover.{}", error.field),
                    error.rule.into_owned(),
                ));
            }
            if !look.is_usable() {
                return Err(invalid(
                    at,
                    "color must be in [0, 1], roughness in [0.03, 1], tiling in [0.01, 1000] and height_contrast in [0, 1]".into(),
                ));
            }
            let document = material
                .material
                .as_ref()
                .map(|id| {
                    materials
                        .iter()
                        .position(|candidate| candidate.id == *id)
                        .ok_or_else(|| {
                            invalid(
                                format!("{at}.material"),
                                "must name a material of this document".into(),
                            )
                        })
                })
                .transpose()?;
            looks.push(look);
            indices.push(document);
        }
        let mut assets = Vec::with_capacity(surface.assets.len());
        let mesh_index = |id: &str, field: String| {
            meshes
                .iter()
                .position(|candidate| candidate.id == id)
                .ok_or_else(|| invalid(field, "must name a mesh of this document".into()))
        };
        for (index, asset) in surface.assets.iter().enumerate() {
            let at = format!("assets[{index}]");
            let mesh = mesh_index(&asset.mesh, format!("{at}.mesh"))?;
            let material = materials
                .iter()
                .position(|candidate| candidate.id == asset.material)
                .ok_or_else(|| {
                    invalid(
                        format!("{at}.material"),
                        "must name a material of this document".into(),
                    )
                })?;
            let lods = asset
                .lods
                .iter()
                .enumerate()
                .map(|(level, lod)| {
                    mesh_index(&lod.mesh, format!("{at}.lods[{level}].mesh"))
                        .map(|mesh| (mesh, lod.error))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let errors: Vec<f32> = lods.iter().map(|&(_, error)| error).collect();
            if let Err((level, rule)) = PlacementLod::check(&errors) {
                let field = match level {
                    Some(level) => format!("{at}.lods[{level}].error"),
                    None => format!("{at}.lods"),
                };
                return Err(invalid(field, rule.into()));
            }
            if !(asset.sway.is_finite() && (0.0..=10.0).contains(&asset.sway)) {
                return Err(invalid(format!("{at}.sway"), "must be in [0, 10]".into()));
            }
            let collider = asset.collider.map(SceneTerrainCollider::runtime);
            if collider.is_some_and(|collider| !collider.is_usable()) {
                return Err(invalid(
                    format!("{at}.collider"),
                    "needs dimensions in [0.0001, 10000] and a finite center".into(),
                ));
            }
            assets.push((mesh, material, lods, asset.sway, collider));
        }
        let biomes = Biomes {
            materials: surface
                .materials
                .iter()
                .map(|material| material.id.clone())
                .collect(),
            biomes: surface.biomes.clone(),
            assets: surface
                .assets
                .iter()
                .map(|asset| asset.id.clone())
                .collect(),
            placements: surface.placements.clone(),
        };
        let surface =
            TerrainSurface::new(&biomes, looks).map_err(|error| invalid(error.path, error.rule))?;
        Ok((surface, indices, assets))
    }

    /// The component at authored path `path`, validated; `files` supplies
    /// heightmaps and terrain files, or with `None` they read as
    /// placeholders, and its surface materials may name the document's
    /// `materials`.
    pub(crate) fn prepare(
        &self,
        path: &str,
        files: Option<&SceneFiles>,
        (materials, meshes): (&[SceneMaterial], &[SceneMesh]),
    ) -> Result<PreparedTerrain, SceneError> {
        let (surface, indices, assets) = self.surface(path, materials, meshes)?;
        let terrain = self.terrain(path, files, surface)?;
        Ok(PreparedTerrain {
            terrain,
            materials: indices,
            assets,
        })
    }

    /// The component at authored path `path`, validated, for tools reading
    /// a document without instantiating it: its surface materials keep no
    /// engine material.
    pub fn prepare_component(
        &self,
        path: &str,
        files: Option<&SceneFiles>,
        (materials, meshes): (&[SceneMaterial], &[SceneMesh]),
    ) -> Result<Terrain, SceneError> {
        Ok(self.prepare(path, files, (materials, meshes))?.terrain)
    }

    fn terrain(
        &self,
        path: &str,
        files: Option<&SceneFiles>,
        surface: TerrainSurface,
    ) -> Result<Terrain, SceneError> {
        let invalid = |field: &str, rule: String| {
            SceneError::new("invalid_terrain", format!("{path}.{field}"), rule)
        };
        if !self.origin.iter().all(|value| value.is_finite()) {
            return Err(invalid("origin", "must be finite".into()));
        }
        let origin = Vec3::from_array(self.origin);
        let source = match (&self.file, self.layers.is_empty()) {
            (Some(_), false) => {
                return Err(invalid(
                    "file",
                    "a terrain has layers or a file, not both".into(),
                ));
            }
            (None, true) => {
                return Err(invalid("layers", "a terrain needs layers or a file".into()));
            }
            (Some(uri), true) => {
                valid_uri(uri, &format!("{path}.file"))?;
                let Some(files) = files else {
                    // Structure only: the file's own dimensions are unknown.
                    let pyramid =
                        Pyramid::new(self.tiles.unwrap_or([1, 1]), self.spacing.unwrap_or(1.0))
                            .map_err(|error| invalid("tiles", error.to_string()))?;
                    return Ok(Terrain {
                        origin,
                        pyramid,
                        source: TerrainSource::Generated(Arc::new(
                            Layers::new(Vec::new()).expect("no layers are valid"),
                        )),
                        surface,
                        collision: self.collision,
                    });
                };
                let (header, url, data) = files.terrain(uri, &format!("{path}.file"))?;
                let pyramid = *header.pyramid();
                if self
                    .tiles
                    .is_some_and(|tiles| tiles != pyramid.base_tiles())
                    || self
                        .spacing
                        .is_some_and(|spacing| spacing != pyramid.spacing(0))
                {
                    return Err(invalid(
                        "tiles",
                        "tiles and spacing, when given, must match the file's".into(),
                    ));
                }
                return Ok(Terrain {
                    origin,
                    pyramid,
                    source: TerrainSource::File {
                        header: Arc::new(header),
                        url: url.unwrap_or_default().into(),
                        data: data.map(Arc::from),
                    },
                    surface,
                    collision: self.collision,
                });
            }
            (None, false) => self.layers(path, files)?,
        };
        let (Some(tiles), Some(spacing)) = (self.tiles, self.spacing) else {
            return Err(invalid(
                "tiles",
                "terrains generated from layers need tiles and spacing".into(),
            ));
        };
        let pyramid =
            Pyramid::new(tiles, spacing).map_err(|error| invalid("tiles", error.to_string()))?;
        Ok(Terrain {
            origin,
            pyramid,
            source: TerrainSource::Generated(Arc::new(source)),
            surface,
            collision: self.collision,
        })
    }

    fn layers(&self, path: &str, files: Option<&SceneFiles>) -> Result<Layers, SceneError> {
        let layers = self
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let at = format!("{path}.layers[{index}]");
                Ok(match layer {
                    SceneTerrainLayer::Noise(noise) => Layer::Noise(*noise),
                    SceneTerrainLayer::Stamp(stamp) => Layer::Stamp(*stamp),
                    SceneTerrainLayer::Heightmap(map) => {
                        Layer::Heightmap(map.prepare(&format!("{at}.heightmap"), files)?)
                    }
                    SceneTerrainLayer::Hydrology(layer) => Layer::Hydrology(HydrologyStage {
                        layer: *layer,
                        baked: None,
                    }),
                    SceneTerrainLayer::Erosion(layer) => Layer::Erosion(ErosionStage {
                        layer: *layer,
                        baked: None,
                    }),
                    SceneTerrainLayer::Path(layer) => Layer::Path(PathStage {
                        layer: Arc::new(layer.clone()),
                        baked: None,
                    }),
                })
            })
            .collect::<Result<Vec<_>, SceneError>>()?;
        Layers::new(layers).map_err(|error| {
            SceneError::new(
                "invalid_terrain",
                format!("{path}.layers[{}].{}", error.layer, error.field),
                format!("must be finite and in [{}, {}]", error.low, error.high),
            )
        })
    }

    /// The scene form of `terrain`: its file by `file`'s name when read from
    /// one, heightmaps by the names their layers keep, and its surface
    /// (omitted when the default) by `materials`' names for the materials
    /// it uses.
    pub fn from_runtime(
        terrain: &Terrain,
        file: Option<&str>,
        materials: impl Fn(MaterialHandle) -> Option<String>,
        meshes: impl Fn(MeshHandle) -> Option<String>,
    ) -> Self {
        let surface = (terrain.surface != TerrainSurface::default()).then(|| {
            let authored = terrain.surface.rules.authored();
            SceneTerrainSurface {
                materials: authored
                    .materials
                    .iter()
                    .zip(&terrain.surface.materials)
                    .map(|(id, look)| SceneTerrainMaterial {
                        id: id.clone(),
                        material: look.material.and_then(&materials),
                        color: look.color,
                        roughness: look.roughness,
                        tiling: look.tiling,
                        height_contrast: look.height_contrast,
                        projection: look.projection,
                        cover: look.cover,
                    })
                    .collect(),
                biomes: authored.biomes.clone(),
                assets: authored
                    .assets
                    .iter()
                    .zip(&terrain.surface.assets)
                    .map(|(id, asset)| SceneTerrainAsset {
                        id: id.clone(),
                        mesh: meshes(asset.mesh).unwrap_or_default(),
                        material: materials(asset.material).unwrap_or_default(),
                        lods: asset
                            .lods
                            .iter()
                            .map(|lod| SceneTerrainLod {
                                mesh: meshes(lod.mesh).unwrap_or_default(),
                                error: lod.error,
                            })
                            .collect(),
                        sway: asset.sway,
                        collider: asset.collider.map(SceneTerrainCollider::authored),
                    })
                    .collect(),
                placements: authored.placements.clone(),
            }
        });
        let layers = match &terrain.source {
            TerrainSource::Generated(layers) => layers
                .layers()
                .iter()
                .map(|layer| match layer {
                    Layer::Noise(noise) => SceneTerrainLayer::Noise(*noise),
                    Layer::Stamp(stamp) => SceneTerrainLayer::Stamp(*stamp),
                    Layer::Hydrology(stage) => SceneTerrainLayer::Hydrology(stage.layer),
                    Layer::Erosion(stage) => SceneTerrainLayer::Erosion(stage.layer),
                    Layer::Path(stage) => SceneTerrainLayer::Path(stage.layer.as_ref().clone()),
                    Layer::Heightmap(map) => SceneTerrainLayer::Heightmap(SceneHeightmap {
                        uri: map.uri.clone().unwrap_or_default(),
                        width: map.image.width(),
                        height: map.image.height(),
                        min: map.min.to_array(),
                        size: map.size.to_array(),
                        low: map.low,
                        high: map.high,
                        falloff: map.falloff,
                        blend: map.blend,
                        source: map.source.clone(),
                        attribution: map.attribution.clone(),
                    }),
                })
                .collect(),
            TerrainSource::File { .. } => Vec::new(),
        };
        let generated = matches!(terrain.source, TerrainSource::Generated(_));
        Self {
            origin: terrain.origin.to_array(),
            tiles: generated.then(|| terrain.pyramid.base_tiles()),
            spacing: generated.then(|| terrain.pyramid.spacing(0)),
            layers,
            file: match &terrain.source {
                TerrainSource::File { url, .. } => Some(
                    file.map(str::to_owned)
                        .unwrap_or_else(|| url.rsplit('/').next().unwrap_or_default().to_owned()),
                ),
                TerrainSource::Generated(_) => None,
            },
            surface,
            collision: terrain.collision,
        }
    }
}

impl SceneHeightmap {
    fn prepare(
        &self,
        path: &str,
        files: Option<&SceneFiles>,
    ) -> Result<HeightmapLayer, SceneError> {
        valid_uri(&self.uri, &format!("{path}.uri"))?;
        for (field, text) in [("source", &self.source), ("attribution", &self.attribution)] {
            if text
                .as_ref()
                .is_some_and(|text| text.chars().count() > MAX_CREDIT)
            {
                return Err(SceneError::new(
                    "invalid_terrain",
                    format!("{path}.{field}"),
                    format!("must hold at most {MAX_CREDIT} characters"),
                ));
            }
        }
        if !(2..=MAX_HEIGHTMAP).contains(&self.width) || !(2..=MAX_HEIGHTMAP).contains(&self.height)
        {
            return Err(SceneError::new(
                "invalid_terrain",
                format!("{path}.width"),
                format!("width and height must be 2 to {MAX_HEIGHTMAP} samples"),
            ));
        }
        let image = match files {
            Some(files) => {
                let bytes = files.buffer(&self.uri, &format!("{path}.uri"))?;
                HeightImage::from_le_bytes(self.width, self.height, bytes).ok_or_else(|| {
                    SceneError::new(
                        "invalid_terrain",
                        format!("{path}.uri"),
                        format!(
                            "a {} by {} heightmap holds {} bytes of 16-bit samples",
                            self.width,
                            self.height,
                            self.width as usize * self.height as usize * 2
                        ),
                    )
                })?
            }
            // Structure only: a flat stand-in of the declared size.
            None => HeightImage::new(
                self.width,
                self.height,
                vec![0; self.width as usize * self.height as usize],
            )
            .expect("declared dimensions checked above"),
        };
        Ok(HeightmapLayer {
            image: Arc::new(image),
            min: Vec2::from_array(self.min),
            size: Vec2::from_array(self.size),
            low: self.low,
            high: self.high,
            falloff: self.falloff,
            blend: self.blend,
            uri: Some(self.uri.clone()),
            source: self.source.clone(),
            attribution: self.attribution.clone(),
        })
    }
}
