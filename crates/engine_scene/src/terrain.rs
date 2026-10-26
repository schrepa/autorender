//! Scene terrain: an entity field describing a
//! heightfield terrain generated from a layer stack, and its surface: the
//! materials its ground is made of and the biomes deciding where each
//! lies.

use engine_settings::ErrorCode;
use std::sync::Arc;

use engine_core::bevy_ecs::world::EntityWorldMut;
use engine_core::terrain_data::MAX_CREDIT;
use engine_core::{
    Biome, Biomes, Blend, Cover, ErosionLayer, ErosionStage, HeightImage, HeightmapLayer,
    HydrologyLayer, HydrologyStage, Layer, Layers, MaterialHandle, MeshHandle, NoiseLayer,
    PathLayer, PathStage, PlacedCollider, PlacedShape, PlacementAsset, PlacementLayer,
    PlacementLod, PlacementPart, Pyramid, RiversLayer, Stamp, SurfaceMaterial, SurfaceProjection,
    Surroundings, Terrain, TerrainSurface,
    glam::{Vec2, Vec3},
};
use engine_settings::{Rule, Setting};
use serde::{Deserialize, Serialize};

use crate::components::{Placement, SceneComponent};
use crate::instance::Lookups;
use crate::{SceneDocument, SceneError, SceneFiles, SceneMaterial, SceneMesh, files::valid_uri};

/// A heightfield terrain on this entity.
///
/// Its first corner (least x and z) lies at `origin` in world space,
/// whatever the entity's transform; heights rise from the origin's y. Its
/// heights come from `layers`, applied in order to every sample (later
/// layers win): an imported terrain is a heightmap layer, its known
/// rivers a rivers layer before its hydrology layer (`terrain_import`
/// writes both). Its tiles load as records, the cook's
/// or made by the same jobs. Level-0 tiles span 256 samples `spacing`
/// meters apart; coarser levels halve the tiles up to one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTerrain {
    pub origin: [f32; 3],
    /// Level-0 tiles along x and z. Required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schema",
        schemars(inner(range(min = 1, max = engine_core::Pyramid::MAX_TILES)))
    )]
    pub tiles: Option<[u32; 2]>,
    /// Meters between level-0 samples. Required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schema",
        schemars(range(min = engine_core::Pyramid::MIN_SPACING, max = engine_core::Pyramid::MAX_SPACING))
    )]
    pub spacing: Option<f32>,
    /// The layers generating its heights, in order; at least one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<SceneTerrainLayer>,
    /// Its materials and biomes. Omitted, the engine's default: grass,
    /// bare dirt, rock on steep ground and snow above 650 meters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<SceneTerrainSurface>,
    /// Whether it collides where collision is installed. Omitted, true.
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub collision: bool,
    /// Ground generated around it to the horizon.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surroundings: Option<SceneSurroundings>,
}

/// Ground generated past a terrain's edges to the horizon:
/// a coarser terrain around it from the same layers, its samples four
/// times as far apart as the terrain's finest, reaching about `reach`
/// meters past every edge (the terrain's inspection reports how far: its
/// tiles align with the terrain's sample grids). The terrain's stamps,
/// heightmaps, bakes and roads fade into its noise over `blend` meters past
/// its edges (1024 when omitted), so the two meet where it ends. It never
/// collides, places objects, grows blades or holds water.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(transform = Surroundings::state_rules)
)]
#[serde(deny_unknown_fields)]
pub struct SceneSurroundings {
    pub reach: f32,
    #[serde(default = "default_blend", skip_serializing_if = "is_default_blend")]
    pub blend: f32,
}

fn default_blend() -> f32 {
    Surroundings::default().blend
}

fn is_default_blend(value: &f32) -> bool {
    *value == default_blend()
}

fn yes() -> bool {
    true
}

fn is_true(value: &bool) -> bool {
    *value
}

/// A terrain's surface: up to 255 materials, which biome
/// layers name by ID, and up to 255 biomes. Later biomes paint over earlier
/// ones where they fit; the first covers wherever none of the others do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainSurface {
    pub materials: Vec<SceneTerrainMaterial>,
    pub biomes: Vec<Biome>,
    /// The material covering its steep ground over every biome.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steep: Option<engine_core::terrain_data::SteepLayer>,
    /// ID of the document material its roads' bridges draw
    /// with; omitted, a weathered stone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bridge: Option<String>,
    /// What placement layers stand on the terrain.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assets: Vec<SceneTerrainAsset>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub placements: Vec<PlacementLayer>,
    /// Objects authors stand on it, each at its own point.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub placed: Vec<engine_core::terrain_data::AuthoredPlacement>,
    /// How strongly each cause tints its macro colors (the tint and growth
    /// its bakes and its shape give its ground over its whole extent), and
    /// the side the midday sun stands on. Omitted, the engine's defaults.
    #[serde(default, skip_serializing_if = "is_default_causes")]
    pub macro_colors: engine_core::terrain_data::MacroCauses,
}

fn is_default_causes(causes: &engine_core::terrain_data::MacroCauses) -> bool {
    *causes == engine_core::terrain_data::MacroCauses::default()
}

/// An object placement layers stand on a terrain: a `mesh`
/// and a `material` of this document, by ID, up to four coarser meshes it
/// draws at a distance, finest first (otherwise the mesh's automatic levels
/// of detail), up to two parts drawn with materials of their own, and how
/// far it sways in the wind: meters at
/// ten meters up in full wind, 0 (rigid, the default) to 10.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(transform = state_asset_rules)
)]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainAsset {
    pub id: String,
    pub mesh: String,
    pub material: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lods: Vec<SceneTerrainLod>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<SceneTerrainPart>,
    /// ID of a simpler mesh of this document its objects draw in place of
    /// its levels into shadow maps, opaque and depth only; without one they
    /// cast their own levels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow: Option<String>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub sway: f32,
    /// Its objects' collider; without one they never
    /// collide.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collider: Option<SceneTerrainCollider>,
}

/// States the rule terrains check on an asset's sway.
#[cfg(feature = "schema")]
fn state_asset_rules(schema: &mut schemars::Schema) {
    engine_settings::state(schema, "sway", PlacementAsset::SWAY);
}

fn is_zero(value: &f32) -> bool {
    *value == 0.0
}

/// The collider of a placement asset's objects, in its
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

/// A part of a placement asset its objects draw with a `material` of its
/// own, such as a tree's masked needle cards over its opaque crown: its
/// meshes by ID, at the asset's mesh and then at each of its coarser
/// meshes, as far as it reaches; coarser levels draw without it. It takes
/// the asset's color, sway and impostor, and casts no shadow: the asset's
/// `shadow`, where it has one, is the whole object's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainPart {
    pub material: String,
    pub meshes: Vec<String>,
}

/// How one surface material looks: a `material` of this document whose
/// maps and factors shade it, or a linear `color` and `roughness`; the
/// meters its textures repeat over (0.01 to 1000); how sharply its height
/// map wins where it blends with another (0 to 1); whether steep ground
/// projects it from the side; and the cover growing where it lies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(transform = SurfaceMaterial::state_rules)
)]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainMaterial {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    #[serde(default = "gray")]
    pub color: [f32; 3],
    #[serde(default = "rough")]
    pub roughness: f32,
    #[serde(default = "four_meters")]
    pub tiling: f32,
    #[serde(default = "half")]
    pub height_contrast: f32,
    #[serde(default)]
    pub projection: SurfaceProjection,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cover: Option<Cover>,
    /// Meters its maps' heights span where the ground's relief shows:
    /// 0, flat, to 1.
    #[serde(default, skip_serializing_if = "is_flat")]
    pub relief: f32,
    /// Meters its maps' heights move the ground itself where that depth
    /// spans a few pixels: 0, none, to 2.
    #[serde(default, skip_serializing_if = "is_flat")]
    pub displacement: f32,
    /// Meters tracks sink it at most: 0, rigid, to 2.
    #[serde(default, skip_serializing_if = "is_flat")]
    pub deform: f32,
}

fn is_flat(relief: &f32) -> bool {
    *relief == 0.0
}

fn gray() -> [f32; 3] {
    SurfaceMaterial::default().color
}

fn rough() -> f32 {
    SurfaceMaterial::default().roughness
}

fn four_meters() -> f32 {
    SurfaceMaterial::default().tiling
}

fn half() -> f32 {
    SurfaceMaterial::default().height_contrast
}

/// A validated terrain whose surface materials name this document's
/// materials by index, which instantiation turns into handles.
#[derive(Clone, Debug)]
pub(crate) struct PreparedTerrain {
    pub terrain: Terrain,
    /// Each surface material's document material, if it names one, and
    /// the bridges'.
    pub materials: Vec<Option<usize>>,
    pub bridge: Option<usize>,
    /// Each placement asset's document mesh and material, and its coarser
    /// meshes with their errors.
    pub assets: Vec<PreparedAsset>,
}

/// A placement asset's document mesh, material, coarser meshes with their
/// errors, parts (each one's document material and meshes), shadow proxy's
/// document mesh, sway and collider.
pub(crate) type PreparedAsset = (
    usize,
    usize,
    Vec<(usize, f32)>,
    Vec<(usize, Vec<usize>)>,
    Option<usize>,
    f32,
    Option<PlacedCollider>,
);

/// Each of `asset`'s parts' document material and meshes, among the
/// document's `materials` and `meshes`, at authored path `at`; `invalid`
/// makes the error of a field breaking its rule.
fn prepared_parts(
    asset: &SceneTerrainAsset,
    (materials, meshes): (&[SceneMaterial], &[SceneMesh]),
    (at, invalid): (&str, &dyn Fn(String, String) -> SceneError),
) -> Result<Vec<(usize, Vec<usize>)>, SceneError> {
    let mut prepared = Vec::with_capacity(asset.parts.len());
    for (index, part) in asset.parts.iter().enumerate() {
        let field = format!("{at}.parts[{index}]");
        let material = materials
            .iter()
            .position(|candidate| candidate.id == part.material)
            .ok_or_else(|| {
                invalid(
                    format!("{field}.material"),
                    "must name a material of this document".into(),
                )
            })?;
        let part_meshes = part
            .meshes
            .iter()
            .enumerate()
            .map(|(level, mesh)| {
                meshes
                    .iter()
                    .position(|candidate| candidate.id == *mesh)
                    .ok_or_else(|| {
                        invalid(
                            format!("{field}.meshes[{level}]"),
                            "must name a mesh of this document".into(),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        prepared.push((material, part_meshes));
    }
    let counts: Vec<usize> = prepared.iter().map(|(_, meshes)| meshes.len()).collect();
    if let Err((part, rule)) = PlacementPart::check(&counts, 1 + asset.lods.len()) {
        let field = match part {
            Some(part) => format!("{at}.parts[{part}].meshes"),
            None => format!("{at}.parts"),
        };
        return Err(invalid(field, rule.into()));
    }
    Ok(prepared)
}

impl PreparedTerrain {
    /// The component, its surface materials and placement assets resolved
    /// through the document's `materials` and `meshes` handles.
    pub(crate) fn runtime(&self, materials: &[MaterialHandle], meshes: &[MeshHandle]) -> Terrain {
        let mut terrain = self.terrain.clone();
        for (look, index) in terrain.surface.materials.iter_mut().zip(&self.materials) {
            look.material = index.map(|index| materials[index]);
        }
        terrain.surface.bridge = self.bridge.map(|index| materials[index]);
        terrain.surface.assets = self
            .assets
            .iter()
            .map(
                |(mesh, material, lods, parts, shadow, sway, collider)| PlacementAsset {
                    mesh: meshes[*mesh],
                    material: materials[*material],
                    lods: lods
                        .iter()
                        .map(|&(mesh, error)| PlacementLod {
                            mesh: meshes[mesh],
                            error,
                        })
                        .collect(),
                    parts: parts
                        .iter()
                        .map(|(material, part)| PlacementPart {
                            material: materials[*material],
                            meshes: part.iter().map(|&mesh| meshes[mesh]).collect(),
                        })
                        .collect(),
                    shadow: shadow.map(|mesh| meshes[mesh]),
                    sway: *sway,
                    collider: *collider,
                },
            )
            .collect();
        terrain
    }
}

/// One layer of a terrain's stack: procedural noise with erosion, a stamp
/// (raise, lower, flatten or cut holes), an imported heightmap, rivers
/// given (drawn, or written by `terrain_import` from vector data) to the
/// hydrology layer after them, which runs them as given before deriving any
/// of its own, an erosion
/// bake of the layers before it, where rivers wear valleys
/// and slopes creep, a hydrology bake of them, which
/// carves channels where water gathers, fills lakes and rivers with water
/// and feeds the `flow`, `wetness` and `lake` masks, or a
/// path: a road leveling the layers before it along its
/// line, which feeds the `path` mask (the `terrain_route` request finds
/// routes for them). Terrains load no tiles until their bakes are done.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SceneTerrainLayer {
    Noise(NoiseLayer),
    Stamp(Stamp),
    Heightmap(SceneHeightmap),
    Rivers(RiversLayer),
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
/// their data requires, as the `terrain_import` tool writes them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(transform = state_heightmap_rules)
)]
#[serde(deny_unknown_fields)]
pub struct SceneHeightmap {
    pub uri: String,
    #[cfg_attr(feature = "schema", schemars(range(min = 2, max = MAX_HEIGHTMAP)))]
    pub width: u32,
    #[cfg_attr(feature = "schema", schemars(range(min = 2, max = MAX_HEIGHTMAP)))]
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
    #[cfg_attr(feature = "schema", schemars(length(max = MAX_CREDIT)))]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(length(max = MAX_CREDIT)))]
    pub attribution: Option<String>,
}

/// States the rules terrains check on a heightmap's placement.
#[cfg(feature = "schema")]
fn state_heightmap_rules(schema: &mut schemars::Schema) {
    for field in ["min", "size", "low", "high", "falloff"] {
        let rule =
            engine_core::HeightmapLayer::rule_of(field).expect("heightmaps declare the field");
        engine_settings::state(schema, field, rule);
    }
}

/// The largest heightmap side, in samples.
const MAX_HEIGHTMAP: u32 = 16_384;

impl SceneTerrain {
    /// The files it references: its heightmaps, as buffers.
    pub(crate) fn files(&self) -> impl Iterator<Item = &String> {
        self.layers.iter().filter_map(|layer| match layer {
            SceneTerrainLayer::Heightmap(map) => Some(&map.uri),
            SceneTerrainLayer::Noise(_)
            | SceneTerrainLayer::Stamp(_)
            | SceneTerrainLayer::Rivers(_)
            | SceneTerrainLayer::Hydrology(_)
            | SceneTerrainLayer::Erosion(_)
            | SceneTerrainLayer::Path(_) => None,
        })
    }

    /// The surface at authored path `path`, its materials naming the
    /// document's `materials` and its assets its `meshes` and materials:
    /// the component's surface (assets left to instantiation), each surface
    /// material's document material index and each asset's mesh and
    /// material indices.
    #[expect(clippy::type_complexity, reason = "the surface and its indices")]
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn surface(
        &self,
        path: &str,
        materials: &[SceneMaterial],
        meshes: &[SceneMesh],
    ) -> Result<(TerrainSurface, Vec<Option<usize>>, Vec<PreparedAsset>), SceneError> {
        let invalid = |field: String, rule: String| {
            SceneError::new(
                ErrorCode::InvalidTerrain,
                format!("{path}.surface.{field}"),
                rule,
            )
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
                relief: material.relief,
                displacement: material.displacement,
                deform: material.deform,
            };
            if let Err(error) = look.validate() {
                return Err(invalid(
                    format!("{at}.{}", error.field),
                    error.rule.into_owned(),
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
            let parts = prepared_parts(asset, (materials, meshes), (&at, &invalid))?;
            let shadow = asset
                .shadow
                .as_deref()
                .map(|id| mesh_index(id, format!("{at}.shadow")))
                .transpose()?;
            if let Err(error) = asset.sway.check("sway", PlacementAsset::SWAY) {
                return Err(invalid(format!("{at}.sway"), error.rule.into_owned()));
            }
            let collider = asset.collider.map(SceneTerrainCollider::runtime);
            if let Some(Err(error)) = collider.map(|collider| collider.check("collider", Rule::Any))
            {
                return Err(invalid(format!("{at}.collider"), error.rule.into_owned()));
            }
            assets.push((mesh, material, lods, parts, shadow, asset.sway, collider));
        }
        let biomes = Biomes {
            materials: surface
                .materials
                .iter()
                .map(|material| material.id.clone())
                .collect(),
            biomes: surface.biomes.clone(),
            steep: surface.steep.clone(),
            assets: surface
                .assets
                .iter()
                .map(|asset| asset.id.clone())
                .collect(),
            placements: surface.placements.clone(),
            placed: surface.placed.clone(),
            macro_colors: surface.macro_colors,
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
        let bridge = match self
            .surface
            .as_ref()
            .and_then(|surface| surface.bridge.as_ref())
        {
            Some(id) => Some(
                materials
                    .iter()
                    .position(|material| material.id == *id)
                    .ok_or_else(|| {
                        SceneError::new(
                            ErrorCode::InvalidTerrain,
                            format!("{path}.surface.bridge"),
                            "must name a material of this document",
                        )
                    })?,
            ),
            None => None,
        };
        let terrain = self.terrain(path, files, surface)?;
        Ok(PreparedTerrain {
            terrain,
            materials: indices,
            bridge,
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
            SceneError::new(ErrorCode::InvalidTerrain, format!("{path}.{field}"), rule)
        };
        if !self.origin.iter().all(|value| value.is_finite()) {
            return Err(invalid("origin", "must be finite".into()));
        }
        let origin = Vec3::from_array(self.origin);
        let surroundings = self.surroundings.map(|surroundings| Surroundings {
            reach: surroundings.reach,
            blend: surroundings.blend,
        });
        if let Some(Err(error)) = surroundings.map(|surroundings| surroundings.validate()) {
            return Err(invalid(
                &format!("surroundings.{}", error.field),
                error.rule.into_owned(),
            ));
        }
        if self.layers.is_empty() {
            return Err(invalid("layers", "a terrain needs layers".into()));
        }
        let source = self.layers(path, files)?;
        let (Some(tiles), Some(spacing)) = (self.tiles, self.spacing) else {
            return Err(invalid("tiles", "terrains need tiles and spacing".into()));
        };
        let pyramid =
            Pyramid::new(tiles, spacing).map_err(|error| invalid("tiles", error.to_string()))?;
        Ok(Terrain {
            origin,
            pyramid,
            layers: Arc::new(source),
            surface,
            collision: self.collision,
            edits: std::sync::Arc::from(Vec::new()),
            rivers: std::sync::Arc::from(Vec::new()),
            surroundings,
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
                    SceneTerrainLayer::Rivers(layer) => Layer::Rivers(Arc::new(layer.clone())),
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
                ErrorCode::InvalidTerrain,
                format!("{path}.layers[{}].{}", error.layer, error.field),
                error.rule,
            )
        })
    }

    /// The scene form of `terrain`: heightmaps by the names their layers
    /// keep, and its surface (omitted when the default) by `materials`'
    /// names for the materials it uses.
    pub fn from_runtime(
        terrain: &Terrain,
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
                        relief: look.relief,
                        displacement: look.displacement,
                        deform: look.deform,
                    })
                    .collect(),
                biomes: authored.biomes.clone(),
                steep: authored.steep.clone(),
                bridge: terrain.surface.bridge.and_then(&materials),
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
                        parts: asset
                            .parts
                            .iter()
                            .map(|part| SceneTerrainPart {
                                material: materials(part.material).unwrap_or_default(),
                                meshes: part
                                    .meshes
                                    .iter()
                                    .map(|&mesh| meshes(mesh).unwrap_or_default())
                                    .collect(),
                            })
                            .collect(),
                        shadow: asset.shadow.and_then(&meshes),
                        sway: asset.sway,
                        collider: asset.collider.map(SceneTerrainCollider::authored),
                    })
                    .collect(),
                placements: authored.placements.clone(),
                placed: authored.placed.clone(),
                macro_colors: authored.macro_colors,
            }
        });
        let layers = terrain
            .layers
            .layers()
            .iter()
            .map(|layer| match layer {
                Layer::Noise(noise) => SceneTerrainLayer::Noise(*noise),
                Layer::Stamp(stamp) => SceneTerrainLayer::Stamp(*stamp),
                Layer::Rivers(layer) => SceneTerrainLayer::Rivers(layer.as_ref().clone()),
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
            .collect();
        Self {
            origin: terrain.origin.to_array(),
            tiles: Some(terrain.pyramid.base_tiles()),
            spacing: Some(terrain.pyramid.spacing(0)),
            layers,
            surface,
            collision: terrain.collision,
            surroundings: terrain.surroundings.map(|surroundings| SceneSurroundings {
                reach: surroundings.reach,
                blend: surroundings.blend,
            }),
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
                    ErrorCode::InvalidTerrain,
                    format!("{path}.{field}"),
                    format!("must hold at most {MAX_CREDIT} characters"),
                ));
            }
        }
        if !(2..=MAX_HEIGHTMAP).contains(&self.width) || !(2..=MAX_HEIGHTMAP).contains(&self.height)
        {
            return Err(SceneError::new(
                ErrorCode::InvalidTerrain,
                format!("{path}.width"),
                format!("width and height must be 2 to {MAX_HEIGHTMAP} samples"),
            ));
        }
        let image = match files {
            Some(files) => {
                let bytes = files.buffer(&self.uri, &format!("{path}.uri"))?;
                HeightImage::from_le_bytes(self.width, self.height, bytes).ok_or_else(|| {
                    SceneError::new(
                        ErrorCode::InvalidTerrain,
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

/// Validated as the scene prepares its terrains.
impl engine_settings::Setting for SceneTerrain {}

impl SceneComponent for SceneTerrain {
    type Prepared = PreparedTerrain;

    // Terrains stream their own tiles.
    const PLACEMENT: Placement = Placement::Entities(
        ErrorCode::InvalidStreamed,
        "terrains stream their own tiles; author them in entities",
    );

    fn prepare(
        &self,
        path: &str,
        document: &SceneDocument,
        files: Option<&SceneFiles>,
    ) -> Result<PreparedTerrain, SceneError> {
        SceneTerrain::prepare(self, path, files, (&document.materials, &document.meshes))
    }

    fn spawn(prepared: &PreparedTerrain, entity: &mut EntityWorldMut<'_>, lookups: &Lookups) {
        entity.insert(prepared.runtime(&lookups.owned_materials, &lookups.owned_meshes));
    }
}
