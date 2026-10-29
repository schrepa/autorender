use engine_settings::ErrorCode;
use std::sync::Arc;

use engine_core::bevy_ecs::world::{EntityWorldMut, World};
use engine_core::glam::Vec3;
use engine_core::{Error, Expected, MaterialHandle, MeshHandle};
use engine_settings::{Rule, Setting};
use engine_terrain::layer::{self, LayerFiles, TerrainLayer, TerrainLayerKinds};
use engine_terrain::{
    Biome, Biomes, Cover, Layers, PlacedCollider, PlacedShape, PlacementAsset, PlacementLayer,
    PlacementLod, PlacementPart, Pyramid, SurfaceMaterial, SurfaceProjection, Surroundings,
    Terrain, TerrainSurface,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::components::{Placement, SceneComponent};
use crate::instance::Lookups;
use crate::validation::{identifier, unknown};
use crate::{SceneDocument, SceneFiles, SceneMaterial, SceneMesh, files::valid_uri};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTerrain {
    pub origin: [f32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schema",
        schemars(inner(range(min = 1, max = engine_terrain::Pyramid::MAX_TILES)))
    )]
    pub tiles: Option<[u32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schema",
        schemars(range(min = engine_terrain::Pyramid::MIN_SPACING, max = engine_terrain::Pyramid::MAX_SPACING))
    )]
    pub spacing: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "layer_list_schema"))]
    pub layers: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<SceneTerrainSurface>,
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub collision: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surroundings: Option<SceneSurroundings>,
}

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainSurface {
    pub materials: Vec<SceneTerrainMaterial>,
    pub biomes: Vec<Biome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steep: Option<engine_terrain::SteepLayer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bridge: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assets: Vec<SceneTerrainAsset>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub placements: Vec<PlacementLayer>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub placed: Vec<engine_terrain::AuthoredPlacement>,
    #[serde(default, skip_serializing_if = "is_default_causes")]
    pub macro_colors: engine_terrain::MacroCauses,
}

fn is_default_causes(causes: &engine_terrain::MacroCauses) -> bool {
    *causes == engine_terrain::MacroCauses::default()
}

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow: Option<String>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub sway: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collider: Option<SceneTerrainCollider>,
}

#[cfg(feature = "schema")]
fn state_asset_rules(schema: &mut schemars::Schema) {
    engine_settings::state(schema, "sway", PlacementAsset::SWAY);
}

fn is_zero(value: &f32) -> bool {
    *value == 0.0
}

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainLod {
    pub mesh: String,
    pub error: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainPart {
    pub material: String,
    pub meshes: Vec<String>,
}

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
    #[serde(default, skip_serializing_if = "is_flat")]
    pub depth: f32,
}

fn is_flat(meters: &f32) -> bool {
    *meters == 0.0
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

#[derive(Clone, Debug)]
pub(crate) struct PreparedTerrain {
    pub terrain: Terrain,
    pub materials: Vec<Option<usize>>,
    pub bridge: Option<usize>,
    pub assets: Vec<PreparedAsset>,
}

pub(crate) type PreparedAsset = (
    usize,
    usize,
    Vec<(usize, f32)>,
    Vec<(usize, Vec<usize>)>,
    Option<usize>,
    f32,
    Option<PlacedCollider>,
);

fn prepared_parts(
    asset: &SceneTerrainAsset,
    (materials, meshes): (&[SceneMaterial], &[SceneMesh]),
    (at, invalid): (&str, &dyn Fn(String) -> Error),
) -> Result<Vec<(usize, Vec<usize>)>, Error> {
    let mut prepared = Vec::with_capacity(asset.parts.len());
    for (index, part) in asset.parts.iter().enumerate() {
        let field = format!("{at}.parts[{index}]");
        let material = materials
            .iter()
            .position(|candidate| candidate.id == part.material)
            .ok_or_else(|| {
                unknown(
                    invalid(format!("{field}.material")),
                    &part.material,
                    materials.iter().map(|material| material.id.as_str()),
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
                        unknown(
                            invalid(format!("{field}.meshes[{level}]")),
                            mesh,
                            meshes.iter().map(|mesh| mesh.id.as_str()),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        prepared.push((material, part_meshes));
    }
    let counts: Vec<usize> = prepared.iter().map(|(_, meshes)| meshes.len()).collect();
    let levels = 1 + asset.lods.len();
    if let Err(error) = PlacementPart::check(&counts, levels) {
        return Err(error.within(at));
    }
    Ok(prepared)
}

impl PreparedTerrain {
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

#[cfg(feature = "schema")]
const LAYER_DEFINITION: &str = "SceneTerrainLayer";

#[cfg(feature = "schema")]
fn layer_list_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
    let mut kinds = layer::layer_kinds();
    kinds.sort_by_key(|kind| kind.name);
    let definitions = generator.definitions_mut();
    let variants: Vec<Value> = kinds
        .into_iter()
        .map(|kind| {
            let definition = kind_definition(kind, definitions);
            serde_json::json!({
                "allOf": [{ "$ref": format!("#/$defs/{definition}") }],
                "properties": {
                    (layer::TYPE_KEY): { "const": kind.name },
                    (layer::ID_KEY): { "type": "string" },
                },
                "required": [layer::TYPE_KEY],
                "unevaluatedProperties": false,
            })
        })
        .collect();
    definitions.insert(
        LAYER_DEFINITION.to_owned(),
        serde_json::json!({ "oneOf": variants }),
    );
    schemars::json_schema!({
        "type": "array",
        "items": { "$ref": format!("#/$defs/{LAYER_DEFINITION}") },
    })
}

#[cfg(feature = "schema")]
fn kind_definition(
    kind: &layer::LayerKind,
    definitions: &mut serde_json::Map<String, Value>,
) -> String {
    let mut schema = serde_json::to_value((kind.schema)()).expect("schema_serializes");
    let Some(fields) = schema.as_object_mut() else {
        definitions.insert(kind.name.to_owned(), schema);
        return kind.name.to_owned();
    };
    fields.remove("$schema");
    if let Some(Value::Object(nested)) = fields.remove("$defs") {
        for (name, definition) in nested {
            definitions.entry(name).or_insert(definition);
        }
    }
    fields.remove("additionalProperties");
    let title = fields
        .remove("title")
        .and_then(|title| title.as_str().map(str::to_owned))
        .unwrap_or_else(|| kind.name.to_owned());
    definitions.insert(title.clone(), schema);
    title
}

struct LayerBuffers<'a>(Option<&'a SceneFiles>);

impl LayerFiles for LayerBuffers<'_> {
    fn buffer(&self, uri: &str, path: &str) -> Result<Option<Arc<[u8]>>, Error> {
        self.0
            .map(|files| files.buffer(uri, path).map(Arc::<[u8]>::from))
            .transpose()
    }
}

fn layer_files(item: &Value) -> Vec<String> {
    let Some(object) = item.as_object() else {
        return Vec::new();
    };
    let Some(kind) = object
        .get(layer::TYPE_KEY)
        .and_then(Value::as_str)
        .and_then(layer::layer_kind)
    else {
        return Vec::new();
    };
    let mut settings = object.clone();
    settings.remove(layer::TYPE_KEY);
    settings.remove(layer::ID_KEY);
    (kind.files)(&Value::Object(settings))
}

fn file_field(item: &Value, uri: &str, at: &str) -> String {
    item.as_object()
        .and_then(|object| object.iter().find(|(_, value)| value.as_str() == Some(uri)))
        .map_or_else(|| at.to_owned(), |(field, _)| format!("{at}.{field}"))
}

impl SceneTerrain {
    pub(crate) fn files(&self) -> Vec<String> {
        self.layers.iter().flat_map(layer_files).collect()
    }

    pub(crate) fn check_layers(&self, world: &World, path: &str) -> Result<(), Error> {
        let offered = world.get_resource::<TerrainLayerKinds>();
        for (index, item) in self.layers.iter().enumerate() {
            let Some(name) = item.get(layer::TYPE_KEY).and_then(Value::as_str) else {
                continue;
            };
            if offered.is_none_or(|kinds| kinds.get(name).is_none()) {
                return Err(Error::new(
                    ErrorCode::MissingPlugin,
                    format!("{path}.layers[{index}].{}", layer::TYPE_KEY),
                )
                .expected(Expected::Present)
                .got(name));
            }
        }
        Ok(())
    }

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
    ) -> Result<(TerrainSurface, Vec<Option<usize>>, Vec<PreparedAsset>), Error> {
        let invalid = |field: String| {
            Error::new(ErrorCode::InvalidTerrain, format!("{path}.surface.{field}"))
        };
        let broken = |error: engine_settings::InvalidSetting, at: &str| {
            error.error(ErrorCode::InvalidTerrain, &format!("{path}.surface.{at}"))
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
                depth: material.depth,
            };
            if let Err(error) = look.validate() {
                return Err(broken(error, &at));
            }
            let document = material
                .material
                .as_ref()
                .map(|id| {
                    materials
                        .iter()
                        .position(|candidate| candidate.id == *id)
                        .ok_or_else(|| {
                            unknown(
                                invalid(format!("{at}.material")),
                                id,
                                materials.iter().map(|material| material.id.as_str()),
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
                .ok_or_else(|| {
                    unknown(
                        invalid(field),
                        id,
                        meshes.iter().map(|mesh| mesh.id.as_str()),
                    )
                })
        };
        for (index, asset) in surface.assets.iter().enumerate() {
            let at = format!("assets[{index}]");
            let mesh = mesh_index(&asset.mesh, format!("{at}.mesh"))?;
            let material = materials
                .iter()
                .position(|candidate| candidate.id == asset.material)
                .ok_or_else(|| {
                    unknown(
                        invalid(format!("{at}.material")),
                        &asset.material,
                        materials.iter().map(|material| material.id.as_str()),
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
            if let Err(error) = PlacementLod::check(&errors) {
                return Err(error.within(&at));
            }
            let parts = prepared_parts(asset, (materials, meshes), (&at, &invalid))?;
            let shadow = asset
                .shadow
                .as_deref()
                .map(|id| mesh_index(id, format!("{at}.shadow")))
                .transpose()?;
            if let Err(error) = asset.sway.check("sway", PlacementAsset::SWAY) {
                return Err(broken(error, &at));
            }
            let collider = asset.collider.map(SceneTerrainCollider::runtime);
            if let Some(Err(error)) = collider.map(|collider| collider.check("collider", Rule::Any))
            {
                return Err(broken(error, &at));
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
        let surface = TerrainSurface::new(&biomes, looks).map_err(|error| {
            error
                .coded(ErrorCode::InvalidTerrain)
                .within(&format!("{path}.surface"))
        })?;
        Ok((surface, indices, assets))
    }

    pub(crate) fn prepare(
        &self,
        path: &str,
        files: Option<&SceneFiles>,
        (materials, meshes): (&[SceneMaterial], &[SceneMesh]),
    ) -> Result<PreparedTerrain, Error> {
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
                        unknown(
                            Error::new(ErrorCode::InvalidTerrain, format!("{path}.surface.bridge")),
                            id,
                            materials.iter().map(|material| material.id.as_str()),
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

    pub fn prepare_component(
        &self,
        path: &str,
        files: Option<&SceneFiles>,
        (materials, meshes): (&[SceneMaterial], &[SceneMesh]),
    ) -> Result<Terrain, Error> {
        Ok(self.prepare(path, files, (materials, meshes))?.terrain)
    }

    fn terrain(
        &self,
        path: &str,
        files: Option<&SceneFiles>,
        surface: TerrainSurface,
    ) -> Result<Terrain, Error> {
        let invalid =
            |field: &str| Error::new(ErrorCode::InvalidTerrain, format!("{path}.{field}"));
        if !self.origin.iter().all(|value| value.is_finite()) {
            return Err(invalid("origin")
                .expected(Expected::Finite)
                .got(self.origin));
        }
        let origin = Vec3::from_array(self.origin);
        let surroundings = self.surroundings.map(|surroundings| Surroundings {
            reach: surroundings.reach,
            blend: surroundings.blend,
        });
        if let Some(Err(error)) = surroundings.map(|surroundings| surroundings.validate()) {
            return Err(error.error(ErrorCode::InvalidTerrain, &format!("{path}.surroundings")));
        }
        if self.layers.is_empty() {
            return Err(invalid("layers").expected(Expected::NonEmpty));
        }
        let (Some(tiles), Some(spacing)) = (self.tiles, self.spacing) else {
            let field = if self.tiles.is_none() {
                "tiles"
            } else {
                "spacing"
            };
            return Err(invalid(field).expected(Expected::Present));
        };
        let pyramid = Pyramid::new(tiles, spacing)
            .map_err(|error| error.coded(ErrorCode::InvalidTerrain).within(path))?;
        let source = self.layers(path, files, &pyramid)?;
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

    fn layers(
        &self,
        path: &str,
        files: Option<&SceneFiles>,
        pyramid: &Pyramid,
    ) -> Result<Layers, Error> {
        let buffers = LayerBuffers(files);
        let mut layers: Vec<Arc<dyn TerrainLayer>> = Vec::with_capacity(self.layers.len());
        let mut ids: Vec<&str> = Vec::new();
        let mut named: Vec<(String, Arc<Layers>)> = Vec::new();
        for (index, authored) in self.layers.iter().enumerate() {
            let at = format!("{path}.layers[{index}]");
            let mut item = authored.clone();
            crate::rules::resolve_values(&mut item, &named, pyramid, &at)?;
            for uri in layer_files(&item) {
                valid_uri(&uri, &file_field(&item, &uri, &at))?;
            }
            layers.push(layer::layer_from_json(&item, &buffers, &at)?);
            let Some(id) = authored.get(layer::ID_KEY) else {
                continue;
            };
            let id_path = format!("{at}.{}", layer::ID_KEY);
            let id = id.as_str().ok_or_else(|| {
                Error::new(ErrorCode::InvalidType, &id_path).expected(Expected::String)
            })?;
            identifier(id, &id_path)?;
            if ids.contains(&id) {
                return Err(Error::new(ErrorCode::DuplicateId, id_path)
                    .expected(Expected::Unique)
                    .got(id));
            }
            ids.push(id);
            if let Ok(through) = Layers::new(layers.clone()) {
                named.push((id.to_owned(), Arc::new(through)));
            }
        }
        Layers::new(layers).map_err(|error| error.coded(ErrorCode::InvalidTerrain).within(path))
    }

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
                        depth: look.depth,
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
            .map(|kept| layer::layer_json(&**kept))
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

impl engine_settings::Setting for SceneTerrain {}

impl SceneComponent for SceneTerrain {
    type Prepared = PreparedTerrain;

    const PLACEMENT: Placement = Placement::Entities(ErrorCode::InvalidStreamed);

    fn prepare(
        &self,
        path: &str,
        document: &SceneDocument,
        files: Option<&SceneFiles>,
    ) -> Result<PreparedTerrain, Error> {
        SceneTerrain::prepare(self, path, files, (&document.materials, &document.meshes))
    }

    fn spawn(prepared: &PreparedTerrain, entity: &mut EntityWorldMut<'_>, lookups: &Lookups) {
        entity.insert(prepared.runtime(&lookups.owned_materials, &lookups.owned_meshes));
    }
}
