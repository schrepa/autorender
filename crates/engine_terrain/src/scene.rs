use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{
    Error, ErrorCode, Expected, Joined, MaterialHandle, MeshHandle, Setting, parse_value_at,
};
use engine_places::ground::{SurfaceMaterial, SurfaceProjection};
use engine_places::terrain::{LayerKind, TerrainLayers};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};

use crate::layer::{self, TerrainLayer};
use crate::runtime::{TerrainBuilt, TerrainScene, Terrains, check_material};
use crate::{
    Biome, Biomes, Layers, PlacedCollider, PlacedShape, PlacementAsset, PlacementLayer, Pyramid,
    Surroundings, Terrain, TerrainSurface,
};

pub const GROUND_HEIGHT: &str = "ground_height";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneTerrain {
    pub origin: [f32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiles: Option<[u32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spacing: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<SceneTerrainSurface>,
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub collision: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surroundings: Option<SceneSurroundings>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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
#[serde(deny_unknown_fields)]
pub struct SceneTerrainSurface {
    pub materials: Vec<SceneTerrainMaterial>,
    pub biomes: Vec<Biome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steep: Option<crate::SteepLayer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bridge: Option<MaterialHandle>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assets: Vec<SceneTerrainAsset>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub placements: Vec<PlacementLayer>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub placed: Vec<crate::AuthoredPlacement>,
    #[serde(default, skip_serializing_if = "is_default_causes")]
    pub macro_colors: crate::MacroCauses,
}

fn is_default_causes(causes: &crate::MacroCauses) -> bool {
    *causes == crate::MacroCauses::default()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainAsset {
    pub id: String,
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collider: Option<SceneTerrainCollider>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SceneColliderShape {
    Cuboid { half_extents: [f32; 3] },
    Sphere { radius: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainCollider {
    pub shape: SceneColliderShape,
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
                SceneColliderShape::Cuboid { half_extents } => PlacedShape::Cuboid { half_extents },
                SceneColliderShape::Sphere { radius } => PlacedShape::Sphere { radius },
            },
            center: self.center,
            memberships: self.memberships,
            filter: self.filter,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainMaterial {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<MaterialHandle>,
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GroundHeight {
    through: String,
    at: [f32; 2],
}

pub fn resolve_values(
    value: &mut Value,
    named: &[(String, Arc<Layers>)],
    pyramid: &Pyramid,
    path: &str,
) -> Result<(), Error> {
    let rule = match value {
        Value::Object(fields) if fields.len() == 1 => fields.get(GROUND_HEIGHT).cloned(),
        Value::Null
        | Value::Bool(_)
        | Value::Number(_)
        | Value::String(_)
        | Value::Array(_)
        | Value::Object(_) => None,
    };
    if let Some(rule) = rule {
        *value = ground_height(rule, named, pyramid, &format!("{path}.{GROUND_HEIGHT}"))?;
        return Ok(());
    }
    match value {
        Value::Object(fields) => {
            for (key, field) in fields.iter_mut().filter(|(_, field)| nested(field)) {
                resolve_values(field, named, pyramid, &format!("{path}.{key}"))?;
            }
        }
        Value::Array(items) => {
            for (index, item) in items
                .iter_mut()
                .enumerate()
                .filter(|(_, item)| nested(item))
            {
                resolve_values(item, named, pyramid, &format!("{path}[{index}]"))?;
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
    Ok(())
}

fn nested(value: &Value) -> bool {
    value.is_object() || value.is_array()
}

fn ground_height(
    rule: Value,
    named: &[(String, Arc<Layers>)],
    pyramid: &Pyramid,
    path: &str,
) -> Result<Value, Error> {
    let GroundHeight { through, at }: GroundHeight = parse_value_at(rule, path)?;
    let Some((_, layers)) = named.iter().find(|(id, _)| *id == through) else {
        return Err(
            Error::new(ErrorCode::UnknownReference, format!("{path}.through"))
                .got(&through)
                .variants(named.iter().map(|(id, _)| id.clone())),
        );
    };
    for (index, coordinate) in at.into_iter().enumerate() {
        if !coordinate.is_finite() || coordinate.abs() > Layers::MAX_COORDINATE {
            let bound = f64::from(Layers::MAX_COORDINATE);
            return Err(
                Error::new(ErrorCode::InvalidNumber, format!("{path}.at[{index}]"))
                    .expected(Expected::Finite)
                    .range(-bound, bound)
                    .got(coordinate),
            );
        }
    }
    let height = crate::tile::height_at(layers, pyramid, Vec2::from(at));
    Number::from_f64(f64::from(height))
        .map(Value::Number)
        .ok_or_else(|| Error::new(ErrorCode::InvalidNumber, path).expected(Expected::Finite))
}

impl SceneTerrain {
    fn surface(&self, path: &str) -> Result<TerrainSurface, Error> {
        let broken = |error: engine_core::InvalidSetting, at: &str| {
            error.error(ErrorCode::InvalidValue, &format!("{path}.surface.{at}"))
        };
        let Some(surface) = &self.surface else {
            return Ok(TerrainSurface::default());
        };
        let mut looks = Vec::with_capacity(surface.materials.len());
        for (index, material) in surface.materials.iter().enumerate() {
            let at = format!("materials[{index}]");
            let look = SurfaceMaterial {
                material: material.material,
                color: material.color,
                roughness: material.roughness,
                tiling: material.tiling,
                height_contrast: material.height_contrast,
                projection: material.projection,
                depth: material.depth,
            };
            if let Err(error) = check_material(&look) {
                return Err(broken(error, &at));
            }
            looks.push(look);
        }
        let mut assets = Vec::with_capacity(surface.assets.len());
        for (index, asset) in surface.assets.iter().enumerate() {
            let at = format!("assets[{index}]");
            let collider = asset.collider.map(SceneTerrainCollider::runtime);
            if let Some(Err(error)) =
                collider.map(|collider| collider.check("collider", engine_core::Rule::Any))
            {
                return Err(broken(error, &at));
            }
            assets.push(PlacementAsset {
                mesh: asset.mesh,
                material: asset.material,
                collider,
            });
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
        let mut built = TerrainSurface::new(&biomes, looks).map_err(|error| {
            let error = if matches!(error.code, ErrorCode::UnknownReference) {
                error
            } else {
                error.coded(ErrorCode::InvalidValue)
            };
            error.within(&format!("{path}.surface"))
        })?;
        built.assets = assets;
        built.bridge = surface.bridge;
        Ok(built)
    }

    pub(crate) fn runtime<'k>(
        &self,
        path: &str,
        kinds: &dyn Fn(&str) -> Option<&'k LayerKind>,
    ) -> Result<Terrain, Error> {
        let surface = self.surface(path)?;
        let invalid = |field: &str| Error::new(ErrorCode::InvalidValue, format!("{path}.{field}"));
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
            return Err(error.error(ErrorCode::InvalidValue, &format!("{path}.surroundings")));
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
            .map_err(|error| error.coded(ErrorCode::InvalidValue).within(path))?;
        let source = self.layers(path, kinds, &pyramid)?;
        surface
            .rules
            .check_maps(&source)
            .map_err(|error| error.within(&format!("{path}.surface")))?;
        Ok(Terrain {
            origin,
            pyramid,
            layers: Arc::new(source),
            surface,
            collision: self.collision,
            edits: Arc::from(Vec::new()),
            rivers: Arc::from(Vec::new()),
            surroundings,
        })
    }

    fn layers<'k>(
        &self,
        path: &str,
        kinds: &dyn Fn(&str) -> Option<&'k LayerKind>,
        pyramid: &Pyramid,
    ) -> Result<Layers, Error> {
        let mut layers: Vec<Arc<dyn TerrainLayer>> = Vec::with_capacity(self.layers.len());
        let mut ids: Vec<&str> = Vec::new();
        let mut named: Vec<(String, Arc<Layers>)> = Vec::new();
        for (index, authored) in self.layers.iter().enumerate() {
            let at = format!("{path}.layers[{index}]");
            let mut item = authored.clone();
            resolve_values(&mut item, &named, pyramid, &at)?;
            let name = item
                .get(layer::TYPE_KEY)
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    Error::new(ErrorCode::MissingField, format!("{at}.{}", layer::TYPE_KEY))
                        .expected(Expected::Present)
                })?;
            let kind = kinds(name).ok_or_else(|| {
                Error::new(ErrorCode::UnknownValue, format!("{at}.{}", layer::TYPE_KEY))
                    .expected(Expected::Present)
                    .got(name)
            })?;
            let mut settings = item.clone();
            if let Some(fields) = settings.as_object_mut() {
                fields.remove(layer::TYPE_KEY);
                fields.remove(layer::ID_KEY);
            }
            layers.push((kind.from_json)(&settings).map_err(|error| error.within(&at))?);
            let Some(id) = authored.get(layer::ID_KEY) else {
                continue;
            };
            let id_path = format!("{at}.{}", layer::ID_KEY);
            let id = id.as_str().ok_or_else(|| {
                Error::new(ErrorCode::InvalidType, &id_path).expected(Expected::String)
            })?;
            if !engine_core::is_id(id) {
                return Err(Error::new(ErrorCode::InvalidId, id_path)
                    .expected(Expected::Id)
                    .at_most(engine_core::MAX_ID_BYTES as u64)
                    .got(id));
            }
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
        Layers::new(layers).map_err(|error| error.coded(ErrorCode::InvalidValue).within(path))
    }
}

pub(crate) fn build_terrains(
    kinds: Joined<TerrainLayers>,
    mut built: Query<(Entity, Ref<TerrainScene>, &mut TerrainBuilt)>,
    mut terrains: ResMut<Terrains>,
) {
    let found = |name: &str| kinds.iter().find(|kind| kind.name == name);
    for (entity, scene, mut terrain) in &mut built {
        if !scene.is_changed() {
            continue;
        }
        terrains.refused.retain(|(refused, _)| *refused != entity);
        match scene.0.runtime("$.terrain", &found) {
            Ok(made) => terrain.0 = Some(made),
            Err(error) => {
                terrain.0 = None;
                terrains.refused.push((entity, error));
            }
        }
    }
}
