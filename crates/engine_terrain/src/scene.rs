use std::sync::Arc;

use engine_core::{
    ComponentKind, ComponentKindOf, Error, ErrorCode, Expected, MaterialHandle, MeshHandle,
    Placement, Rule, SceneCx, ScenePath, Setting, parse_value_at,
};
use engine_heightfield::layer::{LayerFiles, LayerKind};
use engine_heightfield::tile::{
    AuthoredPlacement, Biome, Biomes, MacroCauses, PlacementLayer, SteepLayer,
};
use engine_places::ground::{SurfaceMaterial, SurfaceProjection};
use engine_places::terrain::TerrainLayers;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value, json};

use crate::layer::{self, TerrainLayer};
use crate::runtime::{MATERIAL_RULES, TerrainPrefetch, TerrainScene, check_material};
use crate::{
    Layers, PlacedCollider, PlacedShape, PlacementAsset, Pyramid, Surroundings, Terrain,
    TerrainSurface,
};

pub const GROUND_HEIGHT: &str = "ground_height";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SceneTerrain {
    pub origin: [f32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(inner(range(min = 1, max = crate::Pyramid::MAX_TILES)))]
    pub tiles: Option<[u32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = crate::Pyramid::MIN_SPACING, max = crate::Pyramid::MAX_SPACING))]
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[schemars(transform = Surroundings::state_rules)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainSurface {
    pub materials: Vec<SceneTerrainMaterial>,
    pub biomes: Vec<Biome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steep: Option<SteepLayer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bridge: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assets: Vec<SceneTerrainAsset>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub placements: Vec<PlacementLayer>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub placed: Vec<AuthoredPlacement>,
    #[serde(default, skip_serializing_if = "is_default_causes")]
    pub macro_colors: MacroCauses,
}

fn is_default_causes(causes: &MacroCauses) -> bool {
    *causes == MacroCauses::default()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SceneTerrainAsset {
    pub id: String,
    pub mesh: String,
    pub material: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collider: Option<SceneTerrainCollider>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SceneColliderShape {
    Cuboid {
        #[schemars(inner(range(
            min = crate::MIN_COLLIDER_EXTENT,
            max = crate::MAX_COLLIDER_EXTENT
        )))]
        half_extents: [f32; 3],
    },
    Sphere {
        #[schemars(range(min = crate::MIN_COLLIDER_EXTENT, max = crate::MAX_COLLIDER_EXTENT))]
        radius: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[schemars(transform = material_rules)]
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
    #[serde(default, skip_serializing_if = "is_flat")]
    pub depth: f32,
}

fn material_rules(schema: &mut schemars::Schema) {
    for (field, rule) in MATERIAL_RULES {
        engine_core::fields::state(schema, field, rule);
    }
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

pub struct TerrainAssets<'a> {
    pub materials: Vec<(&'a str, MaterialHandle)>,
    pub meshes: Vec<(&'a str, MeshHandle)>,
}

fn handle_of<H: Copy>(table: &[(&str, H)], id: &str) -> Option<H> {
    table
        .iter()
        .find(|(candidate, _)| *candidate == id)
        .map(|&(_, handle)| handle)
}

fn ids_of<'a, H>(table: &'a [(&'a str, H)]) -> impl Iterator<Item = &'a str> {
    table.iter().map(|&(id, _)| id)
}

const LAYER_DEFINITION: &str = "SceneTerrainLayer";

fn layer_list_schema(
    kinds: &[&'static LayerKind],
    definitions: &mut serde_json::Map<String, Value>,
) -> Value {
    let mut kinds = kinds.to_vec();
    kinds.sort_by_key(|kind| kind.name);
    let variants: Vec<Value> = kinds
        .into_iter()
        .map(|kind| {
            let definition = kind_definition(kind, definitions);
            json!({
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
    definitions.insert(LAYER_DEFINITION.to_owned(), json!({ "oneOf": variants }));
    json!({
        "type": "array",
        "items": { "$ref": format!("#/$defs/{LAYER_DEFINITION}") },
    })
}

fn kind_definition(kind: &LayerKind, definitions: &mut serde_json::Map<String, Value>) -> String {
    let mut schema = (kind.schema)();
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

struct LayerBuffers<'a, 'c>(&'a SceneCx<'c>);

impl LayerFiles for LayerBuffers<'_, '_> {
    fn buffer(&self, uri: &str, path: &str) -> Result<Option<Arc<[u8]>>, Error> {
        self.0.buffer(uri, path)
    }
}

fn listed(cx: &SceneCx<'_>) -> Vec<&'static LayerKind> {
    cx.members::<TerrainLayers>().copied().collect()
}

fn kind_settings(
    item: &Value,
    kinds: &[&'static LayerKind],
) -> Option<(&'static LayerKind, Value)> {
    let object = item.as_object()?;
    let name = object.get(layer::TYPE_KEY).and_then(Value::as_str)?;
    let kind = *kinds.iter().find(|kind| kind.name == name)?;
    let mut settings = object.clone();
    settings.remove(layer::TYPE_KEY);
    settings.remove(layer::ID_KEY);
    Some((kind, Value::Object(settings)))
}

fn layer_files(item: &Value, kinds: &[&'static LayerKind]) -> Vec<String> {
    kind_settings(item, kinds).map_or_else(Vec::new, |(kind, settings)| (kind.files)(&settings))
}

fn layer_points(item: &Value, kinds: &[&'static LayerKind]) -> Option<Vec<[f32; 2]>> {
    kind_settings(item, kinds).and_then(|(kind, settings)| (kind.points)(&settings))
}

fn file_field(item: &Value, uri: &str, at: &str) -> String {
    item.as_object()
        .and_then(|object| object.iter().find(|(_, value)| value.as_str() == Some(uri)))
        .map_or_else(|| at.to_owned(), |(field, _)| format!("{at}.{field}"))
}

fn check_layers(
    terrain: &SceneTerrain,
    kinds: &[&'static LayerKind],
    path: &str,
) -> Result<(), Error> {
    for (index, item) in terrain.layers.iter().enumerate() {
        let Some(name) = item.get(layer::TYPE_KEY).and_then(Value::as_str) else {
            continue;
        };
        if !kinds.iter().any(|kind| kind.name == name) {
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

fn unknown<S: Into<String>>(error: Error, id: &str, known: impl IntoIterator<Item = S>) -> Error {
    let known: Vec<String> = known.into_iter().map(Into::into).collect();
    let error = error.got(id);
    match engine_codec::closest(id, &known) {
        Some(nearest) => error.nearest(nearest),
        None => error,
    }
}

fn identifier(id: &str, path: &str) -> Result<(), Error> {
    if engine_core::is_id(id) {
        return Ok(());
    }
    Err(Error::new(ErrorCode::InvalidId, path)
        .expected(Expected::Id)
        .at_most(engine_core::MAX_ID_BYTES as u64)
        .got(id))
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
    let height = layers.height_at(pyramid, Vec2::from(at));
    Number::from_f64(f64::from(height))
        .map(Value::Number)
        .ok_or_else(|| Error::new(ErrorCode::InvalidNumber, path).expected(Expected::Finite))
}

impl SceneTerrain {
    fn surface(&self, path: &str, assets: &TerrainAssets<'_>) -> Result<TerrainSurface, Error> {
        let invalid = |field: String| {
            Error::new(ErrorCode::InvalidTerrain, format!("{path}.surface.{field}"))
        };
        let broken = |error: engine_core::InvalidSetting, at: &str| {
            error.error(ErrorCode::InvalidTerrain, &format!("{path}.surface.{at}"))
        };
        let Some(surface) = &self.surface else {
            return Ok(TerrainSurface::default());
        };
        let material_index = |id: &str, field: String| {
            handle_of(&assets.materials, id)
                .ok_or_else(|| unknown(invalid(field), id, ids_of(&assets.materials)))
        };
        let mut looks = Vec::with_capacity(surface.materials.len());
        for (index, material) in surface.materials.iter().enumerate() {
            let at = format!("materials[{index}]");
            let mut look = SurfaceMaterial {
                name: Arc::from(material.id.as_str()),
                material: None,
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
            look.material = material
                .material
                .as_ref()
                .map(|id| material_index(id, format!("{at}.material")))
                .transpose()?;
            looks.push(look);
        }
        let mut placed = Vec::with_capacity(surface.assets.len());
        for (index, asset) in surface.assets.iter().enumerate() {
            let at = format!("assets[{index}]");
            let mesh = handle_of(&assets.meshes, &asset.mesh).ok_or_else(|| {
                unknown(
                    invalid(format!("{at}.mesh")),
                    &asset.mesh,
                    ids_of(&assets.meshes),
                )
            })?;
            let material = material_index(&asset.material, format!("{at}.material"))?;
            let collider = asset.collider.map(SceneTerrainCollider::runtime);
            if let Some(Err(error)) = collider.map(|collider| collider.check("collider", Rule::Any))
            {
                return Err(broken(error, &at));
            }
            placed.push(PlacementAsset {
                mesh,
                material,
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
            let error = if matches!(error.code, ErrorCode::UnknownMap | ErrorCode::UnreadableMap) {
                error
            } else {
                error.coded(ErrorCode::InvalidTerrain)
            };
            error.within(&format!("{path}.surface"))
        })?;
        built.assets = placed;
        built.bridge = surface
            .bridge
            .as_ref()
            .map(|id| {
                handle_of(&assets.materials, id).ok_or_else(|| {
                    unknown(
                        Error::new(ErrorCode::InvalidTerrain, format!("{path}.surface.bridge")),
                        id,
                        ids_of(&assets.materials),
                    )
                })
            })
            .transpose()?;
        Ok(built)
    }

    pub fn prepare(
        &self,
        path: &str,
        (files, kinds): (&dyn LayerFiles, &[&'static LayerKind]),
        assets: &TerrainAssets<'_>,
    ) -> Result<Terrain, Error> {
        check_layers(self, kinds, path)?;
        let surface = self.surface(path, assets)?;
        self.terrain(path, (files, kinds), surface)
    }

    fn terrain(
        &self,
        path: &str,
        (files, kinds): (&dyn LayerFiles, &[&'static LayerKind]),
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
        let source = self.layers(path, (files, kinds), &pyramid)?;
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

    fn layers(
        &self,
        path: &str,
        (files, kinds): (&dyn LayerFiles, &[&'static LayerKind]),
        pyramid: &Pyramid,
    ) -> Result<Layers, Error> {
        let mut layers: Vec<Arc<dyn TerrainLayer>> = Vec::with_capacity(self.layers.len());
        let mut ids: Vec<&str> = Vec::new();
        let mut named: Vec<(String, Arc<Layers>)> = Vec::new();
        for (index, authored) in self.layers.iter().enumerate() {
            let at = format!("{path}.layers[{index}]");
            let mut item = authored.clone();
            resolve_values(&mut item, &named, pyramid, &at)?;
            for uri in layer_files(&item, kinds) {
                engine_core::valid_uri(&uri, &file_field(&item, &uri, &at))?;
            }
            layers.push(layer::layer_from_json(&item, files, &at, kinds)?);
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

    pub(crate) fn scene_paths(&self, kinds: &[&'static LayerKind], path: &str) -> Vec<ScenePath> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                let id = item.get(layer::ID_KEY).and_then(Value::as_str)?;
                Some(ScenePath {
                    id: id.to_owned(),
                    origin: self.origin,
                    points: layer_points(item, kinds)?,
                    at: format!("{path}.layers[{index}].points"),
                    ..ScenePath::default()
                })
            })
            .collect()
    }
}

pub(crate) fn terrain_kind() -> ComponentKindOf<TerrainScene> {
    ComponentKind::of::<TerrainScene>("terrain")
        .placement(Placement::Entities(ErrorCode::InvalidStreamed))
        .prepare(prepare_terrain)
        .files_in(terrain_files)
        .paths_in(terrain_paths)
        .schema_in(terrain_schema)
        .example(terrain_example)
}

pub(crate) fn prefetch_kind() -> ComponentKindOf<TerrainPrefetch> {
    ComponentKind::of::<TerrainPrefetch>("prefetch")
        .check(check_prefetch)
        .schema(prefetch_schema)
        .example(object_example)
}


fn prepare_terrain(
    scene: TerrainScene,
    cx: &SceneCx<'_>,
    path: &str,
) -> Result<TerrainScene, Error> {
    let kinds = listed(cx);
    let assets = match cx.scene() {
        Some(scene) => TerrainAssets {
            materials: scene
                .assets
                .materials
                .iter()
                .map(|(id, &handle)| (id.as_str(), handle))
                .collect(),
            meshes: scene
                .assets
                .meshes
                .iter()
                .map(|(id, &handle)| (id.as_str(), handle))
                .collect(),
        },
        None => TerrainAssets {
            materials: Vec::new(),
            meshes: Vec::new(),
        },
    };
    let built = scene
        .authored
        .prepare(path, (&LayerBuffers(cx), &kinds), &assets)?;
    Ok(TerrainScene {
        authored: scene.authored,
        built: Some(built),
    })
}

fn terrain_files(scene: &TerrainScene, cx: &SceneCx<'_>) -> Vec<String> {
    let kinds = listed(cx);
    scene
        .authored
        .layers
        .iter()
        .flat_map(|item| layer_files(item, &kinds))
        .collect()
}

fn terrain_paths(
    scene: &TerrainScene,
    cx: &SceneCx<'_>,
    path: &str,
) -> Result<Vec<ScenePath>, Error> {
    Ok(scene.authored.scene_paths(&listed(cx), path))
}

fn terrain_schema(cx: &SceneCx<'_>) -> Value {
    let mut schema =
        serde_json::to_value(schemars::schema_for!(SceneTerrain)).unwrap_or(Value::Null);
    let Some(fields) = schema.as_object_mut() else {
        return schema;
    };
    let mut definitions = match fields.remove("$defs") {
        Some(Value::Object(definitions)) => definitions,
        Some(_) | None => serde_json::Map::new(),
    };
    let layers = layer_list_schema(&listed(cx), &mut definitions);
    if let Some(Value::Object(properties)) = fields.get_mut("properties") {
        properties.insert("layers".to_owned(), layers);
    }
    fields.insert("$defs".to_owned(), Value::Object(definitions));
    schema
}

fn terrain_example() -> Value {
    json!({
        "origin": [-128.0, 0.0, -128.0],
        "tiles": [2, 2],
        "spacing": 1.0,
        "layers": [{ "type": "noise", "seed": 3, "height": 8.0, "wavelength": 120.0 }]
    })
}

const PREFETCH: &str = "prefetch";

fn check_prefetch(prefetch: &TerrainPrefetch, path: &str) -> Result<(), Error> {
    Setting::check(prefetch, PREFETCH, Rule::Any).map_err(|invalid| Error {
        code: ErrorCode::InvalidValue,
        path: format!(
            "{path}{}",
            invalid.field.strip_prefix(PREFETCH).unwrap_or_default()
        ),
        detail: Box::new(invalid.detail),
    })
}

fn prefetch_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(TerrainPrefetch)).unwrap_or(Value::Null)
}

fn object_example() -> Value {
    json!({})
}
