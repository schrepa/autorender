use std::collections::BTreeMap;
use std::sync::{PoisonError, RwLock};

use engine_core::bevy_ecs::prelude::Resource;
use engine_core::bevy_ecs::world::{EntityWorldMut, World};
use engine_core::glam::{Mat4, Vec3};
use engine_core::smoke::{SmokeEntry, SmokeFailure, SmokeResult, SmokeSession, component};
use engine_core::{
    BrickRegistry, Bricks, CORE_ROW, Error, Expected, Grounded, List, Mesh, Tint, missing_brick,
};
use engine_settings::{ErrorCode, Rule, Setting};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::instance::Lookups;
use crate::{SceneDocument, SceneFiles, SceneObject};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    Anywhere,
    Roots(ErrorCode),
    Entities(ErrorCode),
}

pub struct Reach<'a> {
    pub world: Mat4,
    pub position: Vec3,
    pub from: f32,
    pub meshes: (&'a [Mesh], &'a BTreeMap<&'a str, usize>),
}

impl Reach<'_> {
    pub fn mesh_size(&self, id: &str) -> f32 {
        let (meshes, indices) = self.meshes;
        indices.get(id).map_or(0.0, |&mesh| {
            let bounds = meshes[mesh].bounds();
            (bounds.max() - bounds.min()).length()
        })
    }

    pub fn of_box(&self, min: Vec3, max: Vec3) -> f32 {
        (0..8)
            .map(|corner| {
                let pick = |bit: u32, low: f32, high: f32| {
                    if corner & (1 << bit) == 0 { low } else { high }
                };
                let point = Vec3::new(
                    pick(0, min.x, max.x),
                    pick(1, min.y, max.y),
                    pick(2, min.z, max.z),
                );
                self.world.transform_point3(point).distance(self.position)
            })
            .fold(0.0, engine_core::math::max)
    }
}

pub struct ComponentContext<'a> {
    pub path: &'a str,
    pub document: &'a SceneDocument,
    pub files: Option<&'a SceneFiles>,
}

pub trait PreparedComponent: Send + Sync + 'static {
    fn spawn(&self, entity: &mut EntityWorldMut<'_>, lookups: &Lookups);

    fn reach(&self, _reach: &Reach<'_>, _extend: &mut dyn FnMut(f32)) {}
}

pub type Prepared = Result<Box<dyn PreparedComponent>, Error>;

pub struct ComponentKind {
    pub name: &'static str,
    pub shows: &'static str,
    pub placement: Placement,
    pub schema: fn() -> Value,
    pub example: fn() -> Value,
    pub validate: fn(&Value, &str) -> Result<(), Error>,
    pub validate_world: fn(&Value, Mat4, &str) -> Result<(), Error>,
    pub prepare: fn(&Value, &ComponentContext<'_>) -> Prepared,
    pub files: fn(&Value) -> Vec<String>,
    pub admit: fn(&Value, &World, &str) -> Result<(), Error>,
    pub paths: fn(&Value, &str) -> Result<Vec<ComponentPath>, Error>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComponentPath {
    pub id: String,
    pub origin: [f32; 3],
    pub points: Vec<[f32; 2]>,
    pub at: String,
}

impl std::fmt::Debug for ComponentKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ComponentKind")
            .field("name", &self.name)
            .field("shows", &self.shows)
            .field("placement", &self.placement)
            .finish_non_exhaustive()
    }
}

pub fn valid(_value: &Value, _path: &str) -> Result<(), Error> {
    Ok(())
}

pub fn valid_world(_value: &Value, _world: Mat4, _path: &str) -> Result<(), Error> {
    Ok(())
}

pub fn no_files(_value: &Value) -> Vec<String> {
    Vec::new()
}

pub fn admitted(_value: &Value, _world: &World, _path: &str) -> Result<(), Error> {
    Ok(())
}

pub fn no_paths(_value: &Value, _path: &str) -> Result<Vec<ComponentPath>, Error> {
    Ok(Vec::new())
}

#[macro_export]
macro_rules! component_schema {
    ($ty:ty) => {{
        #[cfg(feature = "schema")]
        fn component_schema() -> ::serde_json::Value {
            ::serde_json::to_value(::schemars::schema_for!($ty))
                .unwrap_or(::serde_json::Value::Null)
        }
        #[cfg(not(feature = "schema"))]
        fn component_schema() -> ::serde_json::Value {
            ::serde_json::Value::Null
        }
        component_schema
    }};
}

pub(crate) fn parsed<T: DeserializeOwned>(value: &Value, path: &str) -> Result<T, Error> {
    engine_core::parse_value_at(value.clone(), path)
}

pub(crate) fn ruled<T: DeserializeOwned + Setting>(
    value: &Value,
    path: &str,
    name: &'static str,
    rule: Rule,
) -> Result<T, Error> {
    let parsed: T = parsed(value, path)?;
    Setting::check(&parsed, name, rule).map_err(|invalid| Error {
        code: ErrorCode::InvalidValue,
        path: format!(
            "{path}{}",
            invalid.field.strip_prefix(name).unwrap_or_default()
        ),
        detail: Box::new(invalid.detail),
    })?;
    Ok(parsed)
}

pub(crate) fn object_example() -> Value {
    json!({})
}

const TINT_RULE: Rule = Rule::range(0.0, 1.0);

static TINT: ComponentKind = ComponentKind {
    name: "tint",
    shows: "Tint",
    placement: Placement::Anywhere,
    schema: tint_schema,
    example: tint_example,
    validate: validate_tint,
    validate_world: valid_world,
    prepare: prepare_tint,
    files: no_files,
    admit: admitted,
    paths: no_paths,
};

#[cfg(feature = "schema")]
fn tint_schema() -> Value {
    let mut schema = schemars::schema_for!([f32; 4]);
    engine_settings::stated(TINT_RULE)(&mut schema);
    serde_json::to_value(schema).unwrap_or(Value::Null)
}

#[cfg(not(feature = "schema"))]
fn tint_schema() -> Value {
    Value::Null
}

fn tint_example() -> Value {
    json!([1.0, 1.0, 1.0, 1.0])
}

fn validate_tint(value: &Value, path: &str) -> Result<(), Error> {
    ruled::<[f32; 4]>(value, path, TINT.name, TINT_RULE).map(drop)
}

fn prepare_tint(value: &Value, context: &ComponentContext<'_>) -> Prepared {
    Ok(Box::new(Tint(parsed(value, context.path)?)))
}

impl PreparedComponent for Tint {
    fn spawn(&self, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*self);
    }
}

pub(crate) static GROUNDED: ComponentKind = ComponentKind {
    name: "grounded",
    shows: "Grounded",
    placement: Placement::Roots(ErrorCode::InvalidGrounded),
    schema: crate::component_schema!(Grounded),
    example: object_example,
    validate: validate_grounded,
    validate_world: valid_world,
    prepare: prepare_grounded,
    files: no_files,
    admit: admitted,
    paths: no_paths,
};

fn validate_grounded(value: &Value, path: &str) -> Result<(), Error> {
    ruled::<Grounded>(value, path, GROUNDED.name, Rule::Any).map(drop)
}

fn prepare_grounded(value: &Value, context: &ComponentContext<'_>) -> Prepared {
    Ok(Box::new(parsed::<Grounded>(value, context.path)?))
}

impl PreparedComponent for Grounded {
    fn spawn(&self, entity: &mut EntityWorldMut<'_>, _: &Lookups) {
        entity.insert(*self);
    }
}

pub static CORE_COMPONENTS: &[&ComponentKind] = &[
    &crate::lighting::LIGHT,
    &crate::lighting::PARTICLES,
    &TINT,
    &GROUNDED,
    &crate::spline::SPLINE,
    &crate::probe::REFLECTION_PROBE,
    &crate::terrain::TERRAIN,
    &crate::terrain::PREFETCH,
];

static KINDS: RwLock<Vec<(&'static str, &'static ComponentKind)>> = RwLock::new(Vec::new());

pub fn register_components(brick: &'static str, kinds: &[&'static ComponentKind]) {
    let mut known = KINDS.write().unwrap_or_else(PoisonError::into_inner);
    for &kind in kinds {
        match known.iter_mut().find(|(_, held)| held.name == kind.name) {
            Some(held) => *held = (brick, kind),
            None => known.push((brick, kind)),
        }
    }
}

pub fn component_kind(name: &str) -> Option<(&'static str, &'static ComponentKind)> {
    if let Some(&kind) = CORE_COMPONENTS.iter().find(|kind| kind.name == name) {
        return Some((CORE_ROW, kind));
    }
    KINDS
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .copied()
        .find(|(_, kind)| kind.name == name)
}

pub fn component_kinds() -> Vec<(&'static str, &'static ComponentKind)> {
    let mut kinds: Vec<(&'static str, &'static ComponentKind)> = CORE_COMPONENTS
        .iter()
        .map(|&kind| (CORE_ROW, kind))
        .collect();
    kinds.extend(
        KINDS
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .copied(),
    );
    kinds
}

pub fn component_kinds_in(registry: &BrickRegistry) -> Vec<(&'static str, &'static ComponentKind)> {
    component_kinds()
        .into_iter()
        .filter(|&(brick, _)| brick == CORE_ROW || registry.has(brick))
        .collect()
}

pub fn component_names() -> Vec<&'static str> {
    component_kinds()
        .into_iter()
        .map(|(_, kind)| kind.name)
        .collect()
}

fn unknown_component(name: &str, path: &str) -> Error {
    Error::new(ErrorCode::UnknownField, path)
        .expected(Expected::Absent)
        .got(name)
        .variants(component_names())
}

impl SceneObject {
    pub fn component(&self, name: &str) -> Option<&Value> {
        self.components.get(name).filter(|value| !value.is_null())
    }

    fn authored(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.components
            .iter()
            .filter(|(_, value)| !value.is_null())
            .map(|(name, value)| (name.as_str(), value))
    }

    fn each_kind(
        &self,
        path: &str,
        mut visit: impl FnMut(&'static str, &'static ComponentKind, &Value, &str) -> Result<(), Error>,
    ) -> Result<(), Error> {
        for (name, value) in self.authored() {
            let at = format!("{path}.{name}");
            let (brick, kind) = component_kind(name).ok_or_else(|| unknown_component(name, &at))?;
            visit(brick, kind, value, &at)?;
        }
        Ok(())
    }

    fn each_known(
        &self,
        path: &str,
        mut visit: impl FnMut(&'static ComponentKind, &Value, &str) -> Result<(), Error>,
    ) -> Result<(), Error> {
        for (name, value) in self.authored() {
            if let Some((_, kind)) = component_kind(name) {
                visit(kind, value, &format!("{path}.{name}"))?;
            }
        }
        Ok(())
    }

    pub(crate) fn validate_components(&self, path: &str) -> Result<(), Error> {
        self.each_known(path, |kind, value, at| (kind.validate)(value, at))
    }

    pub(crate) fn validate_world_components(&self, world: Mat4, path: &str) -> Result<(), Error> {
        self.each_known(path, |kind, value, at| {
            (kind.validate_world)(value, world, at)
        })
    }

    pub(crate) fn check_placement(&self, path: &str) -> Result<(), Error> {
        if self.parent.is_none() {
            return Ok(());
        }
        self.each_known(path, |kind, _, at| match kind.placement {
            Placement::Roots(code) => Err(Error::new(code, at).expected(Expected::Absent)),
            Placement::Anywhere | Placement::Entities(_) => Ok(()),
        })
    }

    pub(crate) fn check_streamed(&self, path: &str) -> Result<(), Error> {
        self.each_known(path, |kind, _, at| match kind.placement {
            Placement::Entities(code) => Err(Error::new(code, at).expected(Expected::Absent)),
            Placement::Anywhere | Placement::Roots(_) => Ok(()),
        })
    }

    pub(crate) fn prepare_components(
        &self,
        path: &str,
        document: &SceneDocument,
        files: Option<&SceneFiles>,
    ) -> Result<Vec<Box<dyn PreparedComponent>>, Error> {
        let mut prepared = Vec::new();
        self.each_kind(path, |_, kind, value, at| {
            prepared.push((kind.prepare)(
                value,
                &ComponentContext {
                    path: at,
                    document,
                    files,
                },
            )?);
            Ok(())
        })?;
        Ok(prepared)
    }

    pub(crate) fn component_files(&self) -> impl Iterator<Item = String> {
        self.authored()
            .filter_map(|(name, value)| component_kind(name).map(|(_, kind)| (kind.files)(value)))
            .flatten()
    }

    pub(crate) fn admit_components(&self, world: &World, path: &str) -> Result<(), Error> {
        let offered = world.get_resource::<SceneComponents>();
        self.each_kind(path, |brick, kind, value, at| {
            if brick != CORE_ROW && offered.and_then(|offered| offered.get(kind.name)).is_none() {
                return Err(missing_brick(brick, at));
            }
            (kind.admit)(value, world, at)
        })
    }
}

#[cfg(feature = "schema")]
pub(crate) fn component_schemas(
    registry: &BrickRegistry,
    definitions: &mut serde_json::Map<String, Value>,
) -> serde_json::Map<String, Value> {
    component_kinds_in(registry)
        .into_iter()
        .map(|(_, kind)| (kind.name.to_owned(), component_property(kind, definitions)))
        .collect()
}

#[cfg(feature = "schema")]
fn component_property(
    kind: &ComponentKind,
    definitions: &mut serde_json::Map<String, Value>,
) -> Value {
    let mut schema = (kind.schema)();
    let Some(fields) = schema.as_object_mut() else {
        return Value::Bool(true);
    };
    fields.remove("$schema");
    if let Some(Value::Object(nested)) = fields.remove("$defs") {
        for (name, definition) in nested {
            definitions.entry(name).or_insert(definition);
        }
    }
    let listed = fields.get("type").and_then(Value::as_str) == Some("array");
    let title = fields
        .remove("title")
        .and_then(|title| title.as_str().map(str::to_owned));
    let shape = match title {
        Some(title) if !listed => {
            definitions.insert(title.clone(), schema);
            json!({ "$ref": format!("#/$defs/{title}") })
        }
        Some(_) | None => schema,
    };
    json!({ "anyOf": [shape, { "type": "null" }] })
}

#[derive(Resource, Default)]
pub struct SceneComponents {
    kinds: Vec<(&'static str, &'static ComponentKind)>,
}

impl SceneComponents {
    pub fn add(&mut self, brick: &'static str, kinds: &[&'static ComponentKind]) {
        for &kind in kinds {
            match self
                .kinds
                .iter_mut()
                .find(|(_, held)| held.name == kind.name)
            {
                Some(held) => *held = (brick, kind),
                None => self.kinds.push((brick, kind)),
            }
        }
    }

    pub fn get(&self, name: &str) -> Option<(&'static str, &'static ComponentKind)> {
        self.kinds
            .iter()
            .copied()
            .find(|(_, kind)| kind.name == name)
    }

    pub fn kinds(&self) -> &[(&'static str, &'static ComponentKind)] {
        &self.kinds
    }
}

pub struct SceneComponentsList;

impl List for SceneComponentsList {
    const NAME: &'static str = "scene_components";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "list_scene_components",
        brick: Self::NAME,
        gpu: false,
        run: scene_components,
    }];
}

pub trait AddSceneComponents {
    fn scene_components(&mut self, kinds: &[&'static ComponentKind]) -> &mut Self;
}

impl AddSceneComponents for Bricks<'_> {
    fn scene_components(&mut self, kinds: &[&'static ComponentKind]) -> &mut Self {
        let brick = self.name();
        register_components(brick, kinds);
        let world = self.world_mut();
        if !world.contains_resource::<SceneComponents>() {
            world.init_resource::<SceneComponents>();
        }
        world.resource_mut::<SceneComponents>().add(brick, kinds);
        self.list::<SceneComponentsList>()
    }
}

const EXAMPLE: &str = "example";

fn scene_components(session: &mut dyn SmokeSession) -> SmokeResult {
    let kinds = offered(session.app().world());
    #[cfg(feature = "schema")]
    listed(session.app().world(), &kinds)?;
    for kind in kinds {
        example_shows(session, kind)?;
    }
    Ok(())
}

fn offered(world: &World) -> Vec<&'static ComponentKind> {
    CORE_COMPONENTS
        .iter()
        .copied()
        .chain(
            world
                .get_resource::<SceneComponents>()
                .into_iter()
                .flat_map(|components| components.kinds().iter().map(|&(_, kind)| kind)),
        )
        .collect()
}

#[cfg(feature = "schema")]
fn listed(world: &World, kinds: &[&'static ComponentKind]) -> SmokeResult {
    let scene: Value = world
        .get_resource::<BrickRegistry>()
        .map(SceneDocument::schema_json)
        .and_then(|source| serde_json::from_str(&source).ok())
        .unwrap_or_default();
    for kind in kinds {
        let schema = !(kind.schema)().is_null();
        let property = !scene["$defs"]["SceneObject"]["properties"][kind.name].is_null();
        engine_core::smoke_eq!(
            json!({ "kind": kind.name, "schema": schema, "listed": property }),
            json!({ "kind": kind.name, "schema": true, "listed": true }),
            "kind_schema_listed"
        );
    }
    Ok(())
}

fn example_scene(entities: Value, streamed: Value) -> Value {
    json!({
        "id": "scene-components",
        "revision": 1,
        "materials": [{ "id": EXAMPLE, "color": [0.7, 0.6, 0.5, 1.0] }],
        "meshes": [{ "id": EXAMPLE, "source": { "kind": "cube" } }],
        "entities": entities,
        "streamed": streamed,
    })
}

fn holding(kind: &ComponentKind, mut entity: Value) -> Value {
    entity[kind.name] = (kind.example)();
    entity
}

fn example_shows(session: &mut dyn SmokeSession, kind: &'static ComponentKind) -> SmokeResult {
    let admitted = (kind.admit)(
        &(kind.example)(),
        session.app().world(),
        &format!("$.entities[0].{}", kind.name),
    );
    if admitted
        .as_ref()
        .is_err_and(|error| error.code == ErrorCode::MissingPlugin)
    {
        return Ok(());
    }
    engine_core::smoke_eq!(
        json!({ "kind": kind.name, "error": admitted.err() }),
        json!({ "kind": kind.name, "error": null }),
        "example_admitted"
    );
    placement_kept(kind)?;
    let scene = example_scene(json!([holding(kind, json!({ "id": EXAMPLE }))]), json!([]));
    let instance = SceneDocument::from_json(&scene.to_string())
        .and_then(|document| document.instantiate(session.app().world_mut()));
    let instance = match instance {
        Ok(instance) => instance,
        Err(error) => {
            return Err(SmokeFailure::new(
                "example_instantiates",
                &json!({ "kind": kind.name, "error": error }),
                &json!({ "kind": kind.name, "error": null }),
                concat!(file!(), ":", line!()),
            ));
        }
    };
    session.step(1);
    let inspector = kind
        .shows
        .split_once('.')
        .map_or(kind.shows, |(inspector, _)| inspector);
    let shown = component(session, EXAMPLE, inspector)?;
    instance.despawn(session.app().world_mut());
    engine_core::smoke_check!(
        !shown.is_null(),
        "example_shows",
        json!({ "kind": kind.name, "inspector": inspector, "shown": shown }),
        json!({ "kind": kind.name, "inspector": inspector })
    );
    Ok(())
}

fn placement_kept(kind: &'static ComponentKind) -> SmokeResult {
    let alone = example_scene(json!([holding(kind, json!({ "id": EXAMPLE }))]), json!([]));
    let validated = SceneDocument::from_json(&alone.to_string()).err();
    engine_core::smoke_eq!(
        json!({ "kind": kind.name, "error": validated }),
        json!({ "kind": kind.name, "error": null }),
        "example_validates"
    );
    let (scene, path, code) = match kind.placement {
        Placement::Anywhere => return Ok(()),
        Placement::Roots(code) => (
            example_scene(
                json!([
                    { "id": "holder" },
                    holding(kind, json!({ "id": EXAMPLE, "parent": "holder" }))
                ]),
                json!([]),
            ),
            format!("$.entities[1].{}", kind.name),
            code,
        ),
        Placement::Entities(code) => (
            example_scene(json!([]), json!([holding(kind, json!({ "id": EXAMPLE }))])),
            format!("$.streamed[0].{}", kind.name),
            code,
        ),
    };
    let refused = SceneDocument::from_json(&scene.to_string()).err();
    engine_core::smoke_eq!(
        json!({
            "kind": kind.name,
            "code": refused.as_ref().map(|error| error.code),
            "path": refused.as_ref().map(|error| error.path.as_str())
        }),
        json!({ "kind": kind.name, "code": code, "path": path }),
        "placement_refused"
    );
    Ok(())
}
