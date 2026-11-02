mod components;
mod poses;

use poses::presentation_view;

use engine_settings::{Error, ErrorCode, Expected};
use std::{collections::BTreeMap, io::Write};

use bevy_ecs::prelude::{Component, Entity, Resource, World};
use engine_core::{
    ActionId, ActionInput, AssetLibrary, Camera, Engine, FixedTime, FrameTime, GlobalTransforms,
    InterpolateTransform, JobExchange, Mesh3d, MoveInput, Parent, PresentationMode,
    PresentationState, Projection, TextureHandle, Transform,
};
use engine_registry::{BrickRecord, OperationScene, Registry};
use engine_scene::{SceneEntity, SceneInstance, SceneMaterial, TextureOrigin};
use serde::Serialize;
use serde_json::{Value, json};

use crate::explain::source_line;
use crate::protocol::{
    DescribeRequest, Detail, Envelope, InspectRequest, Query, Verb, parse_named,
};
use crate::reply::{Observation, Reply, SceneObservation};
use crate::results::{
    AssetsResult, AuthoredEntity, DescribeResult, DescribedBrick, DescribedEntry, DescribedType,
    DistributionResult, EntityComponents, EntityRow, InspectResult, MissingAsset, StatusLimits,
    StatusResult, TextureAsset, TextureState, TypeKind, result,
};
use crate::{
    BrowserControl, MAX_INSPECTOR_BYTES, MAX_INSPECTORS, MAX_PAGE_SIZE, MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES, MAX_SCAN_ENTITIES, PROTOCOL_VERSION,
    views::{
        CameraView, GlobalTransformView, HierarchyView, InputView, InterpolateTransformView,
        LodLevelView, Mesh3dView, MeshBoundsView, MeshRenderView, MeshView, PresentationModeView,
        PresentationStateView, TextureView, TransformView,
    },
};

const DESCRIBED: &[(&str, TypeKind)] = &[
    ("SceneDocument", TypeKind::Authoring),
    ("FrameStats", TypeKind::Observation),
    ("ErrorCodes", TypeKind::Catalog),
];

pub const TOOLING_TYPES: &[&str] = &["ProfileCapture", "GameControls", "PageParameters"];

pub fn reserved_names() -> impl Iterator<Item = &'static str> {
    DESCRIBED
        .iter()
        .map(|&(name, _)| name)
        .chain(Verb::names())
        .chain(TOOLING_TYPES.iter().copied())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum InspectorKind {
    Component,
    Resource,
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct InspectorDescriptor {
    pub name: String,
    pub schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
}

impl InspectorDescriptor {
    pub fn new(name: impl Into<String>, schema: Value) -> Self {
        Self {
            name: name.into(),
            schema,
            default: None,
        }
    }

    pub fn generated(schemas: &Value, name: &str) -> Self {
        let schema = schemas
            .get(name)
            .cloned()
            .unwrap_or_else(|| panic!("schema_generated {name}"));
        Self::new(name, schema)
    }

    #[must_use]
    pub fn with_default(mut self, value: Value) -> Self {
        self.default = Some(value);
        self
    }
}

type ReadInspector =
    Box<dyn Fn(&World, Option<Entity>, &Context<'_>) -> Option<Value> + Send + Sync>;

pub trait RendererFacts {
    fn texture_resident(&self, texture: TextureHandle) -> bool;
}

#[derive(Clone, Copy, Default)]
pub struct Context<'a> {
    pub scene: Option<&'a SceneInstance>,
    pub renderer: Option<&'a dyn RendererFacts>,
}

pub enum ToolCall<'a> {
    Query(&'a str, Value),
    Describe(DescribeRequest),
    Inspect(InspectRequest),
}

impl<'a> ToolCall<'a> {
    pub fn of(envelope: &'a Envelope) -> Result<Self, Error> {
        match envelope.verb {
            Verb::Query => Ok(Self::Query(envelope.name()?, envelope.args()?)),
            Verb::Describe => envelope.body().map(Self::Describe),
            Verb::Inspect => envelope.body().map(Self::Inspect),
            verb @ (Verb::Control
            | Verb::Capture
            | Verb::Profile
            | Verb::Explain
            | Verb::Watch
            | Verb::Check) => Err(Error::new(ErrorCode::UnsupportedCommand, "$.verb")
                .got(verb.name())
                .variants([Verb::Describe, Verb::Inspect, Verb::Query].map(Verb::name))),
        }
    }

    pub fn status() -> Self {
        Self::Query("status", Value::Object(serde_json::Map::new()))
    }

    pub fn inspect(entity: &str, component: &str) -> Self {
        Self::Inspect(InspectRequest {
            entity: Some(entity.to_owned()),
            parent: None,
            component: Some(component.to_owned()),
            offset: 0,
            limit: crate::protocol::default_limit(),
        })
    }
}

fn observe(engine: &Engine, scene: Option<&SceneInstance>) -> Observation {
    let world = engine.app.world();
    Observation {
        frame: world
            .get_resource::<FrameTime>()
            .map(|time| time.frame.to_string()),
        tick: world
            .get_resource::<FixedTime>()
            .map(|time| time.tick.to_string()),
        scene: scene.map(|scene| SceneObservation {
            id: scene.id().to_owned(),
            revision: scene.revision().to_string(),
        }),
        presentation: presentation_view(engine.presentation()),
    }
}

fn replied(observation: Observation, outcome: Result<Value, Error>) -> Reply {
    let (result, error) = match outcome {
        Ok(result) => (Some(result), None),
        Err(error) => (None, Some(error)),
    };
    Reply {
        version: Some(PROTOCOL_VERSION),
        ok: error.is_none(),
        observation: Some(observation),
        result,
        error,
    }
}

fn too_large(reply: Reply) -> Reply {
    Reply {
        ok: false,
        result: None,
        error: Some(
            Error::new(ErrorCode::ResponseTooLarge, "$").at_most(MAX_RESPONSE_BYTES as u64),
        ),
        ..reply
    }
}
type HasInspector = Box<dyn Fn(&World, Option<Entity>) -> bool + Send + Sync>;

pub(crate) struct Inspector {
    pub(crate) descriptor: InspectorDescriptor,
    pub(crate) kind: InspectorKind,
    pub(crate) read: ReadInspector,
    pub(crate) has: HasInspector,
    pub(crate) brick: Option<&'static str>,
}

type BrickInspectors = BTreeMap<String, Inspector>;

#[derive(Clone, Copy)]
struct InspectSelector<'a> {
    entity: Option<&'a str>,
    parent: Option<&'a str>,
    component: Option<&'a str>,
    offset: usize,
    limit: usize,
}

pub struct DevTools {
    pub(crate) inspectors: BTreeMap<String, Inspector>,
    builtins: Option<usize>,
    pub(crate) schemas: Value,
    pub(crate) standard: crate::bricks::StandardBricks,
}

impl DevTools {
    pub(crate) fn bare() -> Self {
        Self {
            inspectors: BTreeMap::new(),
            builtins: None,
            schemas: Value::Null,
            standard: crate::bricks::StandardBricks::default(),
        }
    }

    pub fn with_schemas(schemas: &Value) -> Self {
        let mut tools = Self::bare();
        tools.register_poses(schemas);
        tools.register_components(schemas);
        tools.builtins = Some(tools.inspectors.len());
        tools.schemas = schemas.clone();
        tools.standard = crate::bricks::StandardBricks::of(schemas);
        tools
    }

    fn all_inspectors<'a>(
        &'a self,
        bricks: &'a BrickInspectors,
    ) -> impl Iterator<Item = &'a Inspector> + 'a {
        let mut all: BTreeMap<&str, &Inspector> = BTreeMap::new();
        for (name, inspector) in bricks.iter().chain(&self.inspectors) {
            all.insert(name.as_str(), inspector);
        }
        all.into_values()
    }
}

impl Default for DevTools {
    fn default() -> Self {
        let schemas: Value = serde_json::from_str(include_str!("../generated/inspectors.json"))
            .expect("generated_schema_json");
        Self::with_schemas(&schemas)
    }
}

fn scene_schema(definition: Option<&str>) -> Result<Value, Error> {
    let mut schema: Value = serde_json::from_str(include_str!("../generated/scene.schema.json"))
        .expect("generated_schema_json");
    let definitions = schema
        .as_object_mut()
        .and_then(|schema| schema.remove("$defs"))
        .unwrap_or_default();
    match definition {
        Some(name) => definitions.get(name).cloned().ok_or_else(|| {
            Error::new(ErrorCode::UnknownDefinition, "$.definition")
                .got(name)
                .variants(
                    definitions
                        .as_object()
                        .into_iter()
                        .flat_map(|definitions| definitions.keys().cloned()),
                )
        }),
        None => {
            let names: Vec<&String> = definitions
                .as_object()
                .map(|definitions| definitions.keys().collect())
                .unwrap_or_default();
            schema["definitions"] = json!(names);
            Ok(schema)
        }
    }
}

impl DevTools {
    pub(crate) fn component_views(
        &self,
        world: &World,
        context: Context<'_>,
    ) -> BTreeMap<String, Vec<String>> {
        let entities: Vec<Entity> = world.iter_entities().map(|entity| entity.id()).collect();
        let bricks = self.brick_inspectors(world);
        self.all_inspectors(&bricks)
            .filter(|inspector| inspector.kind == InspectorKind::Component)
            .map(|inspector| {
                let views = entities
                    .iter()
                    .filter_map(|&entity| (inspector.read)(world, Some(entity), &context))
                    .map(|view| view.to_string())
                    .collect();
                (inspector.descriptor.name.clone(), views)
            })
            .collect()
    }

    pub fn register_component<T: Component>(
        &mut self,
        descriptor: InspectorDescriptor,
        inspect: impl Fn(&T) -> Value + Send + Sync + 'static,
    ) -> Result<(), Error> {
        self.insert(Inspector {
            descriptor,
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<T>(entity).is_some())
            }),
            read: Box::new(move |world, entity, _context| world.get::<T>(entity?).map(&inspect)),
            brick: None,
        })
    }

    pub fn register_resource<T: Resource>(
        &mut self,
        descriptor: InspectorDescriptor,
        inspect: impl Fn(&T) -> Value + Send + Sync + 'static,
    ) -> Result<(), Error> {
        self.insert(Inspector {
            descriptor,
            kind: InspectorKind::Resource,
            has: Box::new(|world, _| world.get_resource::<T>().is_some()),
            read: Box::new(move |world, _, _context| world.get_resource::<T>().map(&inspect)),
            brick: None,
        })
    }

    pub(crate) fn insert(&mut self, inspector: Inspector) -> Result<(), Error> {
        let name = &inspector.descriptor.name;
        validate_name(name, "$.name")?;
        if reserved_names().any(|reserved| reserved == name) {
            return Err(Error::new(ErrorCode::ReservedInspector, "$.name")
                .expected(Expected::Unique)
                .got(name));
        }
        if self.inspectors.contains_key(name) {
            return Err(Error::new(ErrorCode::DuplicateInspector, "$.name")
                .expected(Expected::Unique)
                .got(name));
        }
        if let Some(builtins) = self.builtins
            && self.inspectors.len() >= builtins + MAX_INSPECTORS
        {
            return Err(Error::new(ErrorCode::InspectorLimit, "$.name")
                .at_most(MAX_INSPECTORS as u64)
                .got(name));
        }
        if encode(&inspector.descriptor, MAX_INSPECTOR_BYTES).is_err() {
            return Err(Error::new(ErrorCode::InspectorTooLarge, "$.schema")
                .at_most(MAX_INSPECTOR_BYTES as u64));
        }
        self.inspectors.insert(name.clone(), inspector);
        Ok(())
    }

    pub fn execute(&self, engine: &mut Engine, context: Context<'_>, source: &str) -> String {
        let observation = observe(engine, context.scene);
        let outcome = Envelope::parse(source).and_then(|envelope| {
            let call = ToolCall::of(&envelope)?;
            self.run(engine, context, call)
        });
        let reply = replied(observation, outcome);
        encode(&reply, MAX_RESPONSE_BYTES).unwrap_or_else(|_| {
            serde_json::to_string(&too_large(reply)).expect("reply_serializes")
        })
    }

    pub fn call(&self, engine: &mut Engine, context: Context<'_>, call: ToolCall<'_>) -> Value {
        let observation = observe(engine, context.scene);
        let reply = replied(observation, self.run(engine, context, call));
        let reply = match encode(&reply, MAX_RESPONSE_BYTES) {
            Ok(_) => reply,
            Err(_) => too_large(reply),
        };
        serde_json::to_value(reply).expect("reply_serializes")
    }

    fn run(
        &self,
        engine: &mut Engine,
        context: Context<'_>,
        call: ToolCall<'_>,
    ) -> Result<Value, Error> {
        let scene = context.scene;
        let wrong_world =
            |engine: &Engine| scene.is_some_and(|scene| !scene.belongs_to(engine.app.world()));
        let refused = || Error::new(ErrorCode::WrongWorld, "$.scene");
        match call {
            ToolCall::Query(name, args) => {
                if let Some(name) = self.query_operation(engine.app.world(), name)? {
                    if wrong_world(engine) {
                        return Err(refused());
                    }
                    return crate::bricks::run(engine.app.world_mut(), name, args);
                }
                if !Query::NAMES.contains(&name) {
                    return Err(Error::new(ErrorCode::UnknownVariant, "$.name")
                        .got(name)
                        .variants(crate::bricks::operation_names(
                            engine.app.world(),
                            Verb::Query,
                        )));
                }
                let query: Query = parse_named(name, args)?;
                if wrong_world(engine) {
                    return Err(refused());
                }
                self.query(engine, context, query)
            }
            ToolCall::Describe(request) => {
                if wrong_world(engine) {
                    return Err(refused());
                }
                self.describe(engine.app.world(), &request)
                    .map(result::describe)
            }
            ToolCall::Inspect(request) => {
                if wrong_world(engine) {
                    return Err(refused());
                }
                self.inspect_request(engine, context, &request)
                    .map(result::inspect)
            }
        }
    }

    fn query(
        &self,
        engine: &mut Engine,
        context: Context<'_>,
        query: Query,
    ) -> Result<Value, Error> {
        let scene = context.scene;
        if matches!(query, Query::Digest {}) {
            engine.refresh_transforms();
        }
        let engine = &*engine;
        let world = engine.app.world();
        match query {
            Query::Status {} => Ok(result::status(self.status(engine, scene))),
            Query::Assets {} => assets(world, context).map(result::assets),
            Query::Digest {} => Ok(result::digest(self.digest(
                world,
                Context {
                    scene,
                    renderer: None,
                },
            ))),
            Query::Distribution { values } => {
                if values.len() > crate::bounds::MAX_VALUES {
                    return Err(Error::new(ErrorCode::OutOfRange, "$.args.values")
                        .expected(Expected::Array)
                        .at_most(crate::bounds::MAX_VALUES as u64)
                        .got(values.len()));
                }
                Ok(result::distribution(DistributionResult {
                    distribution: engine_core::Distribution::of(&values),
                }))
            }
        }
    }

    pub(crate) fn inspect_request(
        &self,
        engine: &mut Engine,
        context: Context<'_>,
        request: &InspectRequest,
    ) -> Result<InspectResult, Error> {
        let selector = InspectSelector {
            entity: request.entity.as_deref(),
            parent: request.parent.as_deref(),
            component: request.component.as_deref(),
            offset: request.offset,
            limit: request.limit,
        };
        let settled = request.component.as_deref().filter(|key| {
            engine
                .app
                .world()
                .get_resource::<Registry>()
                .is_some_and(|registry| registry.stats_section(key).is_some())
        });
        if let Some(key) = settled {
            self.inspect(engine.app.world(), context, selector)?;
            engine_core::settle_stats(engine.app.world_mut(), key);
        }
        engine.refresh_transforms();
        let mut inspected = self.inspect(engine.app.world(), context, selector)?;
        drawn_levels(engine.app.world_mut(), &mut inspected);
        Ok(inspected)
    }

    fn status(&self, engine: &Engine, scene: Option<&SceneInstance>) -> StatusResult {
        let world = engine.app.world();
        let bricks = self.brick_inspectors(world);
        let registry = world.get_resource::<Registry>();
        StatusResult {
            world_entities: world.entities().len(),
            authored_entities: scene.map(SceneInstance::entity_count),
            inspectors: self.all_inspectors(&bricks).count(),
            culling_enabled: frustum_culling(world),
            features: registry.map(|registry| registry.names().collect()),
            presentation: presentation_view(engine.presentation()),
            collision: registry
                .and_then(|registry| registry.stats_section("collision_stats"))
                .map(|stats| stats.read(world)),
            fixed_delta_seconds: world
                .get_resource::<FixedTime>()
                .map(|time| time.delta_seconds),
            limits: StatusLimits {
                request_bytes: MAX_REQUEST_BYTES,
                response_bytes: MAX_RESPONSE_BYTES,
                page_size: MAX_PAGE_SIZE,
                scan_entities: MAX_SCAN_ENTITIES,
            },
        }
    }

    fn describe(&self, world: &World, request: &DescribeRequest) -> Result<DescribeResult, Error> {
        let bricks = self.brick_inspectors(world);
        let registry = world.get_resource::<Registry>();
        let full = request.detail == Detail::Full;
        let Some(name) = request.name.as_deref() else {
            let mut types: Vec<DescribedType> = self
                .all_inspectors(&bricks)
                .map(|inspector| DescribedType {
                    name: inspector.descriptor.name.clone(),
                    kind: inspector.kind.into(),
                    brick: Some(inspector.brick),
                })
                .collect();
            types.extend(DESCRIBED.iter().map(|&(name, kind)| DescribedType {
                name: name.to_owned(),
                kind,
                brick: None,
            }));
            return Ok(DescribeResult::Listing {
                verbs: Verb::names().collect(),
                bricks: registry
                    .into_iter()
                    .flat_map(Registry::bricks)
                    .map(|brick| described_brick(brick, full))
                    .collect(),
                types,
            });
        };
        validate_name(name, "$.name")?;
        if let Some(verb) = Verb::named(name) {
            return Ok(DescribeResult::Verb {
                name: name.to_owned(),
                kind: TypeKind::Verb,
                schema: self.schema_named(verb.name()),
                control_state: (verb == Verb::Control).then(|| control_state(world)),
            });
        }
        match name {
            "SceneDocument" => {
                return Ok(DescribeResult::Authoring {
                    name: name.to_owned(),
                    kind: TypeKind::Authoring,
                    schema: scene_schema(request.definition.as_deref())?,
                });
            }
            "ErrorCodes" => {
                return Ok(DescribeResult::Catalog {
                    name: name.to_owned(),
                    kind: TypeKind::Catalog,
                    codes: ErrorCode::ALL.iter().map(|&(_, code)| code).collect(),
                });
            }
            "FrameStats" => {
                return Ok(DescribeResult::Observation {
                    name: name.to_owned(),
                    kind: TypeKind::Observation,
                    schema: frame_stats_schema(),
                });
            }
            _ => {}
        }
        if let Some(inspector) = self.inspectors.get(name).or_else(|| bricks.get(name)) {
            return Ok(DescribeResult::Inspector {
                kind: inspector.kind.into(),
                descriptor: inspector.descriptor.clone(),
                brick: inspector.brick,
                at: registry.and_then(|registry| inspector_at(registry, name)),
            });
        }
        if let Some(described) = self.described_operation(world, name) {
            return Ok(described);
        }
        if let Some(brick) = registry.and_then(|registry| registry.brick(name)) {
            return Ok(DescribeResult::Brick {
                kind: TypeKind::Brick,
                brick: described_brick(brick, true),
            });
        }
        if let Some(stats) = registry.and_then(|registry| registry.stats_section(name)) {
            return Ok(DescribeResult::Stats {
                name: name.to_owned(),
                kind: TypeKind::Stats,
                brick: stats.brick,
                at: source_line(stats.at),
                schema: stats_schema(name),
            });
        }
        if let Some(brick) = self
            .standard
            .of_inspector(name)
            .or_else(|| self.standard.of_operation(name))
            && !crate::bricks::added(world, brick)
        {
            return Err(crate::bricks::missing_brick(brick, "$.name"));
        }
        Err(Error::new(ErrorCode::UnknownInspector, "$.name")
            .got(name)
            .variants(self.describable(world, &bricks)))
    }

    fn described_operation(&self, world: &World, name: &str) -> Option<DescribeResult> {
        let registered = world
            .get_resource::<Registry>()
            .and_then(|registry| registry.operation(name));
        let verb = match registered {
            Some(operation) if operation.control => Verb::Control,
            Some(_) => Verb::Query,
            None if Query::NAMES.contains(&name) => Verb::Query,
            None if BrowserControl::NAMES.contains(&name) => Verb::Control,
            None => return None,
        };
        let shows = crate::protocol::CONTROL_STATE
            .iter()
            .find(|(control, _)| *control == name)
            .map(|&(_, shows)| shows);
        Some(DescribeResult::Operation {
            name: name.to_owned(),
            kind: TypeKind::Operation,
            verb: verb.name(),
            brick: registered.map(|operation| operation.brick),
            at: registered.map(|operation| source_line(operation.at)),
            shows: shows.filter(|shows| !shows.is_empty()),
            args: operation_args(&self.schema_named(verb.name()), name),
        })
    }

    fn describable(&self, world: &World, bricks: &BrickInspectors) -> Vec<String> {
        let registry = world.get_resource::<Registry>();
        let mut names: Vec<String> = Verb::names()
            .chain(DESCRIBED.iter().map(|&(name, _)| name))
            .map(str::to_owned)
            .chain(
                self.all_inspectors(bricks)
                    .map(|inspector| inspector.descriptor.name.clone()),
            )
            .chain(
                [Verb::Query, Verb::Control]
                    .into_iter()
                    .flat_map(|verb| crate::bricks::operation_names(world, verb))
                    .map(str::to_owned),
            )
            .chain(
                registry
                    .into_iter()
                    .flat_map(|registry| {
                        registry
                            .names()
                            .chain(registry.stats().map(|stats| stats.key))
                    })
                    .map(str::to_owned),
            )
            .collect();
        names.sort();
        names.dedup();
        names
    }

    fn inspector<'a>(
        &'a self,
        world: &World,
        bricks: &'a BrickInspectors,
        name: &str,
    ) -> Result<&'a Inspector, Error> {
        validate_name(name, "$.component")?;
        if let Some(inspector) = self.inspectors.get(name).or_else(|| bricks.get(name)) {
            return Ok(inspector);
        }
        if let Some(brick) = self.standard.of_inspector(name)
            && !crate::bricks::added(world, brick)
        {
            return Err(crate::bricks::missing_brick(brick, "$.component"));
        }
        let known: Vec<String> = self
            .all_inspectors(bricks)
            .map(|inspector| inspector.descriptor.name.clone())
            .collect();
        Err(Error::new(ErrorCode::UnknownInspector, "$.component")
            .got(name)
            .variants(known))
    }

    fn inspect(
        &self,
        world: &World,
        context: Context<'_>,
        selector: InspectSelector<'_>,
    ) -> Result<InspectResult, Error> {
        let scene = context.scene;
        let InspectSelector {
            entity,
            parent,
            component,
            offset,
            limit,
        } = selector;
        if entity.is_some() && parent.is_some() {
            return Err(
                Error::new(ErrorCode::InvalidSelector, "$.parent").expected(Expected::Absent)
            );
        }
        if !(1..=MAX_PAGE_SIZE).contains(&limit) {
            return Err(Error::new(ErrorCode::InvalidPage, "$.limit")
                .range(1.0, MAX_PAGE_SIZE as f64)
                .got(limit));
        }
        if offset > MAX_SCAN_ENTITIES {
            return Err(Error::new(ErrorCode::InvalidPage, "$.offset")
                .at_most(MAX_SCAN_ENTITIES as u64)
                .got(offset));
        }
        let bricks = self.brick_inspectors(world);
        let inspector = component
            .map(|name| self.inspector(world, &bricks, name))
            .transpose()?;
        if let Some(inspector) =
            inspector.filter(|inspector| inspector.kind == InspectorKind::Resource)
        {
            if entity.is_some() || parent.is_some() || offset != 0 {
                return Err(Error::new(ErrorCode::InvalidSelector, "$.entity")
                    .expected(Expected::Absent)
                    .got(&inspector.descriptor.name));
            }
            let value = read_value(inspector, world, None, &context)?;
            return Ok(InspectResult::Resource {
                resource: inspector.descriptor.name.clone(),
                value,
            });
        }
        let scene = scene
            .map(|scene| scene.current_scene(world))
            .or_else(|| world.get_resource::<OperationScene>().cloned())
            .ok_or_else(|| Error::new(ErrorCode::NoScene, "$.entity").expected(Expected::Entity))?;
        if scene.entity_count() > MAX_SCAN_ENTITIES {
            return Err(Error::new(ErrorCode::ScanLimit, "$.scene")
                .at_most(MAX_SCAN_ENTITIES as u64)
                .got(scene.entity_count()));
        }
        if let Some(id) = entity {
            validate_name(id, "$.entity")?;
            if offset != 0 {
                return Err(Error::new(ErrorCode::InvalidSelector, "$.offset")
                    .range(0.0, 0.0)
                    .got(offset));
            }
            let entity = scene.entity(world, id).ok_or_else(|| {
                Error::new(ErrorCode::UnknownEntity, "$.entity")
                    .expected(Expected::Entity)
                    .got(id)
            })?;
            if world.get_entity(entity).is_err() {
                return Err(Error::new(ErrorCode::EntityRemoved, "$.entity")
                    .expected(Expected::Live)
                    .got(id));
            }
            return Ok(InspectResult::Entity {
                entity: self.entity_row(world, (context, &bricks), (id, entity), inspector)?,
            });
        }
        let parent = parent
            .map(|id| {
                validate_name(id, "$.parent")?;
                let entity = scene.entity(world, id).ok_or_else(|| {
                    Error::new(ErrorCode::UnknownEntity, "$.parent")
                        .expected(Expected::Entity)
                        .got(id)
                })?;
                if world.get_entity(entity).is_err() {
                    return Err(Error::new(ErrorCode::EntityRemoved, "$.parent")
                        .expected(Expected::Live)
                        .got(id));
                }
                Ok(entity)
            })
            .transpose()?;
        let mut rows = Vec::with_capacity(limit);
        let mut total = 0;
        for (id, &entity) in scene.entities.iter() {
            if world.get_entity(entity).is_err()
                || parent.is_some_and(|parent| {
                    world.get::<Parent>(entity).map(Parent::parent) != Some(parent)
                })
                || inspector.is_some_and(|inspector| !(inspector.has)(world, Some(entity)))
            {
                continue;
            }
            if total >= offset && rows.len() < limit {
                rows.push(self.entity_row(
                    world,
                    (context, &bricks),
                    (id.as_str(), entity),
                    inspector,
                )?);
            }
            total += 1;
        }
        let next = offset.saturating_add(rows.len());
        Ok(InspectResult::Page {
            entities: rows,
            offset,
            limit,
            total,
            next_offset: (next < total).then_some(next),
        })
    }

    fn entity_row(
        &self,
        world: &World,
        (context, bricks): (Context<'_>, &BrickInspectors),
        (id, entity): (&str, Entity),
        inspector: Option<&Inspector>,
    ) -> Result<EntityRow, Error> {
        let metadata = world.get::<SceneEntity>(entity);
        let components = if let Some(inspector) = inspector {
            let value = read_value(inspector, world, Some(entity), &context)?;
            EntityComponents::Values(
                [(inspector.descriptor.name.clone(), value)]
                    .into_iter()
                    .collect(),
            )
        } else {
            EntityComponents::Names(
                self.all_inspectors(bricks)
                    .filter(|inspector| {
                        inspector.kind == InspectorKind::Component
                            && (inspector.has)(world, Some(entity))
                    })
                    .map(|inspector| inspector.descriptor.name.clone())
                    .collect(),
            )
        };
        Ok(EntityRow {
            id: id.to_owned(),
            name: metadata.map_or_else(String::new, |metadata| metadata.name.clone()),
            tags: metadata.map(|metadata| metadata.tags.clone()),
            components,
        })
    }
}

fn read_value(
    inspector: &Inspector,
    world: &World,
    entity: Option<Entity>,
    context: &Context<'_>,
) -> Result<Value, Error> {
    let value = (inspector.read)(world, entity, context).ok_or_else(|| {
        Error::new(ErrorCode::MissingComponent, "$.component")
            .expected(Expected::Present)
            .got(&inspector.descriptor.name)
    })?;
    encode(&value, MAX_INSPECTOR_BYTES).map_err(|_| {
        Error::new(ErrorCode::InspectorTooLarge, "$.component")
            .at_most(MAX_INSPECTOR_BYTES as u64)
            .got(&inspector.descriptor.name)
    })?;
    Ok(value)
}

fn drawn_levels(world: &mut World, inspected: &mut InspectResult) {
    let rows = match inspected {
        InspectResult::Entity { entity } => std::slice::from_mut(entity),
        InspectResult::Page { entities, .. } => entities.as_mut_slice(),
        InspectResult::Resource { .. } => return,
    };
    for row in rows {
        let EntityComponents::Values(components) = &mut row.components else {
            continue;
        };
        if let Some(view) = components.get_mut("Mesh3d") {
            let drawn = crate::bricks::mesh_drawn(world, &row.id);
            view["drawn_level"] = drawn.get("drawn_level").cloned().unwrap_or(Value::Null);
        }
    }
}

pub(crate) fn validate_name(name: &str, path: &str) -> Result<(), Error> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
    {
        return Err(Error::new(ErrorCode::InvalidName, path)
            .expected(Expected::Id)
            .at_most(64)
            .got(name));
    }
    Ok(())
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
}

impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::other("byte_budget"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(crate) fn encode(value: &impl Serialize, limit: usize) -> Result<String, serde_json::Error> {
    let mut output = BoundedWriter {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut output, value)?;
    Ok(String::from_utf8(output.bytes).expect("json_utf8"))
}

pub(crate) fn authored_entity(world: &World, scene: &SceneInstance, id: &str) -> Option<Entity> {
    scene.authored_entity(world, id)
}

pub fn frustum_culling(world: &World) -> bool {
    world
        .get_resource::<Registry>()
        .and_then(|registry| registry.section("culling"))
        .is_some_and(|section| section.current(world)["frustum"] == json!(true))
}

fn assets(world: &World, context: Context<'_>) -> Result<AssetsResult, Error> {
    let assets = world.get_resource::<AssetLibrary>().ok_or_else(|| {
        Error::new(ErrorCode::MissingResource, "$.assets")
            .expected(Expected::Present)
            .got("AssetLibrary")
    })?;
    let texture_assets = context
        .scene
        .into_iter()
        .flat_map(SceneInstance::textures)
        .map(|(id, handle)| {
            let texture = assets.texture(handle);
            TextureAsset {
                id: id.to_owned(),
                state: if texture.is_some() {
                    TextureState::CpuReady
                } else {
                    TextureState::Missing
                },
                width: texture.map(|texture| texture.width()),
                height: texture.map(|texture| texture.height()),
                bytes: texture.map(|texture| texture.byte_len()),
                gpu_resident: context
                    .renderer
                    .map(|renderer| renderer.texture_resident(handle)),
            }
        })
        .collect();
    let (missing, missing_total) = missing_assets(world, assets);
    Ok(AssetsResult {
        meshes: assets.mesh_count(),
        materials: assets.material_count(),
        textures: assets.texture_count(),
        texture_cpu_bytes: assets.texture_bytes(),
        texture_assets,
        resident_cpu_bytes: assets.resident_bytes(),
        readiness: "synchronous",
        missing,
        missing_total,
    })
}

pub(crate) fn missing_of(assets: &AssetLibrary, mesh3d: &Mesh3d) -> Option<&'static str> {
    match assets.material(mesh3d.material) {
        None => Some("material"),
        Some(_) if assets.mesh(mesh3d.mesh).is_none() => Some("mesh"),
        Some(material) => material
            .textures()
            .any(|texture| assets.texture(texture).is_none())
            .then_some("texture"),
    }
}

fn missing_assets(world: &World, assets: &AssetLibrary) -> (Vec<MissingAsset>, usize) {
    let transforms = world.get_resource::<GlobalTransforms>();
    let position = |entity: Entity| {
        transforms
            .and_then(|transforms| transforms.get(entity))
            .and_then(|global| global.matrix())
            .map(|matrix| matrix.w_axis.truncate().to_array())
    };
    let mut found: Vec<(Entity, &'static str, &'static str)> = Vec::new();
    if let Some(mut meshes) = world.try_query::<(Entity, &Mesh3d)>() {
        for (entity, mesh3d) in meshes.iter(world) {
            if let Some(missing) = missing_of(assets, mesh3d) {
                found.push((entity, "mesh", missing));
            }
        }
    }
    found.sort_unstable_by_key(|(entity, ..)| *entity);
    let total = found.len();
    let listed = found
        .into_iter()
        .take(crate::protocol::bounds::MAX_PAGE_SIZE)
        .map(|(entity, draws, missing)| MissingAsset {
            entity: AuthoredEntity::of(world, entity),
            draws,
            missing,
            position: position(entity),
        })
        .collect();
    (listed, total)
}

fn control_state(world: &World) -> serde_json::Map<String, Value> {
    let mut state: serde_json::Map<String, Value> = crate::protocol::CONTROL_STATE
        .iter()
        .map(|&(op, shows)| (op.to_owned(), json!(shows)))
        .collect();
    for operation in world
        .get_resource::<Registry>()
        .into_iter()
        .flat_map(Registry::operations)
        .filter(|operation| operation.control)
    {
        state.insert(operation.name.to_owned(), json!(""));
    }
    state
}

fn described_brick(brick: &BrickRecord, full: bool) -> DescribedBrick {
    let at = |at: &'static std::panic::Location<'static>| full.then(|| source_line(at));
    DescribedBrick {
        name: brick.package,
        crate_name: brick.package,
        fills: brick
            .fills()
            .iter()
            .map(|filled| filled.place.name())
            .collect(),
        joins: brick
            .joins()
            .iter()
            .map(|joined| joined.place.name())
            .collect(),
        settings: brick
            .settings_entries()
            .iter()
            .map(|section| DescribedEntry {
                name: section.key,
                key: Some(section.key),
                verb: None,
                at: at(section.at),
            })
            .collect(),
        stats: brick
            .stats_entries()
            .iter()
            .map(|stats| DescribedEntry {
                name: stats.key,
                key: None,
                verb: None,
                at: at(stats.at),
            })
            .collect(),
        operations: brick
            .operations()
            .iter()
            .map(|operation| DescribedEntry {
                name: operation.name,
                key: None,
                verb: Some(if operation.control {
                    Verb::Control.name()
                } else {
                    Verb::Query.name()
                }),
                at: at(operation.at),
            })
            .collect(),
        explains: brick
            .explainers()
            .iter()
            .map(|explainer| source_line(explainer.at))
            .collect(),
    }
}

fn inspector_at(registry: &Registry, name: &str) -> Option<String> {
    registry
        .section(name)
        .map(|section| section.at)
        .or_else(|| registry.stats_section(name).map(|stats| stats.at))
        .map(source_line)
}

fn frame_stats_schema() -> Value {
    serde_json::from_str(crate::FRAME_STATS_SCHEMA).expect("generated_schema_json")
}

fn stats_schema(key: &str) -> Value {
    let schema = frame_stats_schema();
    let property = &schema["properties"][key];
    with_definitions(property.clone(), &schema["$defs"])
}

fn operation_args(verb: &Value, name: &str) -> Value {
    let variant = verb["oneOf"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|variant| variant["properties"]["name"]["const"] == name);
    match variant {
        Some(variant) => with_definitions(
            variant["properties"]
                .get("args")
                .cloned()
                .unwrap_or_else(|| json!({ "type": "object" })),
            &verb["$defs"],
        ),
        None => Value::Null,
    }
}

fn with_definitions(mut schema: Value, definitions: &Value) -> Value {
    let mut wanted = Vec::new();
    references(&schema, &mut wanted);
    let mut kept = serde_json::Map::new();
    while let Some(name) = wanted.pop() {
        if kept.contains_key(&name) {
            continue;
        }
        if let Some(definition) = definitions.get(&name) {
            references(definition, &mut wanted);
            kept.insert(name, definition.clone());
        }
    }
    if !kept.is_empty()
        && let Some(fields) = schema.as_object_mut()
    {
        fields.insert("$defs".to_owned(), Value::Object(kept));
    }
    schema
}

fn references(value: &Value, names: &mut Vec<String>) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                match value
                    .as_str()
                    .and_then(|path| path.strip_prefix("#/$defs/"))
                {
                    Some(name) if key == "$ref" => names.push(name.to_owned()),
                    Some(_) | None => references(value, names),
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                references(value, names);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}
