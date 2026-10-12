mod components;
mod environment;
mod poses;

use poses::presentation_view;

use engine_settings::ErrorCode;
use std::{collections::BTreeMap, io::Write};

use engine_collision::{CollisionWorld, refresh_collision_world};
use engine_core::{
    ActionId, ActionInput, AssetLibrary, Atmosphere, Camera, Decal, Engine, Feature, FixedTime,
    FrameTime, GlobalTransforms, InputState, InterpolateTransform, LocalLight, MaterialHandle,
    Mesh3d, MeshHandle, Parent, ParticleEmitter, PresentationMode, PresentationState,
    PresentedTransforms, ReflectionProbe, TextureHandle, Tint, Transform,
    bevy_ecs::prelude::{Component, Entity, Resource, World},
};
use engine_scene::{
    EnvironmentSection, SceneAtmosphere, SceneDecal, SceneEmitter, SceneEntity, SceneEnvironment,
    SceneInstance, SceneLight, SceneMaterial, SceneReflectionProbe, SectionVisitor,
};
use serde::Serialize;
use serde_json::{Value, json};

use crate::protocol::{Route, ToolsQuery};
use crate::reply::{Observation, Reply, SceneObservation};
use crate::results::{
    AssetsResult, DescribeResult, DescribedType, DistributionResult, EntityComponents, EntityRow,
    InspectResult, StatusLimits, StatusResult, TextureAsset, TextureState, TypeKind, result,
};
use crate::{
    DevError, DevRequest, MAX_DESCRIPTION_BYTES, MAX_INSPECTOR_BYTES, MAX_INSPECTORS,
    MAX_PAGE_SIZE, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, MAX_SCAN_ENTITIES, PROTOCOL_VERSION,
    views::{
        CameraView, DecalView, GlobalTransformView, HierarchyView, InputView,
        InterpolateTransformView, LodLevelView, Mesh3dView, MeshBoundsView, MeshRenderView,
        MeshView, PresentationModeView, PresentationStateView, PresentedTransformView,
        SplineMeshView, TextureView, TransformView, VirtualMeshView,
    },
};

/// The types describe answers itself, by name and kind, with what one
/// is where its name does not say: no inspector may take their names.
const DESCRIBED: &[(&str, TypeKind, &str)] = &[
    ("SceneDocument", TypeKind::Authoring, ""),
    ("protocol", TypeKind::Protocol, ""),
    ("BrowserControl", TypeKind::Protocol, ""),
    ("FrameStats", TypeKind::Observation, ""),
    (
        "ErrorCodes",
        TypeKind::Catalog,
        "every error code replies answer with and what it means",
    ),
];

/// The types development hosts' tooling adds to describe's listing (the
/// agent sessions of web/scripts/agent-browser.mjs and agent-native.mjs):
/// no inspector may take their names either.
pub const TOOLING_TYPES: &[&str] = &["ProfileCapture", "GameControls", "PageParameters"];

/// The names no inspector may take: describe's own types and the hosts'
/// tooling's.
pub fn reserved_names() -> impl Iterator<Item = &'static str> {
    DESCRIBED
        .iter()
        .map(|&(name, _, _)| name)
        .chain(TOOLING_TYPES.iter().copied())
}

/// Names address deliberate SDK projections, never Rust type names or pointers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum InspectorKind {
    Component,
    Resource,
}

/// Structural metadata for a registered projection. Built-in schemas are
/// generated during host compilation from the same DTOs used for responses.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct InspectorDescriptor {
    pub name: String,
    pub description: String,
    pub schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
}

impl InspectorDescriptor {
    pub fn new(name: impl Into<String>, description: impl Into<String>, schema: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            schema,
            default: None,
        }
    }

    /// The descriptor of the generated schema named `name` in `schemas`,
    /// described by its type's doc comment: the engine's are `schemas`', an
    /// app's are its tooling's. A name without a schema, or a schema without
    /// a description, is a programming error.
    pub fn generated(schemas: &Value, name: &str) -> Self {
        let schema = schemas
            .get(name)
            .cloned()
            .unwrap_or_else(|| panic!("no schema named {name} was generated"));
        let description = schema["description"]
            .as_str()
            .unwrap_or_else(|| panic!("the schema of {name} has no description"))
            .replace('\n', " ");
        Self::new(name, description, schema)
    }

    #[must_use]
    pub fn with_default(mut self, value: Value) -> Self {
        self.default = Some(value);
        self
    }
}

type ReadInspector =
    Box<dyn Fn(&World, Option<Entity>, &Context<'_>) -> Option<Value> + Send + Sync>;

/// What a host's renderer made of the world's assets, which inspectors
/// report beside them. Hosts with a renderer answer it; without one, replies
/// say None where it would.
pub trait RendererFacts {
    /// Whether `texture` is resident on the GPU.
    fn texture_resident(&self, texture: TextureHandle) -> bool;
    /// `mesh` as uploaded; None before it uploads.
    fn mesh(&self, mesh: MeshHandle) -> Option<RenderedMesh>;
    /// Whether an impostor card of `mesh` drawn with `material` is baked;
    /// None where it never draws as one.
    fn impostor(&self, mesh: MeshHandle, material: MaterialHandle) -> Option<bool>;
    /// The level of detail render slot `slot` drew at, read back a few
    /// frames late: 0 the mesh, 1 and 2 its simplifications, 3 its impostor
    /// card. Asking watches the slot; None until a frame after that draws
    /// it with levels of detail.
    fn drawn_level(&self, slot: u32) -> Option<u32>;
}

/// A mesh as a renderer uploaded it.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedMesh {
    /// Whether it uploaded with skin joints.
    pub skinned: bool,
    /// Its coarser levels of detail, finest first: each one's index count
    /// and largest deviation from the mesh in mesh units.
    pub levels: Vec<(u32, f32)>,
}

/// What inspectors read beside the world: the active scene, whose authored
/// IDs name what it holds, the renderer's facts where the host has one, and
/// the render slot each entity draws in.
#[derive(Clone, Copy, Default)]
pub struct Context<'a> {
    pub scene: Option<&'a SceneInstance>,
    pub renderer: Option<&'a dyn RendererFacts>,
    pub render_slot: Option<&'a dyn Fn(Entity) -> Option<u32>>,
}
type HasInspector = Box<dyn Fn(&World, Option<Entity>) -> bool + Send + Sync>;

pub(crate) struct Inspector {
    pub(crate) descriptor: InspectorDescriptor,
    pub(crate) kind: InspectorKind,
    pub(crate) read: ReadInspector,
    pub(crate) has: HasInspector,
}

struct InspectSelector<'a> {
    entity: Option<&'a str>,
    parent: Option<&'a str>,
    component: Option<&'a str>,
    offset: usize,
    limit: usize,
}

/// Explicitly opted-in inspection registry. Registrations belong to the game
/// session and cannot be replaced silently. Extension callbacks are trusted
/// game code: keep their work and allocations bounded and never scan the world.
pub struct DevTools {
    pub(crate) inspectors: BTreeMap<String, Inspector>,
    // The engine's own inspectors, once registered: games register up to
    // MAX_INSPECTORS beyond them.
    builtins: Option<usize>,
    // The feature of each inspector of a feature's type, by name.
    pub(crate) features: BTreeMap<String, Feature>,
}

impl Default for DevTools {
    fn default() -> Self {
        let schemas: Value = serde_json::from_str(include_str!("../generated/inspectors.json"))
            .expect("generated schemas are valid JSON");
        let mut tools = Self {
            inspectors: BTreeMap::new(),
            builtins: None,
            features: BTreeMap::new(),
        };
        tools.register_poses(&schemas);
        tools.register_components(&schemas);
        tools.register_environment(&schemas);
        crate::features::register(&mut tools, &schemas);
        tools.builtins = Some(tools.inspectors.len());
        tools
    }
}

/// The scene schema, which exceeds one response: with `definition`, that
/// definition; without, the top level with its definitions' names in place
/// of their bodies.
fn scene_schema(definition: Option<&str>) -> Result<Value, DevError> {
    let mut schema: Value = serde_json::from_str(include_str!("../generated/scene.schema.json"))
        .expect("generated scene schema is JSON");
    let definitions = schema
        .as_object_mut()
        .and_then(|schema| schema.remove("$defs"))
        .unwrap_or_default();
    match definition {
        Some(name) => definitions.get(name).cloned().ok_or_else(|| {
            DevError::new(
                ErrorCode::UnknownDefinition,
                "$.definition",
                "not a definition of the scene schema; describe SceneDocument lists them",
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

/// The first sentence of `doc`, within an inspector description's bytes.
fn summary(doc: &str) -> String {
    let sentence = doc
        .find(". ")
        .map_or(doc, |end| &doc[..=end])
        .replace('\n', " ");
    if sentence.len() <= MAX_DESCRIPTION_BYTES {
        return sentence;
    }
    let mut end = MAX_DESCRIPTION_BYTES - 3;
    while !sentence.is_char_boundary(end) {
        end -= 1;
    }
    let cut = sentence[..end].rfind(' ').unwrap_or(end);
    format!("{}...", &sentence[..cut])
}

impl DevTools {
    /// Every component inspector's views of every entity holding its
    /// component, by inspector name, each view as compact JSON: what
    /// digests hash.
    pub(crate) fn component_views(
        &self,
        world: &World,
        context: Context<'_>,
    ) -> BTreeMap<String, Vec<String>> {
        let entities: Vec<Entity> = world.iter_entities().map(|entity| entity.id()).collect();
        self.inspectors
            .iter()
            .filter(|(_, inspector)| inspector.kind == InspectorKind::Component)
            .map(|(name, inspector)| {
                let views = entities
                    .iter()
                    .filter_map(|&entity| (inspector.read)(world, Some(entity), &context))
                    .map(|view| view.to_string())
                    .collect();
                (name.clone(), views)
            })
            .collect()
    }

    /// Registers a typed, read-only component projection without reflection.
    pub fn register_component<T: Component>(
        &mut self,
        descriptor: InspectorDescriptor,
        inspect: impl Fn(&T) -> Value + Send + Sync + 'static,
    ) -> Result<(), DevError> {
        self.insert(Inspector {
            descriptor,
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<T>(entity).is_some())
            }),
            read: Box::new(move |world, entity, _context| world.get::<T>(entity?).map(&inspect)),
        })
    }

    /// Registers a typed world resource, queried with `component` and no entity.
    pub fn register_resource<T: Resource>(
        &mut self,
        descriptor: InspectorDescriptor,
        inspect: impl Fn(&T) -> Value + Send + Sync + 'static,
    ) -> Result<(), DevError> {
        self.insert(Inspector {
            descriptor,
            kind: InspectorKind::Resource,
            has: Box::new(|world, _| world.get_resource::<T>().is_some()),
            read: Box::new(move |world, _, _context| world.get_resource::<T>().map(&inspect)),
        })
    }

    pub(crate) fn insert(&mut self, inspector: Inspector) -> Result<(), DevError> {
        let name = &inspector.descriptor.name;
        validate_name(name, "$.name")?;
        if reserved_names().any(|reserved| reserved == name) {
            return Err(DevError::new(
                ErrorCode::ReservedInspector,
                "$.name",
                format!("describe answers {name} itself: choose another inspector name"),
            ));
        }
        if self.inspectors.contains_key(name) {
            return Err(DevError::new(
                ErrorCode::DuplicateInspector,
                "$.name",
                "inspector name is already registered",
            ));
        }
        if let Some(builtins) = self.builtins
            && self.inspectors.len() >= builtins + MAX_INSPECTORS
        {
            return Err(DevError::new(
                ErrorCode::InspectorLimit,
                "$.name",
                format!("a game may register at most {MAX_INSPECTORS} inspectors"),
            ));
        }
        if inspector.descriptor.description.len() > MAX_DESCRIPTION_BYTES
            || encode(&inspector.descriptor, MAX_INSPECTOR_BYTES).is_err()
        {
            return Err(DevError::new(
                ErrorCode::InspectorTooLarge,
                "$.schema",
                format!(
                    "descriptor must fit {MAX_INSPECTOR_BYTES} bytes and description {MAX_DESCRIPTION_BYTES} bytes"
                ),
            ));
        }
        self.inspectors.insert(name.clone(), inspector);
        Ok(())
    }

    /// Processes one request outside ordinary frames. At most 10,000 authored
    /// records are scanned, 100 rows returned, and 64 KiB encoded. Counters and
    /// revisions are decimal strings to retain u64 precision across JavaScript.
    /// Refreshes derived transforms without simulation advance, extraction, or GPU work.
    pub fn execute(&self, engine: &mut Engine, context: Context<'_>, source: &str) -> String {
        let scene = context.scene;
        let world = engine.app.world();
        // The state before the request.
        let observation = Observation {
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
        };
        let (result, error) = match self.parse_and_execute(engine, context, source) {
            Ok(result) => (Some(result), None),
            Err(error) => (None, Some(error)),
        };
        let reply = Reply {
            version: Some(PROTOCOL_VERSION),
            ok: error.is_none(),
            observation: Some(observation),
            result,
            error,
        };
        encode(&reply, MAX_RESPONSE_BYTES).unwrap_or_else(|_| {
            let refused = Reply {
                ok: false,
                result: None,
                error: Some(DevError::new(
                    ErrorCode::ResponseTooLarge,
                    "$",
                    "response exceeds 65536 bytes; reduce limit or select one component",
                )),
                ..reply
            };
            serde_json::to_string(&refused).expect("replies serialize")
        })
    }

    fn parse_and_execute(
        &self,
        engine: &mut Engine,
        context: Context<'_>,
        source: &str,
    ) -> Result<Value, DevError> {
        let scene = context.scene;
        if source.len() > MAX_REQUEST_BYTES {
            return Err(DevError::new(
                ErrorCode::RequestTooLarge,
                "$",
                format!("request must fit {MAX_REQUEST_BYTES} UTF-8 bytes"),
            ));
        }
        let request: DevRequest = crate::parse_request(source)?;
        if scene.is_some_and(|scene| !scene.belongs_to(engine.app.world())) {
            return Err(DevError::new(
                ErrorCode::WrongWorld,
                "$.scene",
                "scene instance belongs to a different engine world",
            ));
        }
        let query = match request.route() {
            Route::Feature(query) => return query.answer(engine, scene),
            Route::Tools(query) => query,
        };
        if let ToolsQuery::Inspect {
            component: Some(component),
            entity,
            parent,
            offset,
            limit,
        } = &query
            && matches!(component.as_str(), "Collider" | "CollisionWorld")
        {
            // Validate selectors and pagination before refreshing any derived cache.
            self.inspect(
                engine.app.world(),
                context,
                InspectSelector {
                    entity: entity.as_deref(),
                    parent: parent.as_deref(),
                    component: Some(component),
                    offset: *offset,
                    limit: *limit,
                },
            )?;
            if engine.app.world().contains_resource::<CollisionWorld>() {
                refresh_collision_world(engine.app.world_mut())
                    .map_err(crate::features::collision::native_error)?;
            }
        }
        if matches!(query, ToolsQuery::Inspect { .. } | ToolsQuery::Digest) {
            // Materialize a GPU-presented frame before a refresh changes its inputs.
            engine.prepare_presented_transforms();
            engine.refresh_transforms();
        }
        let engine = &*engine;
        let slot = |entity| engine.render_slot(entity);
        let context = Context {
            render_slot: Some(&slot),
            ..context
        };
        let world = engine.app.world();
        match query {
            ToolsQuery::Status => Ok(result::status(self.status(engine, scene))),
            ToolsQuery::Assets => assets(world, context).map(result::assets),
            // The world's own state: neither what a renderer made of it nor
            // the slots it draws in.
            ToolsQuery::Digest => Ok(result::digest(self.digest(
                world,
                Context {
                    scene,
                    renderer: None,
                    render_slot: None,
                },
            ))),
            ToolsQuery::Describe { name, definition } => self
                .describe(name.as_deref(), definition.as_deref())
                .map(result::describe),
            ToolsQuery::Distribution { values } => {
                if values.len() > crate::bounds::MAX_VALUES {
                    return Err(DevError::new(
                        ErrorCode::OutOfRange,
                        "$.values",
                        format!("at most {} values", crate::bounds::MAX_VALUES),
                    ));
                }
                Ok(result::distribution(DistributionResult {
                    distribution: engine_core::Distribution::of(&values),
                }))
            }
            ToolsQuery::Inspect {
                entity,
                parent,
                component,
                offset,
                limit,
            } => self
                .inspect(
                    world,
                    context,
                    InspectSelector {
                        entity: entity.as_deref(),
                        parent: parent.as_deref(),
                        component: component.as_deref(),
                        offset,
                        limit,
                    },
                )
                .map(result::inspect),
        }
    }

    /// What `status` answers of `engine`, running `scene`.
    fn status(&self, engine: &Engine, scene: Option<&SceneInstance>) -> StatusResult {
        let world = engine.app.world();
        StatusResult {
            world_entities: world.entities().len(),
            authored_entities: scene.map(SceneInstance::entity_count),
            inspectors: self.inspectors.len(),
            culling_enabled: engine.frustum_culling(),
            features: world
                .get_resource::<engine_core::Features>()
                .map(|features| features.iter().map(Feature::name).collect()),
            presentation: presentation_view(engine.presentation()),
            collision: world
                .get_resource::<CollisionWorld>()
                .map(crate::features::collision::world_view),
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

    fn describe(
        &self,
        name: Option<&str>,
        definition: Option<&str>,
    ) -> Result<DescribeResult, DevError> {
        let feature = |name: &str| self.features.get(name).map(|feature| feature.name());
        let Some(name) = name else {
            let mut types: Vec<DescribedType> = self
                .inspectors
                .values()
                .map(|inspector| DescribedType {
                    name: inspector.descriptor.name.clone(),
                    kind: inspector.kind.into(),
                    description: Some(inspector.descriptor.description.clone()),
                    feature: Some(feature(&inspector.descriptor.name)),
                })
                .collect();
            types.extend(
                DESCRIBED
                    .iter()
                    .map(|&(name, kind, description)| DescribedType {
                        name: name.to_owned(),
                        kind,
                        description: (!description.is_empty()).then(|| description.to_owned()),
                        feature: None,
                    }),
            );
            return Ok(DescribeResult::Listing { types });
        };
        match name {
            "SceneDocument" => Ok(DescribeResult::Authoring {
                name: name.to_owned(),
                kind: TypeKind::Authoring,
                schema: scene_schema(definition)?,
                constraints: format!(
                    "Rust validation checks references, unique expanded IDs, cycles, depth, finite world matrices, collision admission, environment values and asset budgets. Prefabs cannot nest. An environment sets every section on instantiation, omitted ones to the engine default state; the opt-in ones ({}) are then off.",
                    SceneEnvironment::opt_in_sections().join(", ")
                ),
            }),
            "protocol" | "BrowserControl" => {
                let schemas: Value =
                    serde_json::from_str(include_str!("../generated/inspectors.json"))
                        .expect("generated schemas are JSON");
                Ok(DescribeResult::Protocol {
                    name: name.to_owned(),
                    kind: TypeKind::Protocol,
                    schema: schemas[name].clone(),
                    constraints: format!(
                        "version {PROTOCOL_VERSION}; every control but pause and resume requires a paused host; reload revisions must fit u64. Numeric bounds, lengths and defaults are in the schema. Spatial queries need CollisionPlugin but no pause; directions are nonzero and finite; rotations are normalized XYZW. Layer filtering is symmetric; sensors are included, excluded or the only ones. Raycasts and casts return the closest complete hit, terrain heightfields included; casts translate with a fixed orientation. Terrain heights read null where no tile is resident; terrain sweeps compare queries, collision rays and layers. Budget failures never return partial hits. Events are non-consuming latest fixed-tick sensor transitions, with attempted and committed ticks. Present and capture previews preserve clocks and input. The terrain control freezes streaming at the camera's position. Ordinary render, step and capture show current poses. Benchmarks in step mode advance one tick a sample, in present mode sample alpha 0.5 without ticks, and clear input. Controls do not save source."
                    ),
                    // Where each control's effect shows: an inspector, status, the game's controls, or nothing.
                    control_state: crate::protocol::CONTROL_STATE
                        .iter()
                        .map(|&(op, shows)| (op.to_owned(), json!(shows)))
                        .collect(),
                })
            }
            "ErrorCodes" => Ok(DescribeResult::Catalog {
                name: name.to_owned(),
                kind: TypeKind::Catalog,
                codes: ErrorCode::ALL
                    .iter()
                    .map(|&(code, text, _)| (text.to_owned(), json!(code.description())))
                    .collect(),
            }),
            "FrameStats" => Ok(DescribeResult::Observation {
                name: name.to_owned(),
                kind: TypeKind::Observation,
                schema: serde_json::from_str(crate::FRAME_STATS_SCHEMA)
                    .expect("generated schema is JSON"),
                source: "every successful reply's result.rendering: the last frame's stats",
            }),
            _ => {
                let inspector = self.inspector(name)?;
                Ok(DescribeResult::Inspector {
                    kind: inspector.kind.into(),
                    descriptor: inspector.descriptor.clone(),
                    feature: feature(name),
                })
            }
        }
    }

    /// Fails with `missing_plugin` when inspector `name` is of a feature
    /// `world`'s engine lacks.
    fn check_feature(&self, world: &World, name: &str) -> Result<(), DevError> {
        match self.features.get(name) {
            Some(&feature) if !engine_core::has_feature(world, feature) => {
                Err(crate::features::missing(feature))
            }
            _ => Ok(()),
        }
    }

    fn inspector(&self, name: &str) -> Result<&Inspector, DevError> {
        validate_name(name, "$.component")?;
        self.inspectors.get(name).ok_or_else(|| {
            DevError::new(
                ErrorCode::UnknownInspector,
                "$.component",
                "use describe to discover registered inspectors",
            )
        })
    }

    fn inspect(
        &self,
        world: &World,
        context: Context<'_>,
        selector: InspectSelector<'_>,
    ) -> Result<InspectResult, DevError> {
        let scene = context.scene;
        let InspectSelector {
            entity,
            parent,
            component,
            offset,
            limit,
        } = selector;
        if entity.is_some() && parent.is_some() {
            return Err(DevError::new(
                ErrorCode::InvalidSelector,
                "$.parent",
                "entity and parent selectors are mutually exclusive",
            ));
        }
        if !(1..=MAX_PAGE_SIZE).contains(&limit) || offset > MAX_SCAN_ENTITIES {
            return Err(DevError::new(
                ErrorCode::InvalidPage,
                "$.limit",
                "limit must be 1..100 and offset 0..10000",
            ));
        }
        let inspector = component.map(|name| self.inspector(name)).transpose()?;
        if let Some(inspector) =
            inspector.filter(|inspector| inspector.kind == InspectorKind::Resource)
        {
            if entity.is_some() || parent.is_some() || offset != 0 {
                return Err(DevError::new(
                    ErrorCode::InvalidSelector,
                    "$.entity",
                    "resources have no entity or parent selector or pagination offset",
                ));
            }
            self.check_feature(world, &inspector.descriptor.name)?;
            let value = read_value(inspector, world, None, &context)?;
            return Ok(InspectResult::Resource {
                resource: inspector.descriptor.name.clone(),
                value,
            });
        }
        let scene = scene.ok_or_else(|| {
            DevError::new(
                ErrorCode::NoScene,
                "$.entity",
                "authored entity inspection requires an active scene instance",
            )
        })?;
        if scene.entity_count() > MAX_SCAN_ENTITIES {
            return Err(DevError::new(
                ErrorCode::ScanLimit,
                "$.scene",
                "scene exceeds the 10000 authored-record scan budget",
            ));
        }
        if let Some(id) = entity {
            validate_name(id, "$.entity")?;
            if offset != 0 {
                return Err(DevError::new(
                    ErrorCode::InvalidSelector,
                    "$.offset",
                    "a selected entity has no pagination offset",
                ));
            }
            let entity = authored_entity(world, scene, id).ok_or_else(|| {
                DevError::new(
                    ErrorCode::UnknownEntity,
                    "$.entity",
                    "authored ID does not exist in the active scene (streamed entities only while their cell is loaded)",
                )
            })?;
            if world.get_entity(entity).is_err() {
                return Err(DevError::new(
                    ErrorCode::EntityRemoved,
                    "$.entity",
                    "gameplay removed this authored entity",
                ));
            }
            return Ok(InspectResult::Entity {
                entity: self.entity_row(world, context, scene, id, entity, inspector)?,
            });
        }
        let parent = parent
            .map(|id| {
                validate_name(id, "$.parent")?;
                let entity = authored_entity(world, scene, id).ok_or_else(|| {
                    DevError::new(
                        ErrorCode::UnknownEntity,
                        "$.parent",
                        "parent authored ID does not exist in the active scene",
                    )
                })?;
                if world.get_entity(entity).is_err() {
                    return Err(DevError::new(
                        ErrorCode::EntityRemoved,
                        "$.parent",
                        "gameplay removed this authored parent",
                    ));
                }
                Ok(entity)
            })
            .transpose()?;
        let mut rows = Vec::with_capacity(limit);
        let mut total = 0;
        for (id, entity) in scene.entities() {
            if world.get_entity(entity).is_err()
                || parent.is_some_and(|parent| {
                    world.get::<Parent>(entity).map(Parent::parent) != Some(parent)
                })
                || inspector.is_some_and(|inspector| !(inspector.has)(world, Some(entity)))
            {
                continue;
            }
            if total >= offset && rows.len() < limit {
                rows.push(self.entity_row(world, context, scene, id, entity, inspector)?);
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
        context: Context<'_>,
        scene: &SceneInstance,
        id: &str,
        entity: Entity,
        inspector: Option<&Inspector>,
    ) -> Result<EntityRow, DevError> {
        let metadata = world.get::<SceneEntity>(entity);
        let components = if let Some(inspector) = inspector {
            self.check_feature(world, &inspector.descriptor.name)?;
            let context = Context {
                scene: Some(scene),
                ..context
            };
            let value = read_value(inspector, world, Some(entity), &context)?;
            EntityComponents::Values(
                [(inspector.descriptor.name.clone(), value)]
                    .into_iter()
                    .collect(),
            )
        } else {
            EntityComponents::Names(
                self.inspectors
                    .values()
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
) -> Result<Value, DevError> {
    let value = (inspector.read)(world, entity, context).ok_or_else(|| {
        DevError::new(
            ErrorCode::MissingComponent,
            "$.component",
            "selected component or resource is absent",
        )
    })?;
    encode(&value, MAX_INSPECTOR_BYTES).map_err(|_| {
        DevError::new(
            ErrorCode::InspectorTooLarge,
            "$.component",
            "inspector projection exceeds 16384 encoded bytes",
        )
    })?;
    Ok(value)
}

fn validate_name(name: &str, path: &str) -> Result<(), DevError> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
    {
        return Err(DevError::new(
            ErrorCode::InvalidName,
            path,
            "names require 1..64 ASCII letters, digits, underscores, dots, or hyphens",
        ));
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
            return Err(std::io::Error::other("JSON byte budget exceeded"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn encode(value: &impl Serialize, limit: usize) -> Result<String, serde_json::Error> {
    let mut output = BoundedWriter {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut output, value)?;
    Ok(String::from_utf8(output.bytes).expect("JSON is UTF-8"))
}

/// The entity of authored ID `id` in `scene`: one it spawned, or else a
/// loaded streamed entity it owns.
pub(crate) fn authored_entity(world: &World, scene: &SceneInstance, id: &str) -> Option<Entity> {
    scene.entity(id).or_else(|| {
        scene.streamed()?;
        world
            .try_query::<(Entity, &SceneEntity)>()?
            .iter(world)
            .find(|(_, metadata)| metadata.authored_id == id && metadata.scene_id == scene.id())
            .map(|(entity, _)| entity)
    })
}

/// What `assets` answers of `world`'s asset library and the scene
/// `context` names: missing_resource without a library.
fn assets(world: &World, context: Context<'_>) -> Result<AssetsResult, DevError> {
    let assets = world.get_resource::<AssetLibrary>().ok_or_else(|| {
        DevError::new(
            ErrorCode::MissingResource,
            "$.assets",
            "AssetLibrary is not installed",
        )
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
    Ok(AssetsResult {
        meshes: assets.mesh_count(),
        materials: assets.material_count(),
        textures: assets.texture_count(),
        texture_cpu_bytes: assets.texture_bytes(),
        texture_assets,
        resident_cpu_bytes: assets.resident_bytes(),
        readiness: "synchronous",
        ownership: "library payload excludes retained snapshots and allocator metadata",
    })
}
