use std::collections::BTreeMap;

use bevy_ecs::prelude::World;
use engine_registry::Registry;
use engine_scene::SceneEnvironment;
use engine_settings::{Error, ErrorCode, Expected};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::inspect::Inspector;
use crate::protocol::{Envelope, Query, Verb};
use crate::{BrowserControl, Control, DevTools, InspectorDescriptor, InspectorKind};

#[derive(Debug, Default, Deserialize)]
pub(crate) struct StandardBrick {
    name: String,
    #[serde(default)]
    operations: Vec<String>,
    #[serde(default)]
    inspectors: Vec<String>,
}

#[derive(Debug, Default)]
pub(crate) struct StandardBricks(Vec<StandardBrick>);

impl StandardBricks {
    pub(crate) fn of(schemas: &Value) -> Self {
        Self(
            schemas
                .get("bricks")
                .and_then(|bricks| serde_json::from_value(bricks.clone()).ok())
                .unwrap_or_default(),
        )
    }

    fn find(&self, listed: impl Fn(&StandardBrick) -> &[String], name: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|brick| listed(brick).iter().any(|listed| listed == name))
            .map(|brick| brick.name.as_str())
    }

    pub(crate) fn of_operation(&self, op: &str) -> Option<&str> {
        self.find(|brick| brick.operations.as_slice(), op)
    }

    pub(crate) fn of_inspector(&self, name: &str) -> Option<&str> {
        self.find(|brick| brick.inspectors.as_slice(), name)
    }
}

pub(crate) fn added(world: &World, brick: &str) -> bool {
    world
        .get_resource::<Registry>()
        .is_some_and(|registry| registry.has(brick))
}

pub(crate) fn missing_brick(name: &str, path: impl Into<String>) -> Error {
    Error::new(ErrorCode::MissingPlugin, path)
        .expected(Expected::Present)
        .got(name)
}

fn registered(world: &World, name: &str, control: bool) -> Option<&'static str> {
    world
        .get_resource::<Registry>()
        .and_then(|registry| registry.operation(name))
        .filter(|operation| operation.control == control)
        .map(|operation| operation.name)
}

pub fn operation_names(world: &World, verb: Verb) -> Vec<&'static str> {
    let control = verb == Verb::Control;
    let core: &[&'static str] = if control {
        &BrowserControl::NAMES
    } else {
        &Query::NAMES
    };
    core.iter()
        .copied()
        .chain(
            world
                .get_resource::<Registry>()
                .into_iter()
                .flat_map(Registry::operations)
                .filter(|operation| operation.control == control)
                .map(|operation| operation.name),
        )
        .collect()
}

pub(crate) fn run(world: &mut World, name: &str, args: Value) -> Result<Value, Error> {
    let operation = world
        .get_resource::<Registry>()
        .and_then(|registry| registry.operation(name))
        .copied()
        .ok_or_else(|| missing_brick(name, "$.name"))?;
    operation
        .run(world, args)
        .map_err(|error| error.within("$.args"))
}

impl DevTools {
    pub(crate) fn query_operation(
        &self,
        world: &World,
        name: &str,
    ) -> Result<Option<&'static str>, Error> {
        if let Some(name) = registered(world, name, false) {
            return Ok(Some(name));
        }
        self.unadded(world, name)?;
        Ok(None)
    }

    fn unadded(&self, world: &World, name: &str) -> Result<(), Error> {
        match self.standard.of_operation(name) {
            Some(brick) if !added(world, brick) => Err(missing_brick(brick, "$.name")),
            Some(_) | None => Ok(()),
        }
    }

    pub fn parse_control(
        &self,
        world: &World,
        envelope: &Envelope,
        paused: bool,
    ) -> Result<Control, Error> {
        let name = envelope.name()?;
        let args = envelope.args()?;
        if let Some(name) = registered(world, name, true) {
            if !paused {
                return Err(Error::new(ErrorCode::PauseRequired, "$")
                    .expected(Expected::Paused)
                    .got(name));
            }
            return Ok(Control::Operation(crate::controls::OperationCall::Brick {
                name,
                request: args,
            }));
        }
        self.unadded(world, name)?;
        if !BrowserControl::NAMES.contains(&name) {
            return Err(Error::new(ErrorCode::UnknownVariant, "$.name")
                .got(name)
                .variants(operation_names(world, Verb::Control)));
        }
        if name == "environment" {
            let mut sections = args.clone();
            if let Some(fields) = sections.as_object_mut() {
                fields.remove("merge");
            }
            let environment: SceneEnvironment =
                engine_scene::parse_value(sections).map_err(|error| error.within("$.args"))?;
            environment
                .check(world, "$")
                .map_err(|error| error.within("$.args"))?;
        }
        Ok(BrowserControl::parse(name, args, paused)?.route())
    }

    pub(crate) fn brick_inspectors(&self, world: &World) -> BTreeMap<String, Inspector> {
        let mut inspectors = BTreeMap::new();
        let Some(registry) = world.get_resource::<Registry>() else {
            return inspectors;
        };
        for section in registry.sections().copied() {
            inspectors.insert(
                section.key.to_owned(),
                Inspector {
                    descriptor: InspectorDescriptor::new(
                        section.key,
                        schema_value(section.schema()),
                    )
                    .with_default(section.default_value()),
                    kind: InspectorKind::Resource,
                    has: Box::new(move |world, _| section.present(world)),
                    read: Box::new(move |world, _, _| {
                        section
                            .present(world)
                            .then(|| in_scene(world, || section.current(world)))
                    }),
                    brick: Some(section.brick),
                },
            );
        }
        for stats in registry.stats().copied() {
            inspectors.insert(
                stats.key.to_owned(),
                Inspector {
                    descriptor: InspectorDescriptor::new(stats.key, schema_value(stats.schema())),
                    kind: InspectorKind::Resource,
                    has: Box::new(|_, _| true),
                    read: Box::new(move |world, _, _| Some(stats.read(world))),
                    brick: Some(stats.brick),
                },
            );
        }
        for kind in engine_core::scene_components(world) {
            let key = kind.key();
            if inspectors.contains_key(key) {
                continue;
            }
            let schema = match kind.json_schema_in(&engine_core::SceneCx::of(world)) {
                Value::Null => json!({ "type": "object" }),
                schema => schema,
            };
            let (has, read) = (kind.clone(), kind.clone());
            inspectors.insert(
                key.to_owned(),
                Inspector {
                    descriptor: InspectorDescriptor::new(key, schema),
                    kind: InspectorKind::Component,
                    has: Box::new(move |world, entity| {
                        entity.is_some_and(|entity| has.value(world, entity).is_some())
                    }),
                    read: Box::new(move |world, entity, _| read.value(world, entity?)),
                    brick: None,
                },
            );
        }
        inspectors.insert(
            "Environment".to_owned(),
            Inspector {
                descriptor: InspectorDescriptor::new("Environment", sections_schema(registry)),
                kind: InspectorKind::Resource,
                has: Box::new(|_, _| true),
                read: Box::new(|world, _, _| Some(json!(SceneEnvironment::current(world)))),
                brick: None,
            },
        );
        inspectors
    }

    pub(crate) fn schema_named(&self, name: &str) -> Value {
        self.schemas
            .get(name)
            .cloned()
            .unwrap_or_else(|| json!({ "type": "object" }))
    }
}

pub(crate) fn in_scene<R>(world: &World, run: impl FnOnce() -> R) -> R {
    match world.get_resource::<engine_core::SceneContext>() {
        Some(context) => context.scope(run),
        None => run(),
    }
}

pub(crate) fn schema_value(schema: impl serde::Serialize) -> Value {
    serde_json::to_value(schema).expect("schema_serializes")
}

pub(crate) fn sections_schema(registry: &Registry) -> Value {
    let properties: serde_json::Map<String, Value> = registry
        .sections()
        .map(|section| {
            (
                section.key.to_owned(),
                json!({
                    "type": ["object", "null"],
                    "title": section.key,
                }),
            )
        })
        .collect();
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SceneEnvironment",
        "type": "object",
        "additionalProperties": false,
        "properties": properties,
    })
}
