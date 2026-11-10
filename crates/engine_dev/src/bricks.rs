use std::collections::BTreeMap;

use bevy_ecs::prelude::World;
use engine_registry::{BrickStates, Registry};
use engine_scene::SceneEnvironment;
use engine_settings::{Error, ErrorCode, Expected};
use serde_json::{Value, json};

use crate::inspect::Inspector;
use crate::protocol::{Envelope, Query, Verb};
use crate::{BrowserControl, Control, DevTools, InspectorDescriptor, InspectorKind};

fn missing_brick(name: &str, path: impl Into<String>) -> Error {
    Error::new(ErrorCode::MissingPlugin, path)
        .expected(Expected::Present)
        .got(name)
}

fn gated_off(world: &World, brick: &str) -> bool {
    let (Some(registry), Some(states)) = (
        world.get_resource::<Registry>(),
        world.get_resource::<BrickStates>(),
    ) else {
        return false;
    };
    registry.id(brick).is_some_and(|id| !states.brick_on(id))
}

fn brick_states(world: &World) -> Value {
    let (Some(registry), Some(states)) = (
        world.get_resource::<Registry>(),
        world.get_resource::<BrickStates>(),
    ) else {
        return json!({});
    };
    states
        .iter()
        .filter_map(|(brick, state)| {
            let record = registry.record(brick)?;
            Some((record.package.to_owned(), json!(state)))
        })
        .collect::<serde_json::Map<String, Value>>()
        .into()
}

fn registered(world: &World, name: &str, control: bool) -> Result<Option<&'static str>, Error> {
    let Some(operation) = world
        .get_resource::<Registry>()
        .and_then(|registry| registry.operation(name))
        .filter(|operation| operation.control == control)
    else {
        return Ok(None);
    };
    if gated_off(world, operation.brick) {
        return Err(missing_brick(operation.brick, "$.name"));
    }
    Ok(Some(operation.name))
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

impl DevTools {
    pub(crate) fn query_operation(
        &self,
        world: &World,
        name: &str,
    ) -> Result<Option<&'static str>, Error> {
        registered(world, name, false)
    }

    pub fn parse_control(
        &self,
        world: &World,
        envelope: &Envelope,
        paused: bool,
    ) -> Result<Control, Error> {
        let name = envelope.name()?;
        let args = envelope.args()?;
        self.control(world, (name, args), paused)
    }

    pub fn control(
        &self,
        world: &World,
        (name, args): (&str, Value),
        paused: bool,
    ) -> Result<Control, Error> {
        if let Some(name) = registered(world, name, true)? {
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
                        schema_value(
                            registry
                                .section_schema(section.key)
                                .unwrap_or_else(|| section.schema()),
                        ),
                    )
                    .with_default(section.default_value()),
                    kind: InspectorKind::Resource,
                    has: Box::new(move |world, _| section.present(world)),
                    read: Box::new(move |world, _, _| {
                        section
                            .present(world)
                            .then(|| {
                                in_scene(world, || section.current(world))
                                    .unwrap_or_else(|error| json!({ "unreadable": error }))
                            })
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
            let name = if inspectors.contains_key(key) {
                format!("{key}_component")
            } else {
                key.to_owned()
            };
            let schema = match kind.view_schema() {
                Some(schema) => schema,
                None => match kind.json_schema_in(&engine_core::SceneCx::of(world)) {
                    Value::Null => json!({ "type": "object" }),
                    schema @ (Value::Bool(_)
                    | Value::Number(_)
                    | Value::String(_)
                    | Value::Array(_)
                    | Value::Object(_)) => schema,
                },
            };
            let (has, read) = (kind.clone(), kind.clone());
            inspectors.insert(
                name.clone(),
                Inspector {
                    descriptor: InspectorDescriptor::new(name, schema),
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
            "Bricks".to_owned(),
            Inspector {
                descriptor: InspectorDescriptor::new("Bricks", json!({ "type": "object" })),
                kind: InspectorKind::Resource,
                has: Box::new(|_, _| true),
                read: Box::new(|world, _, _| Some(brick_states(world))),
                brick: None,
            },
        );
        inspectors.insert(
            "Ownership".to_owned(),
            Inspector {
                descriptor: InspectorDescriptor::new("Ownership", json!({ "type": "array" })),
                kind: InspectorKind::Resource,
                has: Box::new(|_, _| true),
                read: Box::new(|world, _, _| {
                    world
                        .get_resource::<Registry>()
                        .map(|registry| json!(registry.ownership()))
                }),
                brick: None,
            },
        );
        inspectors.insert(
            "Environment".to_owned(),
            Inspector {
                descriptor: InspectorDescriptor::new("Environment", sections_schema(registry)),
                kind: InspectorKind::Resource,
                has: Box::new(|_, _| true),
                read: Box::new(|world, _, _| {
                    Some(match SceneEnvironment::current(world) {
                        Ok(environment) => json!(environment),
                        Err(error) => json!({ "unreadable": error }),
                    })
                }),
                brick: None,
            },
        );
        inspectors
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
