use std::collections::BTreeMap;

use bevy_ecs::prelude::World;
use engine_registry::Registry;
use engine_scene::SceneEnvironment;
use engine_settings::{Error, ErrorCode, Expected};
use serde_json::{Value, json};

use crate::inspect::Inspector;
use crate::protocol::{Envelope, Query, Verb};
use crate::{BrowserControl, Control, DevTools, InspectorDescriptor, InspectorKind};

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
        .ok_or_else(|| {
            Error::new(ErrorCode::UnknownVariant, "$.name")
                .expected(Expected::Present)
                .got(name)
        })?;
    operation
        .run(world, args)
        .map_err(|error| error.within("$.args"))
}

impl DevTools {
    pub(crate) fn query_operation(&self, world: &World, name: &str) -> Option<&'static str> {
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
                        section.present(world).then(|| section.current(world))
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
            let schema = match kind.json_schema() {
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
