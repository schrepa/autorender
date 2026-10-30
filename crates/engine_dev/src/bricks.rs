use std::collections::BTreeMap;

use engine_core::bevy_ecs::prelude::World;
use engine_core::{BrickRegistry, InspectedKind, missing_brick};
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
    settings: Vec<String>,
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

    pub(crate) fn of_section(&self, key: &str) -> Option<&str> {
        self.find(|brick| brick.settings.as_slice(), key)
    }
}

fn added(world: &World, brick: &str) -> bool {
    world
        .get_resource::<BrickRegistry>()
        .is_some_and(|registry| registry.has(brick))
}

fn registered(world: &World, name: &str, control: bool) -> Option<&'static str> {
    world
        .get_resource::<BrickRegistry>()
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
                .get_resource::<BrickRegistry>()
                .into_iter()
                .flat_map(BrickRegistry::operations)
                .filter(|operation| operation.control == control)
                .map(|operation| operation.name),
        )
        .collect()
}

pub(crate) fn run(world: &mut World, name: &str, args: Value) -> Result<Value, Error> {
    let operation = world
        .get_resource::<BrickRegistry>()
        .and_then(|registry| registry.operation(name))
        .copied()
        .ok_or_else(|| {
            Error::new(ErrorCode::MissingPlugin, "$.name")
                .expected(Expected::Present)
                .got(name)
        })?;
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
                engine_core::parse_value(sections).map_err(|error| error.within("$.args"))?;
            environment
                .check(world, "$")
                .map_err(|error| self.named_brick(error).within("$.args"))?;
        }
        Ok(BrowserControl::parse(name, args, paused)?.route())
    }

    pub(crate) fn named_brick(&self, error: Error) -> Error {
        if error.code != ErrorCode::MissingPlugin {
            return error;
        }
        let key = error.path.rsplit('.').next().unwrap_or_default();
        match self.standard.of_section(key) {
            Some(brick) => missing_brick(brick, error.path),
            None => error,
        }
    }

    pub(crate) fn brick_inspectors(&self, world: &World) -> BTreeMap<String, Inspector> {
        let mut inspectors = BTreeMap::new();
        let Some(registry) = world.get_resource::<BrickRegistry>() else {
            return inspectors;
        };
        for section in registry.sections().copied() {
            inspectors.insert(
                section.name.to_owned(),
                Inspector {
                    descriptor: InspectorDescriptor::new(
                        section.name,
                        self.schema_named(section.name),
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
        for entry in registry.inspectors() {
            let (read, has) = (entry.clone(), entry.clone());
            inspectors.insert(
                entry.name.to_owned(),
                Inspector {
                    descriptor: InspectorDescriptor::new(entry.name, self.schema_named(entry.name)),
                    kind: match entry.kind {
                        InspectedKind::Component => InspectorKind::Component,
                        InspectedKind::Resource => InspectorKind::Resource,
                    },
                    has: Box::new(move |world, entity| has.has(world, entity)),
                    read: Box::new(move |world, entity, _| read.read(world, entity)),
                    brick: Some(entry.brick),
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

pub(crate) fn sections_schema(registry: &BrickRegistry) -> Value {
    let properties: serde_json::Map<String, Value> = registry
        .sections()
        .map(|section| {
            (
                section.key.to_owned(),
                json!({
                    "type": ["object", "null"],
                    "title": section.name,
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
