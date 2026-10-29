use std::collections::BTreeMap;

use engine_core::bevy_ecs::prelude::World;
use engine_core::{BrickRegistry, InspectedKind, missing_brick};
use engine_scene::SceneEnvironment;
use engine_settings::ErrorCode;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::inspect::Inspector;
use crate::protocol::operation_request;
use crate::{BrowserControl, Control, DevError, DevTools, InspectorDescriptor, InspectorKind};

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

pub fn is_control(world: &World, op: &str) -> bool {
    crate::CONTROL_OPS.contains(&op)
        || world
            .get_resource::<BrickRegistry>()
            .and_then(|registry| registry.operation(op))
            .is_some_and(|operation| operation.control)
}

fn added(world: &World, brick: &str) -> bool {
    world
        .get_resource::<BrickRegistry>()
        .is_some_and(|registry| registry.has(brick))
}

pub(crate) fn run(world: &mut World, name: &str, request: Value) -> Result<Value, DevError> {
    let operation = world
        .get_resource::<BrickRegistry>()
        .and_then(|registry| registry.operation(name))
        .copied()
        .ok_or_else(|| {
            DevError::new(
                ErrorCode::MissingPlugin,
                "$.op",
                format!("no added brick answers {name}"),
            )
        })?;
    operation.run(world, request).map_err(DevError::from)
}

impl DevTools {
    pub(crate) fn query_operation(
        &self,
        world: &World,
        source: &str,
    ) -> Result<Option<(&'static str, Value)>, DevError> {
        let Some(op) = op_of(source) else {
            return Ok(None);
        };
        let registry = world.get_resource::<BrickRegistry>();
        if let Some(operation) = registry.and_then(|registry| registry.operation(&op))
            && !operation.control
        {
            return Ok(Some((operation.name, operation_request(source)?)));
        }
        match self.standard.of_operation(&op) {
            Some(brick) if !added(world, brick) => Err(missing_brick(brick, "$.op").into()),
            Some(_) | None => Ok(None),
        }
    }

    pub fn parse_control(
        &self,
        world: &World,
        source: &str,
        op: &str,
        paused: bool,
    ) -> Result<Control, DevError> {
        let registry = world.get_resource::<BrickRegistry>();
        if let Some(operation) = registry.and_then(|registry| registry.operation(op))
            && operation.control
        {
            let request = operation_request(source)?;
            if !paused {
                return Err(DevError::new(
                    ErrorCode::PauseRequired,
                    "$",
                    format!("pause the host before {op}"),
                ));
            }
            return Ok(Control::Operation(crate::controls::OperationCall::Brick {
                name: operation.name,
                request,
            }));
        }
        if let Some(brick) = self.standard.of_operation(op)
            && !added(world, brick)
        {
            return Err(missing_brick(brick, "$.op").into());
        }
        if op == "environment" {
            let mut request: Value = engine_scene::parse_json(source)?;
            if let Some(fields) = request.as_object_mut() {
                for key in ["op", "version", "merge"] {
                    fields.remove(key);
                }
            }
            let environment: SceneEnvironment = engine_scene::parse_value(request)?;
            environment
                .check(world, "$")
                .map_err(|error| self.named_brick(DevError::from(error)))?;
        }
        Ok(BrowserControl::parse(source, op, paused)?.route())
    }

    pub(crate) fn named_brick(&self, error: DevError) -> DevError {
        if error.code != ErrorCode::MissingPlugin {
            return error;
        }
        let key = error.path.rsplit('.').next().unwrap_or_default();
        match self.standard.of_section(key) {
            Some(brick) => missing_brick(brick, error.path).into(),
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
                descriptor: InspectorDescriptor::new("Environment", sections_schema(registry))
                    .with_description(
                        "The whole current look as a scene environment: paste it into a scene to \
                         persist it, or send sections to the environment control. Missing effect \
                         resources read as off. Each section's inspector describes its fields.",
                    ),
                kind: InspectorKind::Resource,
                has: Box::new(|_, _| true),
                read: Box::new(|world, _, _| Some(json!(SceneEnvironment::current(world)))),
                brick: None,
            },
        );
        inspectors
    }

    fn schema_named(&self, name: &str) -> Value {
        self.schemas
            .get(name)
            .cloned()
            .unwrap_or_else(|| json!({ "type": "object" }))
    }
}

fn op_of(source: &str) -> Option<String> {
    serde_json::from_str::<Value>(source)
        .ok()?
        .get("op")?
        .as_str()
        .map(str::to_owned)
}

pub(crate) fn sections_schema(registry: &BrickRegistry) -> Value {
    let properties: serde_json::Map<String, Value> = std::iter::once(("atmosphere", "Atmosphere"))
        .chain(
            registry
                .sections()
                .map(|section| (section.key, section.name)),
        )
        .map(|(field, inspector)| {
            (
                field.to_owned(),
                json!({
                    "type": ["object", "null"],
                    "description": format!("The {inspector} inspector describes its fields."),
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
