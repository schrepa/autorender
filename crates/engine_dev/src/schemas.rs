use engine_core::bevy_ecs::prelude::World;
use engine_core::{BrickRegistry, OperationEntry};
use serde_json::{Map, Value, json};

use crate::{protocol, views};

pub struct Generated {
    pub files: Vec<(&'static str, String)>,
    pub summary: Value,
    pub frame_stats: Value,
    pub inspectors: Value,
}

fn registry(world: &World) -> Result<&BrickRegistry, String> {
    world
        .get_resource::<BrickRegistry>()
        .ok_or_else(|| "the schemas' engine adds no bricks: add the standard bricks".to_owned())
}

fn serialized(schema: schemars::Schema) -> Value {
    serde_json::to_value(schema).expect("schemas serialize")
}

fn definition_of(mut schema: Value, definitions: &mut Map<String, Value>) -> (String, Value) {
    let Some(fields) = schema.as_object_mut() else {
        return (String::new(), schema);
    };
    if let Some(Value::Object(theirs)) = fields.remove("$defs") {
        for (name, definition) in theirs {
            definitions.entry(name).or_insert(definition);
        }
    }
    fields.remove("$schema");
    let name = fields
        .get("title")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    (name, schema)
}

pub fn reply(world: &World) -> Result<Value, String> {
    let registry = registry(world)?;
    let mut reply = serialized(schemars::schema_for!(crate::Reply));
    let mut definitions = reply["$defs"].as_object().cloned().unwrap_or_default();
    for schema in [
        serialized(schemars::schema_for!(crate::Step)),
        serialized(schemars::schema_for!(crate::Outcome)),
        serialized(schemars::schema_for!(crate::HostContext)),
    ] {
        let mut theirs = Map::new();
        let (name, schema) = definition_of(schema, &mut theirs);
        theirs.remove("Reply");
        for (other, definition) in theirs {
            definitions.entry(other).or_insert(definition);
        }
        if name.is_empty() {
            return Err("root schemas name their type".to_owned());
        }
        definitions.insert(name, schema);
    }
    let mut results = Map::new();
    let typed = crate::results::result_schemas().into_iter().chain(
        registry
            .operations()
            .filter_map(|operation| Some((operation.name, operation.result_schema()?))),
    );
    for (op, schema) in typed {
        let (name, schema) = definition_of(serialized(schema), &mut definitions);
        if name.is_empty() {
            return Err(format!("{op}'s result schema names no type"));
        }
        results.insert(op.to_owned(), json!({ "$ref": format!("#/$defs/{name}") }));
        definitions.insert(name, schema);
    }
    let required: Vec<&String> = results.keys().collect();
    let op_results = json!({
        "description": "Each op's result where it is typed, by op: what a successful reply's `result` holds besides what hosts add.",
        "type": "object",
        "properties": results,
        "required": required,
    });
    definitions.insert("OpResults".to_owned(), op_results);
    reply["$defs"] = Value::Object(definitions);
    Ok(reply)
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub fn generated(world: &World) -> Result<Generated, String> {
    let registry = registry(world)?;
    let mut problems = Vec::new();
    let mut schemas = json!({
        "Transform": schemars::schema_for!(views::TransformView),
        "GlobalTransform": schemars::schema_for!(views::GlobalTransformView),
        "InterpolateTransform": schemars::schema_for!(views::InterpolateTransformView),
        "PresentationState": schemars::schema_for!(views::PresentationStateView),
        "Hierarchy": schemars::schema_for!(views::HierarchyView),
        "Camera": schemars::schema_for!(views::CameraView),
        "Mesh3d": schemars::schema_for!(views::Mesh3dView),
        "Atmosphere": schemars::schema_for!(engine_scene::SceneAtmosphere),
        "SplineMesh": schemars::schema_for!(views::SplineMeshView),
        "ReflectionProbe": schemars::schema_for!(engine_scene::SceneReflectionProbe),
        "Input": schemars::schema_for!(views::InputView),
        "LocalLight": schemars::schema_for!(engine_scene::SceneLight),
        "ParticleEmitter": schemars::schema_for!(engine_scene::SceneEmitter),
        "Collider": schemars::schema_for!(engine_collision::ColliderView),
        "RigidBody": schemars::schema_for!(engine_physics::RigidBodyView),
        "Terrain": schemars::schema_for!(views::TerrainView),
        "Tint": schemars::schema_for!(views::TintView),
        "Grounded": schemars::schema_for!(views::GroundedView),
        "Jobs": schemars::schema_for!(views::JobsView),
        "StaticTables": schemars::schema_for!(views::StaticTablesView),
        "StreamedId": schemars::schema_for!(views::StreamedIdView),
        "protocol": protocol_schema(registry),
        "BrowserControl": control_schema(registry),
        "FrameStats": frame_stats(registry, &mut problems),
    });
    if let Some(all) = schemas.as_object_mut() {
        for inspector in registry.inspectors() {
            match inspector.schema() {
                Some(schema) => {
                    all.insert(inspector.name.to_owned(), serialized(schema));
                }
                None => problems.push(format!(
                    "the {} brick's inspector {} has no schema",
                    inspector.brick, inspector.name
                )),
            }
        }
    }
    reserved_inspectors(&mut problems);
    let Some(described) = schemas.as_object() else {
        return Err("the inspectors' schemas are not an object".to_owned());
    };
    for (name, schema) in described {
        let bytes = schema.to_string().len();
        if bytes + 8 * 1024 > protocol::MAX_RESPONSE_BYTES {
            problems.push(format!(
                "describe {name} would take {bytes} bytes of a {}-byte reply: split the type or page its schema",
                protocol::MAX_RESPONSE_BYTES
            ));
        }
    }
    let mut sections = Map::new();
    for section in registry.sections() {
        sections.insert(section.name.to_owned(), serialized(section.schema()));
        if let Some(component) = section.name.strip_suffix('s')
            && schemas.get(component).is_some()
        {
            problems.push(format!(
                "settings {} read as a store of {component}s: name them {component}Settings",
                section.name
            ));
        }
    }
    if let Some(all) = schemas.as_object_mut() {
        all.extend(sections);
        all.insert("bricks".to_owned(), bricks_summary(registry));
    }
    let scene_schema = engine_scene::SceneDocument::schema_json(registry);
    let queries = ops(&schemas["protocol"], &mut problems);
    let controls = ops(&schemas["BrowserControl"], &mut problems);
    control_state(registry, &controls, &schemas, &mut problems);
    typed_results(registry, &queries, &controls, &mut problems);
    match serde_json::from_str::<Value>(&scene_schema) {
        Ok(scene) => authored_parity(&scene, &schemas, &mut problems),
        Err(error) => problems.push(format!("the scene schema is not JSON: {error}")),
    }
    if !problems.is_empty() {
        return Err(problems.join("\n"));
    }
    let constant = |name: &str, what: &str, ops: &[String]| {
        format!(
            "/// The op of every {what}, as requests name it.
pub const {name}: [&str; {}] = {:?};
",
            ops.len(),
            ops
        )
    };
    let summary = json!({
        "version": protocol::PROTOCOL_VERSION,
        "max_request_bytes": protocol::MAX_REQUEST_BYTES,
        "profile_max_events": engine_profile::MAX_EVENTS,
        "queries": queries,
        "controls": controls,
        "bricks": registry.names().collect::<Vec<_>>(),
        "errors": engine_settings::ErrorCode::ALL
            .iter()
            .map(|&(_, name)| name)
            .collect::<Vec<_>>(),
    });
    let pretty = |value: &Value| {
        serde_json::to_string_pretty(value).map_err(|error| format!("cannot write JSON: {error}"))
    };
    let files = vec![
        (
            "crates/engine_dev/src/generated/inspectors.json",
            schemas.to_string(),
        ),
        (
            "crates/engine_dev/src/generated/frame_stats.schema.json",
            pretty(&schemas["FrameStats"])?,
        ),
        (
            "crates/engine_dev/src/generated/scene.schema.json",
            scene_schema,
        ),
        (
            "crates/engine_dev/src/generated/ops.rs",
            constant("QUERY_OPS", "query", &queries)
                + &constant("CONTROL_OPS", "control", &controls),
        ),
        ("web/src/generated/protocol.json", pretty(&summary)? + "\n"),
    ];
    Ok(Generated {
        files,
        summary,
        frame_stats: schemas["FrameStats"].clone(),
        inspectors: schemas,
    })
}

fn bricks_summary(registry: &BrickRegistry) -> Value {
    registry
        .bricks()
        .iter()
        .map(|brick| {
            let settings: Vec<&str> = brick.settings.iter().map(|section| section.key).collect();
            let operations: Vec<&str> = brick
                .operations
                .iter()
                .map(|operation| operation.name)
                .collect();
            let inspectors: Vec<&str> = brick
                .inspectors
                .iter()
                .map(|inspector| inspector.name)
                .collect();
            json!({
                "name": brick.name,
                "settings": settings,
                "operations": operations,
                "inspectors": inspectors,
            })
        })
        .collect()
}

fn typed_results(
    registry: &BrickRegistry,
    queries: &[String],
    controls: &[String],
    problems: &mut Vec<String>,
) {
    let ops: Vec<&str> = queries
        .iter()
        .chain(controls)
        .map(String::as_str)
        .filter(|op| registry.operation(op).is_none())
        .collect();
    let (typed, untyped) = (
        crate::results::TYPED_RESULTS,
        crate::results::UNTYPED_RESULTS,
    );
    for op in &ops {
        if typed.contains(op) == untyped.contains(op) {
            problems.push(format!(
                "op {op}'s result is typed (results!) or listed in UNTYPED_RESULTS, not both or neither"
            ));
        }
    }
    for op in typed.iter().chain(untyped) {
        if !ops.contains(op) {
            problems.push(format!("{op} in results.rs is no op"));
        }
    }
    if untyped.windows(2).any(|pair| pair[0] >= pair[1]) {
        problems.push("UNTYPED_RESULTS (engine_dev results.rs) must be sorted and unique".into());
    }
}

fn version_schema() -> Value {
    serialized(schemars::SchemaGenerator::default().subschema_for::<u32>())
}

fn operation_variant(operation: &OperationEntry, definitions: &mut Map<String, Value>) -> Value {
    let (_, mut variant) = definition_of(serialized(operation.schema()), definitions);
    let Some(fields) = variant.as_object_mut() else {
        return variant;
    };
    fields.remove("title");
    fields.insert("type".to_owned(), json!("object"));
    let properties = fields
        .entry("properties")
        .or_insert_with(|| json!({}));
    if let Some(properties) = properties.as_object_mut() {
        properties.insert(
            "op".to_owned(),
            json!({ "const": operation.name, "type": "string" }),
        );
        properties.insert("version".to_owned(), version_schema());
    }
    let mut required = vec![json!("op"), json!("version")];
    if let Some(Value::Array(theirs)) = fields.remove("required") {
        required.extend(theirs);
    }
    fields.insert("required".to_owned(), Value::Array(required));
    variant
}

fn with_operations(mut schema: Value, registry: &BrickRegistry, control: bool) -> Value {
    let mut definitions = schema["$defs"].as_object().cloned().unwrap_or_default();
    let variants: Vec<Value> = registry
        .operations()
        .filter(|operation| operation.control == control)
        .map(|operation| operation_variant(operation, &mut definitions))
        .collect();
    if let Some(all) = schema.get_mut("oneOf").and_then(Value::as_array_mut) {
        all.extend(variants);
    }
    if !definitions.is_empty() {
        schema["$defs"] = Value::Object(definitions);
    }
    schema
}

fn protocol_schema(registry: &BrickRegistry) -> Value {
    with_operations(
        serialized(schemars::schema_for!(protocol::DevRequest)),
        registry,
        false,
    )
}

fn control_schema(registry: &BrickRegistry) -> Value {
    let mut schema = with_operations(
        serialized(schemars::schema_for!(protocol::BrowserControl)),
        registry,
        true,
    );
    let mut titles = vec![("SceneAtmosphere".to_owned(), "Atmosphere")];
    let mut sections = Map::new();
    for section in registry.sections() {
        let title = serialized(section.schema())["title"]
            .as_str()
            .unwrap_or(section.name)
            .to_owned();
        sections.insert(
            section.key.to_owned(),
            json!({
                "anyOf": [{ "$ref": format!("#/$defs/{title}") }, { "type": "null" }],
                "default": null,
            }),
        );
        titles.push((title, section.name));
    }
    if let Some(environment) = schema["oneOf"]
        .as_array_mut()
        .into_iter()
        .flatten()
        .find(|variant| variant["properties"]["op"]["const"] == "environment")
        .and_then(|variant| variant["properties"].as_object_mut())
    {
        environment.extend(sections);
    }
    if let Some(definitions) = schema
        .as_object_mut()
        .map(|schema| schema.entry("$defs").or_insert_with(|| json!({})))
        .and_then(Value::as_object_mut)
    {
        for (title, inspector) in titles {
            definitions.insert(
                title,
                json!({
                    "type": "object",
                    "description": format!("The {inspector} inspector describes its fields."),
                }),
            );
        }
    }
    let mut reached = std::collections::BTreeSet::new();
    let mut pending = Vec::new();
    let mut root = schema.clone();
    if let Some(object) = root.as_object_mut() {
        object.remove("$defs");
    }
    references(&root, &mut pending);
    while let Some(name) = pending.pop() {
        if reached.insert(name.clone())
            && let Some(definition) = schema["$defs"].get(&name)
        {
            references(definition, &mut pending);
        }
    }
    if let Some(definitions) = schema["$defs"].as_object_mut() {
        definitions.retain(|name, _| reached.contains(name));
    }
    schema
}

fn references(value: &Value, names: &mut Vec<String>) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                if key == "$ref"
                    && let Some(name) = value
                        .as_str()
                        .and_then(|path| path.strip_prefix("#/$defs/"))
                {
                    names.push(name.to_owned());
                } else {
                    references(value, names);
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

fn frame_stats(registry: &BrickRegistry, problems: &mut Vec<String>) -> Value {
    let mut schema = serialized(schemars::schema_for!(engine_core::FrameStats));
    let mut definitions = schema["$defs"].as_object().cloned().unwrap_or_default();
    let mut properties = schema["properties"].as_object().cloned().unwrap_or_default();
    let mut required = schema["required"].as_array().cloned().unwrap_or_default();
    for stats in registry.stats() {
        let (name, definition) = definition_of(serialized(stats.schema()), &mut definitions);
        let mut definition = definition;
        if let Some(fields) = definition.as_object_mut() {
            fields.remove("title");
        }
        if name.is_empty() {
            problems.push(format!("the {} brick's stats schema names no type", stats.brick));
            continue;
        }
        definitions.insert(name.clone(), definition);
        properties.insert(
            stats.key.to_owned(),
            json!({ "$ref": format!("#/$defs/{name}") }),
        );
        required.push(json!(stats.key));
    }
    schema["$defs"] = Value::Object(definitions);
    schema["properties"] = Value::Object(properties);
    schema["required"] = Value::Array(required);
    let objects = std::iter::once(("FrameStats", &schema)).chain(
        schema["$defs"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(name, definition)| (name.as_str(), definition)),
    );
    for (name, object) in objects {
        for field in object["properties"]
            .as_object()
            .into_iter()
            .flat_map(Map::keys)
        {
            if field.contains("dropped") && field != "dropped" && !field.starts_with("dropped_") {
                problems.push(format!(
                    "{name}.{field}: name what budgets drop `dropped` or `dropped_*`"
                ));
            }
        }
    }
    let render = serialized(schemars::schema_for!(engine_core::RenderStats));
    for (field, property) in render["properties"].as_object().into_iter().flatten() {
        if property.get("$ref").is_some() && registry.section(field).is_none() {
            problems.push(format!(
                "RenderStats.{field} groups a feature's stats under a key that is no environment \
                 section's: name it after the section"
            ));
        }
    }
    schema
}

fn reserved_inspectors(problems: &mut Vec<String>) {
    let mut tools = crate::DevTools::bare();
    for name in crate::reserved_names() {
        let probe = crate::InspectorDescriptor::new(name, json!({"type": "object"}));
        match tools.register_resource::<engine_core::FixedTime>(probe, |_| Value::Null) {
            Err(error) if error.code == engine_settings::ErrorCode::ReservedInspector => {}
            Err(error) => problems.push(format!(
                "an inspector named {name}, which describe answers itself, is refused as {} instead of reserved_inspector",
                error.code
            )),
            Ok(()) => problems.push(format!(
                "an inspector may take the name {name}, which describe answers itself: refuse it as reserved"
            )),
        }
    }
}

fn control_state(
    registry: &BrickRegistry,
    controls: &[String],
    schemas: &Value,
    problems: &mut Vec<String>,
) {
    let mut state: Vec<(&str, &str)> = protocol::CONTROL_STATE.to_vec();
    state.extend(
        registry
            .operations()
            .filter(|operation| operation.control)
            .map(|operation| (operation.name, operation.shows)),
    );
    for op in controls {
        let listed = state.iter().filter(|(listed, _)| listed == op).count();
        if listed != 1 {
            problems.push(format!(
                "control {op} must name where its state shows once in CONTROL_STATE or its \
                 operation's SHOWS"
            ));
        }
    }
    for (op, shows) in &state {
        if !controls.iter().any(|control| control == op) {
            problems.push(format!("CONTROL_STATE lists {op}, which is no control"));
        }
        if !["", "status", "GameControls", "Environment"].contains(shows)
            && schemas.get(*shows).is_none()
        {
            problems.push(format!(
                "control {op}'s state shows in {shows}, which is no inspector"
            ));
        }
    }
}

const AUTHORED: &[(&str, &str)] = &[
    ("id", "header"),
    ("name", "header"),
    ("tags", "header"),
    ("mesh", "Mesh3d.mesh"),
    ("material", "Mesh3d.material"),
    ("parent", "Hierarchy.authored_parent"),
    ("prefab", "Hierarchy.prefab_id"),
    ("transform", "Transform"),
];

const DOCUMENT: &[(&str, &str)] = &[
    ("id", "structure"),
    ("revision", "structure"),
    ("entities", "structure"),
    ("prefabs", "structure"),
    ("streamed", "structure"),
    ("environment", "Environment"),
    ("camera", "Camera"),
    ("meshes", "Mesh3d.mesh"),
    ("materials", "Mesh3d.material"),
    ("textures", "Mesh3d.textures"),
];

const RENAMED: &[(&str, &str)] = &[
    ("SceneTransform.rotation_degrees", "rotation"),
    ("SceneCamera.fov_degrees", "vertical_fov_radians"),
    ("SceneTexture.rgba8", "inline"),
];

fn authored_parity(scene: &Value, schemas: &Value, problems: &mut Vec<String>) {
    let definitions = &scene["$defs"];
    let object = definitions["SceneObject"]["properties"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    let authored = || AUTHORED.iter().chain(engine_scene::COMPONENT_SHOWS);
    for (property, schema) in &object {
        match authored().find(|(authored, _)| authored == property) {
            Some(&(_, shows)) => shown(
                (definitions, schemas),
                &format!("entities[].{property}"),
                schema,
                shows,
                problems,
            ),
            None => problems.push(format!(
                "entity property {property} needs a place in AUTHORED"
            )),
        }
    }
    for (property, _) in authored() {
        if object.get(*property).is_none() {
            problems.push(format!(
                "AUTHORED lists {property}, which entities do not author"
            ));
        }
    }
    let document = scene["properties"].as_object().cloned().unwrap_or_default();
    for (property, schema) in &document {
        match DOCUMENT.iter().find(|(authored, _)| authored == property) {
            Some(&(_, shows)) => shown((definitions, schemas), property, schema, shows, problems),
            None => problems.push(format!(
                "scene property {property} needs a place in DOCUMENT"
            )),
        }
    }
    for (property, _) in DOCUMENT {
        if document.get(*property).is_none() {
            problems.push(format!(
                "DOCUMENT lists {property}, which scenes do not author"
            ));
        }
    }
}

fn shown(
    (definitions, schemas): (&Value, &Value),
    what: &str,
    property: &Value,
    shows: &str,
    problems: &mut Vec<String>,
) {
    if ["header", "structure", "Environment"].contains(&shows) {
        return;
    }
    let (inspector, field) = shows.split_once('.').unwrap_or((shows, ""));
    let Some(root) = schemas.get(inspector) else {
        problems.push(format!(
            "{what} shows in {inspector}, which is no inspector"
        ));
        return;
    };
    let mut target = root;
    if !field.is_empty() {
        match resolve(&root["$defs"], &root["properties"][field]) {
            Some(resolved) => target = resolved,
            None => {
                problems.push(format!("{what} shows in {shows}, which is no field"));
                return;
            }
        }
    }
    let target = resolve(&root["$defs"], target).unwrap_or(target);
    let Some((name, authored)) = definition(definitions, property) else {
        return;
    };
    let Some(fields) = authored["properties"].as_object() else {
        return;
    };
    if target.as_bool() == Some(true) || target.as_object().is_some_and(Map::is_empty) {
        return;
    }
    for field in fields.keys() {
        let shown = RENAMED
            .iter()
            .find(|(renamed, _)| *renamed == format!("{name}.{field}"))
            .map_or(field.as_str(), |(_, shown)| shown);
        if target["properties"].get(shown).is_none() {
            problems.push(format!(
                "{what}.{field} ({name}) is authored but {shows} does not show it: add a field named {shown}, or rename it in RENAMED"
            ));
        }
    }
}

fn definition<'a>(definitions: &'a Value, schema: &'a Value) -> Option<(String, &'a Value)> {
    if let Some(reference) = schema["$ref"].as_str() {
        let name = reference.rsplit('/').next()?;
        return Some((name.to_owned(), definitions.get(name)?));
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        for option in schema[key].as_array().into_iter().flatten() {
            if let Some(found) = definition(definitions, option) {
                return Some(found);
            }
        }
    }
    definition(definitions, schema.get("items")?)
}

fn resolve<'a>(definitions: &'a Value, schema: &'a Value) -> Option<&'a Value> {
    if schema.is_null() {
        return None;
    }
    if schema.get("properties").is_some() {
        return Some(schema);
    }
    definition(definitions, schema)
        .map(|(_, body)| body)
        .or_else(|| {
            schema
                .get("items")
                .and_then(|items| resolve(definitions, items))
        })
        .or(Some(schema))
}

fn ops(schema: &Value, problems: &mut Vec<String>) -> Vec<String> {
    let variants = schema
        .get("oneOf")
        .or_else(|| schema.get("anyOf"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if variants.is_empty() {
        problems.push("request enums are one of their variants".to_owned());
    }
    variants
        .iter()
        .filter_map(|variant| {
            let op = &variant["properties"]["op"];
            let name = op
                .get("const")
                .or_else(|| op["enum"].get(0))
                .and_then(Value::as_str);
            if name.is_none() {
                problems.push(format!("a request variant names no op: {variant}"));
            }
            name.map(str::to_owned)
        })
        .collect()
}
