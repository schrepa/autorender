use bevy_ecs::prelude::World;
use engine_core::SceneComponents;
use engine_registry::{MembersOf, Registry};
use serde_json::{Map, Value, json};

use crate::protocol::{self, Verb};
use crate::views;

pub struct Generated {
    pub files: Vec<(&'static str, String)>,
    pub summary: Value,
    pub frame_stats: Value,
    pub inspectors: Value,
}

fn registry(world: &World) -> Result<&Registry, String> {
    world
        .get_resource::<Registry>()
        .ok_or_else(|| "the schemas' engine lists no bricks: list the game's bricks".to_owned())
}

fn serialized(schema: schemars::Schema) -> Value {
    serde_json::to_value(schema).expect("schema_serializes")
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
    registry(world)?;
    let mut reply = serialized(schemars::schema_for!(crate::Reply));
    let mut definitions = reply["$defs"].as_object().cloned().unwrap_or_default();
    for schema in [
        serialized(schemars::schema_for!(crate::Step)),
        serialized(schemars::schema_for!(crate::Outcome)),
        serialized(schemars::schema_for!(crate::HostContext)),
        serialized(schemars::schema_for!(crate::results::WatchRecord)),
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
    for (op, schema) in crate::results::result_schemas() {
        let (name, schema) = definition_of(serialized(schema), &mut definitions);
        if name.is_empty() {
            return Err(format!("{op}'s result schema names no type"));
        }
        results.insert(op.to_owned(), json!({ "$ref": format!("#/$defs/{name}") }));
        definitions.insert(name, schema);
    }
    let required: Vec<&String> = results.keys().collect();
    let op_results = json!({
        "type": "object",
        "properties": results,
        "required": required,
    });
    definitions.insert("OpResults".to_owned(), op_results);
    reply["$defs"] = Value::Object(definitions);
    Ok(reply)
}

pub fn verbs(world: &World) -> Result<Vec<(&'static str, Value)>, String> {
    let registry = registry(world)?;
    Ok(Verb::ALL
        .into_iter()
        .map(|verb| (verb.name(), verb_schema(registry, verb)))
        .collect())
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
        "Input": schemars::schema_for!(views::InputView),
        "FrameStats": frame_stats(registry, &mut problems),
    });
    if let Some(all) = schemas.as_object_mut() {
        for verb in Verb::ALL {
            all.insert(verb.name().to_owned(), verb_schema(registry, verb));
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
        sections.insert(section.key.to_owned(), serialized(section.schema()));
    }
    for stats in registry.stats() {
        sections.insert(stats.key.to_owned(), serialized(stats.schema()));
    }
    if let Some(all) = schemas.as_object_mut() {
        all.extend(sections);
    }
    let scene_schema = engine_scene::SceneDocument::schema_json(registry);
    let queries = names(&schemas[Verb::Query.name()], &mut problems);
    let controls = names(&schemas[Verb::Control.name()], &mut problems);
    core_names(
        registry,
        (&queries, &protocol::Query::NAMES),
        "Query::NAMES",
        &mut problems,
    );
    core_names(
        registry,
        (&controls, &protocol::BrowserControl::NAMES),
        "BrowserControl::NAMES",
        &mut problems,
    );
    control_state(registry, &controls, &schemas, &mut problems);
    typed_results(registry, &queries, &controls, &mut problems);
    match serde_json::from_str::<Value>(&scene_schema) {
        Ok(scene) => authored_parity(world, &scene, &schemas, &mut problems),
        Err(error) => problems.push(format!("the scene schema is not JSON: {error}")),
    }
    if !problems.is_empty() {
        return Err(problems.join("\n"));
    }
    let summary = json!({
        "version": protocol::PROTOCOL_VERSION,
        "max_request_bytes": protocol::MAX_REQUEST_BYTES,
        "profile_max_events": engine_profile::MAX_EVENTS,
        "verbs": Verb::names().collect::<Vec<_>>(),
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
        ("web/src/generated/protocol.json", pretty(&summary)? + "\n"),
    ];
    Ok(Generated {
        files,
        summary,
        frame_stats: schemas["FrameStats"].clone(),
        inspectors: schemas,
    })
}

fn core_names(
    registry: &Registry,
    (listed, declared): (&[String], &[&str]),
    constant: &str,
    problems: &mut Vec<String>,
) {
    let core: Vec<&str> = listed
        .iter()
        .map(String::as_str)
        .filter(|name| registry.operation(name).is_none())
        .collect();
    if core != declared {
        problems.push(format!(
            "{constant} (engine_dev protocol.rs) must list the core names the schema has, in order: {core:?}"
        ));
    }
}

fn typed_results(
    registry: &Registry,
    queries: &[String],
    controls: &[String],
    problems: &mut Vec<String>,
) {
    let mut ops: Vec<&str> = Verb::names()
        .filter(|verb| {
            ![Verb::Query.name(), Verb::Control.name()].contains(verb)
                && !crate::results::HOST_VERBS.contains(verb)
        })
        .collect();
    ops.extend(
        queries
            .iter()
            .chain(controls)
            .map(String::as_str)
            .filter(|op| registry.operation(op).is_none()),
    );
    let (typed, untyped) = (
        crate::results::TYPED_RESULTS,
        crate::results::UNTYPED_RESULTS,
    );
    for op in &ops {
        if typed.iter().any(|name| name == op) == untyped.iter().any(|name| name == op) {
            problems.push(format!(
                "{op}'s result is typed (results!) or listed in UNTYPED_RESULTS, not both or neither"
            ));
        }
    }
    for op in typed.iter().chain(untyped) {
        if !ops.iter().any(|name| name == op) {
            problems.push(format!("{op} in results.rs is no verb or core operation"));
        }
    }
    if untyped.windows(2).any(|pair| pair[0] >= pair[1]) {
        problems.push("UNTYPED_RESULTS (engine_dev results.rs) must be sorted and unique".into());
    }
}

fn version_schema() -> Value {
    serialized(schemars::SchemaGenerator::default().subschema_for::<u32>())
}

fn verb_schema(registry: &Registry, verb: Verb) -> Value {
    match verb {
        Verb::Describe => enveloped(verb, schemars::schema_for!(protocol::DescribeRequest)),
        Verb::Inspect => enveloped(verb, schemars::schema_for!(protocol::InspectRequest)),
        Verb::Query => query_schema(registry),
        Verb::Control => control_schema(registry),
        Verb::Capture => enveloped(verb, schemars::schema_for!(protocol::CaptureRequest)),
        Verb::Profile => enveloped(verb, schemars::schema_for!(protocol::ProfileRequest)),
        Verb::Explain => enveloped(verb, schemars::schema_for!(protocol::ExplainQuestion)),
        Verb::Watch => enveloped(verb, schemars::schema_for!(protocol::WatchRequest)),
        Verb::Check => enveloped(verb, schemars::schema_for!(protocol::CheckRequest)),
    }
}

fn marked(verb: Verb, object: &mut Value) {
    let Some(fields) = object.as_object_mut() else {
        return;
    };
    let properties = fields
        .entry("properties")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("properties_object");
    properties.insert(
        "verb".to_owned(),
        json!({ "const": verb.name(), "type": "string" }),
    );
    properties.insert("version".to_owned(), version_schema());
    let mut required = vec![json!("verb"), json!("version")];
    if let Some(Value::Array(theirs)) = fields.remove("required") {
        required.extend(theirs);
    }
    fields.insert("required".to_owned(), Value::Array(required));
}

fn enveloped(verb: Verb, schema: schemars::Schema) -> Value {
    let mut schema = serialized(schema);
    match schema.get_mut("oneOf").and_then(Value::as_array_mut) {
        Some(variants) => {
            for variant in variants {
                marked(verb, variant);
            }
        }
        None => marked(verb, &mut schema),
    }
    if let Some(fields) = schema.as_object_mut() {
        fields.insert("title".to_owned(), json!(verb.name()));
    }
    schema
}

fn split_tagged(variant: &Value) -> Option<(String, Value)> {
    let op = &variant["properties"]["op"];
    let name = op
        .get("const")
        .or_else(|| op["enum"].get(0))
        .and_then(Value::as_str)?
        .to_owned();
    let mut args = variant.clone();
    if let Some(fields) = args.as_object_mut() {
        fields.remove("title");
        if let Some(Value::Object(properties)) = fields.get_mut("properties") {
            properties.remove("op");
        }
        let empty = match fields.get_mut("required") {
            Some(Value::Array(required)) => {
                required.retain(|field| field != "op");
                required.is_empty()
            }
            Some(_) | None => false,
        };
        if empty {
            fields.remove("required");
        }
    }
    Some((name, args))
}

fn named_variant(verb: Verb, name: &str, args: Value) -> Value {
    let mut required = vec![json!("verb"), json!("version"), json!("name")];
    if args["required"]
        .as_array()
        .is_some_and(|fields| !fields.is_empty())
    {
        required.push(json!("args"));
    }
    json!({
        "type": "object",
        "properties": {
            "verb": { "const": verb.name(), "type": "string" },
            "version": version_schema(),
            "name": { "const": name, "type": "string" },
            "args": args,
        },
        "required": required,
        "additionalProperties": false,
    })
}

fn named_schema(registry: &Registry, verb: Verb, core: schemars::Schema) -> Value {
    let control = verb == Verb::Control;
    let mut core = serialized(core);
    let definitions = core["$defs"].as_object().cloned().unwrap_or_default();
    let mut variants: Vec<Value> = core["oneOf"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(split_tagged)
        .map(|(name, args)| named_variant(verb, &name, args))
        .collect();
    variants.extend(
        registry
            .operations()
            .filter(|operation| operation.control == control)
            .map(|operation| named_variant(verb, operation.name, json!({ "type": "object" }))),
    );
    let mut schema = json!({
        "$schema": core["$schema"].take(),
        "title": verb.name(),
        "oneOf": variants,
    });
    if !definitions.is_empty() {
        schema["$defs"] = Value::Object(definitions);
    }
    schema
}

fn query_schema(registry: &Registry) -> Value {
    named_schema(
        registry,
        Verb::Query,
        schemars::schema_for!(protocol::Query),
    )
}

fn control_schema(registry: &Registry) -> Value {
    let mut schema = named_schema(
        registry,
        Verb::Control,
        schemars::schema_for!(protocol::BrowserControl),
    );
    let mut titles = Vec::new();
    let mut sections = Map::new();
    for section in registry.sections() {
        let title = serialized(section.schema())["title"]
            .as_str()
            .unwrap_or(section.key)
            .to_owned();
        sections.insert(
            section.key.to_owned(),
            json!({
                "anyOf": [{ "$ref": format!("#/$defs/{title}") }, { "type": "null" }],
                "default": null,
            }),
        );
        titles.push((title, section.key));
    }
    if let Some(environment) = schema["oneOf"]
        .as_array_mut()
        .into_iter()
        .flatten()
        .find(|variant| variant["properties"]["name"]["const"] == "environment")
        .and_then(|variant| variant["properties"]["args"]["properties"].as_object_mut())
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
                    "title": inspector,
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

fn names(schema: &Value, problems: &mut Vec<String>) -> Vec<String> {
    let variants = schema
        .get("oneOf")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if variants.is_empty() {
        problems.push("named verbs are one of their names".to_owned());
    }
    variants
        .iter()
        .filter_map(|variant| {
            let name = variant["properties"]["name"]["const"].as_str();
            if name.is_none() {
                problems.push(format!("a named variant names nothing: {variant}"));
            }
            name.map(str::to_owned)
        })
        .collect()
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

fn frame_stats(registry: &Registry, problems: &mut Vec<String>) -> Value {
    let mut schema = serialized(schemars::schema_for!(engine_core::FrameStats));
    let mut definitions = schema["$defs"].as_object().cloned().unwrap_or_default();
    let mut properties = schema["properties"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    let mut required = schema["required"].as_array().cloned().unwrap_or_default();
    for stats in registry.stats() {
        let (name, definition) = definition_of(serialized(stats.schema()), &mut definitions);
        let mut definition = definition;
        if let Some(fields) = definition.as_object_mut() {
            fields.remove("title");
        }
        if name.is_empty() {
            problems.push(format!(
                "the {} brick's stats schema names no type",
                stats.brick
            ));
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
    registry: &Registry,
    controls: &[String],
    schemas: &Value,
    problems: &mut Vec<String>,
) {
    let mut state: Vec<(&str, &str)> = protocol::CONTROL_STATE.to_vec();
    state.extend(
        registry
            .operations()
            .filter(|operation| operation.control)
            .map(|operation| (operation.name, "")),
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
    ("along", "structure"),
    ("views", "structure"),
    ("looks", "structure"),
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

fn authored_parity(
    world: &World,
    scene: &Value,
    schemas: &Value,
    problems: &mut Vec<String>,
) {
    let definitions = &scene["$defs"];
    let object = definitions["SceneObject"]["properties"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    let kinds: Vec<(&str, &str)> = world
        .get_resource::<MembersOf<SceneComponents>>()
        .into_iter()
        .flat_map(MembersOf::all)
        .map(|kind| (kind.key(), "structure"))
        .collect();
    let authored = || AUTHORED.iter().chain(&kinds);
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
