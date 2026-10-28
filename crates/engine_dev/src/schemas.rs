use engine_scene::{EnvironmentSection, SectionVisitor};
use serde_json::{Map, Value, json};

use crate::{protocol, views};

pub struct Generated {
    pub files: Vec<(&'static str, String)>,
    pub summary: Value,
    pub frame_stats: Value,
    pub inspectors: Value,
}

pub fn reply() -> Value {
    let serialized = |schema: schemars::Schema| -> Value {
        serde_json::to_value(schema).expect("schemas serialize")
    };
    let mut reply = serialized(schemars::schema_for!(crate::Reply));
    let mut definitions = reply["$defs"].as_object().cloned().unwrap_or_default();
    for mut schema in [
        serialized(schemars::schema_for!(crate::Step)),
        serialized(schemars::schema_for!(crate::Outcome)),
        serialized(schemars::schema_for!(crate::HostContext)),
    ] {
        let Some(fields) = schema.as_object_mut() else {
            continue;
        };
        if let Some(Value::Object(theirs)) = fields.remove("$defs") {
            for (name, definition) in theirs {
                if name != "Reply" {
                    definitions.entry(name).or_insert(definition);
                }
            }
        }
        fields.remove("$schema");
        let name = fields["title"]
            .as_str()
            .expect("root schemas name their type")
            .to_owned();
        definitions.insert(name, schema);
    }
    let mut results = Map::new();
    for (op, schema) in crate::results::result_schemas() {
        let mut schema = serialized(schema);
        let Some(fields) = schema.as_object_mut() else {
            continue;
        };
        if let Some(Value::Object(theirs)) = fields.remove("$defs") {
            for (name, definition) in theirs {
                definitions.entry(name).or_insert(definition);
            }
        }
        fields.remove("$schema");
        let name = fields["title"]
            .as_str()
            .expect("root schemas name their type")
            .to_owned();
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
    reply
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub fn generated() -> Result<Generated, String> {
    let mut problems = Vec::new();
    let mut schemas = json!({
        "Transform": schemars::schema_for!(views::TransformView),
        "GlobalTransform": schemars::schema_for!(views::GlobalTransformView),
        "InterpolateTransform": schemars::schema_for!(views::InterpolateTransformView),
        "PresentedTransform": schemars::schema_for!(views::PresentedTransformView),
        "PresentationState": schemars::schema_for!(views::PresentationStateView),
        "Hierarchy": schemars::schema_for!(views::HierarchyView),
        "Camera": schemars::schema_for!(views::CameraView),
        "Mesh3d": schemars::schema_for!(views::Mesh3dView),
        "Atmosphere": schemars::schema_for!(engine_scene::SceneAtmosphere),
        "Decal": schemars::schema_for!(views::DecalView),
        "SplineMesh": schemars::schema_for!(views::SplineMeshView),
        "ReflectionProbe": schemars::schema_for!(engine_scene::SceneReflectionProbe),
        "Input": schemars::schema_for!(views::InputView),
        "LocalLight": schemars::schema_for!(engine_scene::SceneLight),
        "ParticleEmitter": schemars::schema_for!(engine_scene::SceneEmitter),
        "Tint": schemars::schema_for!(views::TintView),
        "Grounded": schemars::schema_for!(views::GroundedView),
        "Jobs": schemars::schema_for!(views::JobsView),
        "StaticTables": schemars::schema_for!(views::StaticTablesView),
        "protocol": schemars::schema_for!(protocol::DevRequest),
        "BrowserControl": control_schema(),
        "FrameStats": frame_stats(&mut problems),
    });
    if let Some(all) = schemas.as_object_mut() {
        crate::features::schemas(all);
    }
    dev_features(&mut problems);
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
    let mut sections = Sections(Map::new());
    engine_scene::SceneEnvironment::default().visit(&mut sections);
    for name in sections.0.keys() {
        if let Some(component) = name.strip_suffix('s')
            && schemas.get(component).is_some()
        {
            problems.push(format!(
                "settings {name} read as a store of {component}s: name them {component}Settings"
            ));
        }
    }
    if let Some(all) = schemas.as_object_mut() {
        all.extend(sections.0);
    }
    let scene_schema = engine_scene::SceneDocument::schema_json();
    let queries = ops(&schemas["protocol"], &mut problems);
    let controls = ops(&schemas["BrowserControl"], &mut problems);
    control_state(&controls, &schemas, &mut problems);
    typed_results(&queries, &controls, &mut problems);
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
        "features": engine_core::Feature::ALL
            .iter()
            .map(|feature| json!({
                "name": feature.name(),
                "plugin": feature.plugin(),
                "needs": feature.needs().iter().map(|need| need.name()).collect::<Vec<_>>(),
            }))
            .collect::<Vec<_>>(),
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

fn typed_results(queries: &[String], controls: &[String], problems: &mut Vec<String>) {
    let ops: Vec<&str> = queries.iter().chain(controls).map(String::as_str).collect();
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

fn control_schema() -> Value {
    struct Titles(Vec<(String, &'static str)>);
    impl SectionVisitor for Titles {
        fn section<T: EnvironmentSection>(&mut self, _: &'static str, _: Option<T>) {
            let schema = serde_json::to_value(T::schema()).unwrap_or_default();
            if let Some(title) = schema["title"].as_str() {
                self.0.push((title.to_owned(), T::NAME));
            }
        }
    }
    let mut schema =
        serde_json::to_value(schemars::schema_for!(protocol::BrowserControl)).unwrap_or_default();
    let mut titles = Titles(vec![("SceneAtmosphere".to_owned(), "Atmosphere")]);
    engine_scene::SceneEnvironment::default().visit(&mut titles);
    let Some(definitions) = schema["$defs"].as_object_mut() else {
        return schema;
    };
    for (title, inspector) in titles.0 {
        if let Some(definition) = definitions.get_mut(&title) {
            *definition = json!({
                "type": "object",
                "description": format!("The {inspector} inspector describes its fields."),
            });
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

struct Sections(Map<String, Value>);

impl SectionVisitor for Sections {
    fn section<T: EnvironmentSection>(&mut self, _: &'static str, _: Option<T>) {
        let schema = serde_json::to_value(T::schema()).unwrap_or_default();
        self.0.insert(T::NAME.to_owned(), schema);
    }
}

fn frame_stats(problems: &mut Vec<String>) -> Value {
    let schema =
        serde_json::to_value(schemars::schema_for!(engine_core::FrameStats)).unwrap_or_default();
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
    let render =
        serde_json::to_value(schemars::schema_for!(engine_core::RenderStats)).unwrap_or_default();
    for (field, property) in render["properties"].as_object().into_iter().flatten() {
        if property.get("$ref").is_some()
            && !engine_core::Environment::KEYS.contains(&field.as_str())
        {
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

fn dev_features(problems: &mut Vec<String>) {
    let names = crate::features::NAMES;
    for name in names {
        if !engine_core::Feature::ALL
            .iter()
            .any(|feature| feature.name() == *name)
        {
            problems.push(format!(
                "dev_features! lists {name}, which is no feature: name it as engine_features! does"
            ));
        }
    }
    if !names.windows(2).all(|pair| pair[0] < pair[1]) {
        problems.push("dev_features! lists its features sorted, each once".to_owned());
    }
}

fn control_state(controls: &[String], schemas: &Value, problems: &mut Vec<String>) {
    for op in controls {
        let listed = protocol::CONTROL_STATE
            .iter()
            .filter(|(listed, _)| listed == op)
            .count();
        if listed != 1 {
            problems.push(format!(
                "control {op} must name where its state shows once in CONTROL_STATE"
            ));
        }
    }
    for (op, shows) in protocol::CONTROL_STATE {
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
