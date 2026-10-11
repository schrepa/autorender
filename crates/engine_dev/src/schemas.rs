//! The development protocol's generated files, which `cargo xtask schema`
//! writes and lint keeps current: every inspector's schema, the frame
//! stats' and the scene's, the ops of queries and controls, and the
//! protocol's summary for hosts and tools. Generating them checks the rules
//! that keep what agents see complete: every type and field describes
//! itself, fits one reply, and whatever a scene authors shows in an
//! inspector.

use engine_scene::{EnvironmentSection, SectionVisitor};
use serde_json::{Map, Value, json};

use crate::{protocol, views};

/// The development protocol's generated files, and the values tools
/// generate others from.
pub struct Generated {
    /// Each file by path from the repository root, with its contents.
    pub files: Vec<(&'static str, String)>,
    /// The protocol's summary: its version, request limit, ops, features
    /// and error codes.
    pub summary: Value,
    /// The schema of what a frame observed ([`engine_core::FrameStats`]).
    pub frame_stats: Value,
}

/// The schema of every development reply ([`crate::Reply`]), which the
/// page's reply types are generated from, with the bridge's types beside
/// its own definitions: the steps the engine answers ([`crate::Step`]),
/// the outcomes of the host's effects ([`crate::Outcome`]) and the host's
/// side of each call ([`crate::HostContext`]).
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
        // The reply is the root here, the definitions the others' too.
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
    // Each typed op's result, and one object naming them by op.
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

/// The generated files and values, or every rule the types break.
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
        "protocol": schemars::schema_for!(protocol::DevRequest),
        "BrowserControl": schemars::schema_for!(protocol::BrowserControl),
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
    // Each type's description fits one reply, its envelope and constraints
    // beside it.
    for (name, schema) in described {
        let bytes = schema.to_string().len();
        if bytes + 8 * 1024 > protocol::MAX_RESPONSE_BYTES {
            problems.push(format!(
                "describe {name} would take {bytes} bytes of a {}-byte reply: split the type or page its schema",
                protocol::MAX_RESPONSE_BYTES
            ));
        }
    }
    // Inspectors describe themselves by their types' doc comments.
    for (name, schema) in described {
        if ["protocol", "BrowserControl", "FrameStats"].contains(&name.as_str()) {
            continue;
        }
        let description = schema["description"].as_str().unwrap_or_default();
        if description.is_empty() || description.len() > protocol::bounds::MAX_DESCRIPTION_BYTES {
            problems.push(format!(
                "{name} needs a doc comment of at most {} bytes: inspectors describe themselves",
                protocol::bounds::MAX_DESCRIPTION_BYTES
            ));
        }
    }
    let mut sections = Sections {
        schemas: Map::new(),
        problems: Vec::new(),
    };
    engine_scene::SceneEnvironment::default().visit(&mut sections);
    problems.append(&mut sections.problems);
    // A settings section is never named as the plural of a component, which
    // reads as a store of them: `LocalLight`'s settings are
    // `LocalLightSettings`, not `LocalLights`.
    for name in sections.schemas.keys() {
        if let Some(component) = name.strip_suffix('s')
            && schemas.get(component).is_some()
        {
            problems.push(format!(
                "settings {name} read as a store of {component}s: name them {component}Settings"
            ));
        }
    }
    if let Some(all) = schemas.as_object_mut() {
        all.extend(sections.schemas);
    }
    let scene_schema = engine_scene::SceneDocument::schema_json();
    // The ops of queries and controls, from their schemas: no list to keep.
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
        // Profiling captures, which the agent's bridge requests.
        "profile_max_events": engine_profile::MAX_EVENTS,
        "queries": queries,
        "controls": controls,
        // The engine's features in its bundle's order: each one's name, the
        // plugin adding it and the features it needs.
        "features": engine_core::Feature::ALL
            .iter()
            .map(|feature| json!({
                "name": feature.name(),
                "plugin": feature.plugin(),
                "needs": feature.needs().iter().map(|need| need.name()).collect::<Vec<_>>(),
            }))
            .collect::<Vec<_>>(),
        // Every error code replies answer with and what it means.
        "errors": engine_settings::ErrorCode::ALL
            .iter()
            .map(|&(code, name, _)| (name.to_owned(), json!(code.description())))
            .collect::<Map<_, _>>(),
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
    })
}

/// Checks that every op's result is typed (`results!`) or listed in
/// `UNTYPED_RESULTS`, not both, that both name only ops, and that the
/// untyped list is sorted.
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

/// Each scene environment section's schema, by the name of its inspector,
/// and the fields that lack a doc comment.
struct Sections {
    schemas: Map<String, Value>,
    problems: Vec<String>,
}

impl SectionVisitor for Sections {
    fn section<T: EnvironmentSection>(&mut self, _: &'static str, _: Option<T>) {
        let schema = serde_json::to_value(T::schema()).unwrap_or_default();
        // Agents learn each field from its own doc comment.
        for (field, property) in schema["properties"].as_object().into_iter().flatten() {
            if property["description"].as_str().is_none_or(str::is_empty) {
                self.problems.push(format!(
                    "{}.{field} needs a doc comment: sections describe themselves",
                    T::NAME
                ));
            }
        }
        self.schemas.insert(T::NAME.to_owned(), schema);
    }
}

/// The schema of what a frame observed, after checking the rules that keep it
/// one readable definition: every field has a doc comment, a feature's
/// numbers live under its environment section's key, and what a budget or
/// the device's limits left out is named `dropped` or `dropped_*`.
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
        for (field, property) in object["properties"].as_object().into_iter().flatten() {
            if property["description"].as_str().is_none_or(str::is_empty) {
                problems.push(format!(
                    "{name}.{field} needs a doc comment: frame stats describe themselves"
                ));
            }
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
            // Radiance is a shared numerical boundary, not an optional feature.
            && field != "radiance"
        {
            problems.push(format!(
                "RenderStats.{field} groups a feature's stats under a key that is no environment \
                 section's: name it after the section"
            ));
        }
    }
    schema
}

/// Adds to `problems` each name describe or the hosts' tooling answers
/// that an inspector may take: a probe registered under each must be
/// refused as reserved.
fn reserved_inspectors(problems: &mut Vec<String>) {
    let mut tools = crate::DevTools::default();
    for name in crate::reserved_names() {
        let probe = crate::InspectorDescriptor::new(name, "a probe", json!({"type": "object"}));
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

/// Checks that `dev_features!` lists features by name, sorted.
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

/// Checks that every control names, once, where its state shows, and that
/// the inspectors named exist.
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
        // The environment's inspector is its sections' together.
        if !["", "status", "GameControls", "Environment"].contains(shows)
            && schemas.get(*shows).is_none()
        {
            problems.push(format!(
                "control {op}'s state shows in {shows}, which is no inspector"
            ));
        }
    }
}

/// Where each property scenes author on an entity shows when it is
/// inspected: in every entity reply's header, as an inspector, or as one
/// field of an inspector. A property shown as an object shows each of its
/// fields there. The components engine_scene lists say where they show
/// themselves ([`engine_scene::COMPONENT_SHOWS`]).
const AUTHORED: &[(&str, &str)] = &[
    ("id", "header"),
    ("name", "header"),
    ("tags", "header"),
    ("mesh", "Mesh3d.mesh"),
    ("material", "Mesh3d.material"),
    ("parent", "Hierarchy.authored_parent"),
    ("prefab", "Hierarchy.prefab_id"),
    ("animation", "AnimationPlayer"),
    ("skin", "Skin.skin"),
    ("transform", "Transform"),
];

/// Where each top-level property of a scene shows: the definitions of the
/// assets entities use, shown with them; the environment and camera, as
/// their inspectors; and the document's structure, in the entities and
/// replies' observations.
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
    ("skins", "Skin"),
    ("animations", "AnimationPlayer.clip"),
];

/// Authored fields inspectors show under another name: the scene's form and
/// the runtime's.
const RENAMED: &[(&str, &str)] = &[
    ("SceneTransform.rotation_degrees", "rotation"),
    ("SceneCamera.fov_degrees", "vertical_fov_radians"),
    ("SceneTexture.rgba8", "inline"),
    ("SceneSkin.id", "skin"),
];

/// Checks that whatever a scene authors, an inspector shows: each entity
/// property and top-level property where [`AUTHORED`] and [`DOCUMENT`] say,
/// every field of an object there, under its name or the one [`RENAMED`]
/// gives. A property without an entry fails, so new ones come with their
/// inspection.
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

/// Checks that authored property `what` (of schema `property`) shows where
/// `shows` says, every field of it: `definitions` are the scene's, `schemas`
/// the inspectors'.
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
    // A field holding the scene's own type refers agents to its
    // definition there instead of repeating it.
    let declared = format!("`{name}`");
    if target.get("properties").is_none()
        && target["description"]
            .as_str()
            .is_some_and(|description| description.contains(&declared))
    {
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

/// The definition `schema` names, directly, as an option, or as the items
/// of a list: its name and body.
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

/// The object schema `schema` resolves to through references, options and
/// list items, within `definitions`.
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

/// The `op` of every variant of an internally tagged request enum's schema.
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
