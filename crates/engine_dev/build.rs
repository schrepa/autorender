#[allow(dead_code)]
#[path = "src/protocol.rs"]
mod protocol;
#[allow(dead_code)]
#[path = "src/views.rs"]
mod views;

/// Each scene environment section's schema, by the name of its inspector.
struct Sections(serde_json::Map<String, serde_json::Value>);

impl engine_scene::SectionVisitor for Sections {
    fn section<T: engine_scene::EnvironmentSection>(&mut self, _: &'static str, _: Option<T>) {
        let schema = serde_json::to_value(T::schema()).expect("schema serializes");
        // Agents learn each field from its own doc comment.
        for (field, property) in schema["properties"].as_object().into_iter().flatten() {
            if property["description"].as_str().is_none_or(str::is_empty) {
                panic!(
                    "{}.{field} needs a doc comment: sections describe themselves",
                    T::NAME
                );
            }
        }
        self.0.insert(T::NAME.to_owned(), schema);
    }
}

fn main() {
    println!("cargo:rerun-if-changed=src/views.rs");
    println!("cargo:rerun-if-changed=src/protocol.rs");
    let mut schemas = serde_json::json!({
        "Transform": schemars::schema_for!(views::TransformView),
        "GlobalTransform": schemars::schema_for!(views::GlobalTransformView),
        "InterpolateTransform": schemars::schema_for!(views::InterpolateTransformView),
        "PresentedTransform": schemars::schema_for!(views::PresentedTransformView),
        "PresentationState": schemars::schema_for!(views::PresentationStateView),
        "Hierarchy": schemars::schema_for!(views::HierarchyView),
        "Camera": schemars::schema_for!(views::CameraView),
        "Mesh3d": schemars::schema_for!(views::Mesh3dView),
        "Collider": schemars::schema_for!(views::ColliderView),
        "CollisionWorld": schemars::schema_for!(views::CollisionWorldView),
        "Atmosphere": schemars::schema_for!(engine_scene::SceneAtmosphere),
        "Decal": schemars::schema_for!(views::DecalView),
        "SplineMesh": schemars::schema_for!(views::SplineMeshView),
        "ReflectionProbe": schemars::schema_for!(engine_scene::SceneReflectionProbe),
        "Terrain": schemars::schema_for!(views::TerrainView),
        "Terrains": schemars::schema_for!(views::TerrainsView),
        "TerrainDebug": schemars::schema_for!(views::TerrainDebugView),
        "Input": schemars::schema_for!(views::InputView),
        "LocalLight": schemars::schema_for!(engine_scene::SceneLight),
        "ParticleEmitter": schemars::schema_for!(engine_scene::SceneEmitter),
        "Tint": schemars::schema_for!(views::TintView),
        "Floating": schemars::schema_for!(engine_core::Floating),
        "Deformer": schemars::schema_for!(engine_core::Deformer),
        "Tracks": schemars::schema_for!(views::TracksView),
        "WorldCells": schemars::schema_for!(views::WorldCellsView),
        "Grounded": schemars::schema_for!(views::GroundedView),
        "TerrainPrefetch": schemars::schema_for!(engine_core::TerrainPrefetch),
        "WorldChanges": schemars::schema_for!(views::WorldChangesView),
        "StreamedId": schemars::schema_for!(views::StreamedIdView),
        "NavAgent": schemars::schema_for!(views::NavAgentView),
        "NavAgents": schemars::schema_for!(views::NavAgentsView),
        "Jobs": schemars::schema_for!(views::JobsView),
        "AnimationPlayer": schemars::schema_for!(views::AnimationPlayerView),
        "Skin": schemars::schema_for!(views::SkinView),
        "protocol": schemars::schema_for!(protocol::DevRequest),
        "BrowserControl": schemars::schema_for!(protocol::BrowserControl),
        "FrameStats": frame_stats(),
    });
    // Each type's description fits one reply, its envelope and constraints
    // beside it.
    for (name, schema) in schemas.as_object().expect("schemas are an object") {
        let bytes = schema.to_string().len();
        if bytes + 8 * 1024 > protocol::MAX_RESPONSE_BYTES {
            panic!(
                "describe {name} would take {bytes} bytes of a {}-byte reply: split the type or page its schema",
                protocol::MAX_RESPONSE_BYTES
            );
        }
    }
    // Inspectors describe themselves by their types' doc comments.
    for (name, schema) in schemas.as_object().expect("schemas are an object") {
        if ["protocol", "BrowserControl", "FrameStats"].contains(&name.as_str()) {
            continue;
        }
        let description = schema["description"].as_str().unwrap_or_default();
        if description.is_empty() || description.len() > protocol::bounds::MAX_DESCRIPTION_BYTES {
            panic!(
                "{name} needs a doc comment of at most {} bytes: inspectors describe themselves",
                protocol::bounds::MAX_DESCRIPTION_BYTES
            );
        }
    }
    let mut sections = Sections(serde_json::Map::new());
    engine_scene::SceneEnvironment::default().visit(&mut sections);
    // A settings section is never named as the plural of a component, which
    // reads as a store of them: `LocalLight`'s settings are
    // `LocalLightSettings`, not `LocalLights`.
    for name in sections.0.keys() {
        if let Some(component) = name.strip_suffix('s')
            && schemas.get(component).is_some()
        {
            panic!(
                "settings {name} read as a store of {component}s: name them {component}Settings"
            );
        }
    }
    schemas
        .as_object_mut()
        .expect("schemas are an object")
        .extend(sections.0);
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
    std::fs::write(output.join("inspectors.json"), schemas.to_string())
        .expect("write generated development schemas");
    std::fs::write(
        output.join("frame_stats.schema.json"),
        serde_json::to_string_pretty(&schemas["FrameStats"]).expect("schema serializes"),
    )
    .expect("write generated frame stats schema");
    std::fs::write(
        output.join("scene.schema.json"),
        engine_scene::SceneDocument::schema_json(),
    )
    .expect("write generated scene development schema");
    // The ops of queries and controls, from their schemas: no list to keep.
    let queries = ops(&schemas["protocol"]);
    let controls = ops(&schemas["BrowserControl"]);
    control_state(&controls, &schemas);
    let scene: serde_json::Value =
        serde_json::from_str(&engine_scene::SceneDocument::schema_json()).expect("scene schema");
    authored_parity(&scene, &schemas);
    let constant = |name: &str, what: &str, ops: &[String]| {
        format!(
            "/// The op of every {what}, as requests name it.
pub const {name}: [&str; {}] = {:?};
",
            ops.len(),
            ops
        )
    };
    std::fs::write(
        output.join("ops.rs"),
        constant("QUERY_OPS", "query", &queries) + &constant("CONTROL_OPS", "control", &controls),
    )
    .expect("write generated op lists");
    let summary = serde_json::json!({
        "version": protocol::PROTOCOL_VERSION,
        "max_request_bytes": protocol::MAX_REQUEST_BYTES,
        // Profiling captures, which the agent's bridge requests.
        "profile_max_events": engine_profile::MAX_EVENTS,
        "queries": queries,
        "controls": controls,
        // Every error code replies answer with and what it means.
        "errors": engine_settings::ErrorCode::ALL
            .iter()
            .map(|&(code, name, _)| (name.to_owned(), serde_json::json!(code.description())))
            .collect::<serde_json::Map<_, _>>(),
    });
    std::fs::write(
        output.join("protocol.json"),
        serde_json::to_string_pretty(&summary).expect("summary serializes")
            + "
",
    )
    .expect("write generated protocol summary");
}

/// The schema of what a frame observed, after checking the rules that keep it
/// one readable definition: every field has a doc comment, a feature's
/// numbers live under its environment section's key, and what a budget or
/// the device's limits left out is named `dropped` or `dropped_*`.
fn frame_stats() -> serde_json::Value {
    let schema = serde_json::to_value(schemars::schema_for!(engine_core::FrameStats))
        .expect("schema serializes");
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
                panic!("{name}.{field} needs a doc comment: frame stats describe themselves");
            }
            if field.contains("dropped") && field != "dropped" && !field.starts_with("dropped_") {
                panic!("{name}.{field}: name what budgets drop `dropped` or `dropped_*`");
            }
        }
    }
    let render = serde_json::to_value(schemars::schema_for!(engine_core::RenderStats))
        .expect("schema serializes");
    for (field, property) in render["properties"].as_object().into_iter().flatten() {
        if property.get("$ref").is_some()
            && !engine_core::Environment::KEYS.contains(&field.as_str())
        {
            panic!(
                "RenderStats.{field} groups a feature's stats under a key that is no environment \
                 section's: name it after the section"
            );
        }
    }
    schema
}

/// Checks that every control names, once, where its state shows, and that
/// the inspectors named exist.
fn control_state(controls: &[String], schemas: &serde_json::Value) {
    for op in controls {
        let listed = protocol::CONTROL_STATE
            .iter()
            .filter(|(listed, _)| listed == op)
            .count();
        if listed != 1 {
            panic!("control {op} must name where its state shows once in CONTROL_STATE");
        }
    }
    for (op, shows) in protocol::CONTROL_STATE {
        if !controls.iter().any(|control| control == op) {
            panic!("CONTROL_STATE lists {op}, which is no control");
        }
        // The environment's inspector is its sections' together.
        if !["", "status", "GameControls", "Environment"].contains(shows)
            && schemas.get(*shows).is_none()
        {
            panic!("control {op}'s state shows in {shows}, which is no inspector");
        }
    }
}

/// Where each property scenes author on an entity shows when it is
/// inspected: in every entity reply's header, as an inspector, or as one
/// field of an inspector. A property shown as an object shows each of its
/// fields there.
const AUTHORED: &[(&str, &str)] = &[
    ("id", "header"),
    ("name", "header"),
    ("tags", "header"),
    ("mesh", "Mesh3d.mesh"),
    ("material", "Mesh3d.material"),
    ("parent", "Hierarchy.authored_parent"),
    ("prefab", "Hierarchy.prefab_id"),
    ("collider", "Collider"),
    ("light", "LocalLight"),
    ("particles", "ParticleEmitter"),
    ("tint", "Tint"),
    ("animation", "AnimationPlayer"),
    ("skin", "Skin.skin"),
    ("decal", "Decal"),
    ("reflection_probe", "ReflectionProbe"),
    ("terrain", "Terrain.terrain"),
    ("spline", "SplineMesh"),
    ("floating", "Floating"),
    ("deformer", "Deformer"),
    ("nav_agent", "NavAgent.agent"),
    ("grounded", "Grounded"),
    ("prefetch", "TerrainPrefetch"),
    ("transform", "Transform"),
];

/// Where each top-level property of a scene shows: the definitions of the
/// assets entities use, shown with them; the environment and camera, as
/// their inspectors; and the document's structure, in the entities and
/// replies' observations.
const DOCUMENT: &[(&str, &str)] = &[
    ("version", "structure"),
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
fn authored_parity(scene: &serde_json::Value, schemas: &serde_json::Value) {
    let definitions = &scene["$defs"];
    let check = |what: &str, property: &serde_json::Value, shows: &str| {
        if ["header", "structure", "Environment"].contains(&shows) {
            return;
        }
        let (inspector, field) = shows.split_once('.').unwrap_or((shows, ""));
        let root = schemas
            .get(inspector)
            .unwrap_or_else(|| panic!("{what} shows in {inspector}, which is no inspector"));
        let mut target = root;
        if !field.is_empty() {
            target = resolve(&root["$defs"], &root["properties"][field])
                .unwrap_or_else(|| panic!("{what} shows in {shows}, which is no field"));
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
                panic!(
                    "{what}.{field} ({name}) is authored but {shows} does not show it: add a field named {shown}, or rename it in RENAMED"
                );
            }
        }
    };
    let object = &definitions["SceneObject"]["properties"];
    for (property, schema) in object.as_object().expect("entities have properties") {
        let shows = AUTHORED
            .iter()
            .find(|(authored, _)| authored == property)
            .unwrap_or_else(|| panic!("entity property {property} needs a place in AUTHORED"))
            .1;
        check(&format!("entities[].{property}"), schema, shows);
    }
    for (property, shows) in AUTHORED {
        if object.get(*property).is_none() {
            panic!("AUTHORED lists {property}, which entities do not author");
        }
        let _ = shows;
    }
    let document = &scene["properties"];
    for (property, schema) in document.as_object().expect("documents have properties") {
        let shows = DOCUMENT
            .iter()
            .find(|(authored, _)| authored == property)
            .unwrap_or_else(|| panic!("scene property {property} needs a place in DOCUMENT"))
            .1;
        check(property, schema, shows);
    }
}

/// The definition `schema` names, directly, as an option, or as the items
/// of a list: its name and body.
fn definition<'a>(
    definitions: &'a serde_json::Value,
    schema: &'a serde_json::Value,
) -> Option<(String, &'a serde_json::Value)> {
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
fn resolve<'a>(
    definitions: &'a serde_json::Value,
    schema: &'a serde_json::Value,
) -> Option<&'a serde_json::Value> {
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
fn ops(schema: &serde_json::Value) -> Vec<String> {
    let variants = schema
        .get("oneOf")
        .or_else(|| schema.get("anyOf"))
        .and_then(serde_json::Value::as_array)
        .expect("request enums are one of their variants");
    variants
        .iter()
        .map(|variant| {
            let op = &variant["properties"]["op"];
            op.get("const")
                .or_else(|| op["enum"].get(0))
                .and_then(serde_json::Value::as_str)
                .unwrap_or_else(|| panic!("a request variant names no op: {variant}"))
                .to_owned()
        })
        .collect()
}
