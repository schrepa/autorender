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
        "LocalLight": schemars::schema_for!(engine_scene::SceneLight),
        "ParticleEmitter": schemars::schema_for!(engine_scene::SceneEmitter),
        "Tint": schemars::schema_for!([f32; 4]),
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
    let mut sections = Sections(serde_json::Map::new());
    engine_scene::SceneEnvironment::default().visit(&mut sections);
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
        "queries": queries,
        "controls": controls,
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
