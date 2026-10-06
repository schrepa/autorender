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
        "WorldChanges": schemars::schema_for!(views::WorldChangesView),
        "StreamedId": schemars::schema_for!(views::StreamedIdView),
        "NavAgent": schemars::schema_for!(views::NavAgentView),
        "NavAgents": schemars::schema_for!(views::NavAgentsView),
        "AnimationPlayer": schemars::schema_for!(views::AnimationPlayerView),
        "Skin": schemars::schema_for!(views::SkinView),
        "protocol": schemars::schema_for!(protocol::DevRequest),
        "BrowserControl": schemars::schema_for!(protocol::BrowserControl),
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
        output.join("scene.schema.json"),
        engine_scene::SceneDocument::schema_json(),
    )
    .expect("write generated scene development schema");
}
