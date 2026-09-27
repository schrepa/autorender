#[allow(dead_code)]
#[path = "src/protocol.rs"]
mod protocol;
#[allow(dead_code)]
#[path = "src/views.rs"]
mod views;

fn main() {
    println!("cargo:rerun-if-changed=src/views.rs");
    println!("cargo:rerun-if-changed=src/protocol.rs");
    let schemas = serde_json::json!({
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
        "Shadows": schemars::schema_for!(engine_scene::SceneShadows),
        "AmbientOcclusion": schemars::schema_for!(engine_scene::SceneAmbientOcclusion),
        "Bloom": schemars::schema_for!(engine_scene::SceneBloom),
        "Environment": schemars::schema_for!(engine_scene::SceneEnvironment),
        "LocalLights": schemars::schema_for!(engine_scene::SceneLocalLights),
        "SunShafts": schemars::schema_for!(engine_scene::SceneSunShafts),
        "Reflections": schemars::schema_for!(engine_scene::SceneReflections),
        "Particles": schemars::schema_for!(engine_scene::SceneParticles),
        "LocalLight": schemars::schema_for!(engine_scene::SceneLight),
        "ParticleEmitter": schemars::schema_for!(engine_scene::SceneEmitter),
        "protocol": schemars::schema_for!(protocol::DevRequest),
        "BrowserControl": schemars::schema_for!(protocol::BrowserControl),
    });
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
    std::fs::write(output.join("inspectors.json"), schemas.to_string())
        .expect("write generated development schemas");
    std::fs::write(
        output.join("scene.schema.json"),
        engine_scene::SceneDocument::schema_json(),
    )
    .expect("write generated scene development schema");
}
