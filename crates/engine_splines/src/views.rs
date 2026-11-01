use serde::Serialize;

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SplineMeshView {
    pub points: Vec<[f32; 3]>,
    pub mesh: Option<String>,
    pub material: Option<String>,
    pub length: Option<f32>,
    pub copies: Option<u32>,
    pub chunks: Option<usize>,
    pub vertices: Option<u32>,
    pub error: Option<engine_settings::Error>,
}
