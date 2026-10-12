use engine::browser::SceneFileSet;
use wasm_bindgen::prelude::*;

use engine::collections::HashMap;

use crate::{StressConfig, StressFile, StressLab, set_view, stress_observation};

/// Files workloads read, by name: the dense scene's virtual meshes,
/// whose prefixes the page fetches, and the highlands'
/// terrain material maps, which it decodes.
#[wasm_bindgen]
#[derive(Default)]
pub struct StressFiles {
    files: HashMap<String, StressFile>,
}

#[wasm_bindgen]
impl StressFiles {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds virtual mesh `name`'s prefix and the URL renderers stream its
    /// other pages from.
    pub fn add_streamed(&mut self, name: &str, bytes: Vec<u8>, url: String) {
        self.files.insert(
            name.to_owned(),
            StressFile {
                bytes,
                url: Some(url),
                size: None,
            },
        );
    }

    /// Adds image `name`: `width` by `height` RGBA8 pixels, rows from the
    /// top.
    pub fn add_image(&mut self, name: &str, width: u32, height: u32, pixels: Vec<u8>) {
        self.files.insert(
            name.to_owned(),
            StressFile {
                bytes: pixels,
                url: None,
                size: Some([width, height]),
            },
        );
    }

    /// The files the dense scene reads, as a JSON array of names.
    pub fn dense_names() -> String {
        serde_json::to_string(&crate::DENSE_FILES).unwrap_or_else(|_| "[]".into())
    }

    /// The images the highlands read, as a JSON array of names.
    pub fn highlands_names() -> String {
        serde_json::to_string(&crate::HIGHLANDS_FILES).unwrap_or_else(|_| "[]".into())
    }

    /// The scanned rock maps the highlands read when present, as a JSON
    /// array of names under `stress/rocks/`; the page adds each found as
    /// `rocks/NAME`.
    pub fn rock_names() -> String {
        serde_json::to_string(&crate::ROCK_FILES).unwrap_or_else(|_| "[]".into())
    }
}

engine::export_game!(
    /// The Stress Lab: a deterministic workload or an authored scene.
    pub struct StressGame(StressLab);
);

#[wasm_bindgen]
impl StressGame {
    /// The workload `config_json` configures, with the `files` it reads.
    pub async fn create(
        canvas: web_sys::HtmlCanvasElement,
        config_json: &str,
        files: Option<StressFiles>,
    ) -> Result<StressGame, JsValue> {
        let config = StressConfig::from_json(config_json).map_err(js_error)?;
        let files = files.map(|files| files.files).unwrap_or_default();
        Self::launch(canvas, move || StressLab::workload(config, files)).await
    }

    /// The authored scene `source` in place of a workload, with the image
    /// and buffer `files` it references, on an engine with every feature but
    /// the comma-separated ones `without` names (`terrain`, `navigation`...).
    pub async fn create_scene(
        canvas: web_sys::HtmlCanvasElement,
        source: &str,
        files: SceneFileSet,
        without: &str,
    ) -> Result<StressGame, JsValue> {
        let without = without
            .split(',')
            .filter(|name| !name.is_empty())
            .map(|name| {
                engine::Feature::from_name(name)
                    .ok_or_else(|| JsValue::from_str(&format!("no feature is named {name}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::launch(canvas, move || {
            StressLab::authored(source, files.files(), &without)
        })
        .await
    }

    /// The authored scene's ID, revision, frame, player position and asset
    /// counts, as a JSON object (`null` for workloads).
    pub fn scene_progress(&self) -> String {
        let host = self.host();
        host.game().scene_progress(host.engine()).to_string()
    }

    /// Builds the workload `config_json` configures, then runs it in place
    /// of the current one at a frame boundary.
    pub fn configure(
        &mut self,
        config_json: &str,
        files: Option<StressFiles>,
    ) -> Result<(), JsValue> {
        let config = StressConfig::from_json(config_json).map_err(js_error)?;
        let files = files.map(|files| files.files).unwrap_or_default();
        let host = self.host_mut();
        let engine = host.game_mut().configure(config, files).map_err(js_error)?;
        host.replace_engine(engine);
        let (engine, lab) = host.parts_mut();
        lab.configured(engine).map_err(js_error)
    }

    /// Rendering effects: `all`, `none`, or comma-separated `no-<effect>`
    /// exclusions.
    pub fn set_effects(&mut self, name: &str) -> Result<(), JsValue> {
        let (engine, lab) = self.host_mut().parts_mut();
        lab.set_effects(engine, name).map_err(js_error)
    }

    /// Lighting preset: `scene` (the scene's own), `golden`, `dusk`,
    /// `night` or `day`.
    pub fn set_look(&mut self, name: &str) -> Result<(), JsValue> {
        let (engine, lab) = self.host_mut().parts_mut();
        lab.set_look(engine, name).map_err(js_error)
    }

    /// The workload's snapshot, as JSON.
    pub fn stress_stats(&self) -> String {
        let host = self.host();
        serde_json::to_string(&stress_observation(host.engine(), host.game().generation()))
            .expect("snapshots serialize")
    }

    /// [`crate::stress_sample`]: a recording's per-frame workload numbers.
    pub fn frame_sample(&self) -> Vec<f64> {
        crate::stress_sample(self.host().engine()).values()
    }

    pub fn set_view(&mut self, view: &str) -> Result<(), JsValue> {
        set_view(self.host_mut().engine_mut(), view).map_err(js_error)
    }

    pub fn set_culling(&mut self, enabled: bool) {
        let engine = self.host_mut().engine_mut();
        engine.set_frustum_culling(enabled);
        if let Some(mut config) = engine.app.world_mut().get_resource_mut::<StressConfig>() {
            config.culling = enabled;
        }
    }

    /// Whether instances draw at levels of detail (on by default) or always
    /// as their mesh, in this and later workloads.
    pub fn set_level_of_detail(&mut self, enabled: bool) {
        let (engine, lab) = self.host_mut().parts_mut();
        lab.set_level_of_detail(engine, enabled);
    }

    /// Whether draw groups draw impostors where they fit (on by default) or
    /// only their meshes' levels, in this and later workloads.
    pub fn set_impostors(&mut self, enabled: bool) {
        let (engine, lab) = self.host_mut().parts_mut();
        lab.set_impostors(engine, enabled);
    }

    /// Whether virtual geometry shades through the visibility buffer or as
    /// it draws (the default), in this and later workloads.
    pub fn set_visibility_buffer(&mut self, enabled: bool) {
        let (engine, lab) = self.host_mut().parts_mut();
        lab.set_visibility_buffer(engine, enabled);
    }
}

fn js_error(message: String) -> JsValue {
    JsValue::from_str(&message)
}
