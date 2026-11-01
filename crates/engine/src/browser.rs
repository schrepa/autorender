use crate::GameHost;
use crate::game::Game;
use crate::host::{JobId, JobOrder, JobShare};
use engine_core::{BrickList, Engine};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

#[doc(hidden)]
pub struct SceneLoading {
    load: Option<crate::host::SceneLoad>,
}

impl SceneLoading {
    pub fn new(source: &str) -> Result<Self, JsValue> {
        Ok(Self {
            load: Some(crate::host::SceneLoad::new(source)?),
        })
    }

    pub fn take_reads(&mut self) -> Result<js_sys::Array, JsValue> {
        let out = js_sys::Array::new();
        for read in self.loading()?.take_reads() {
            let object = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&object, &"uri".into(), &read.uri.into());
            out.push(&object);
        }
        Ok(out)
    }

    pub fn read(&mut self, uri: &str, bytes: Vec<u8>) -> Result<bool, JsValue> {
        Ok(self.loading()?.read(uri, bytes)?.is_some())
    }

    pub fn image(
        &mut self,
        uri: &str,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    ) -> Result<(), JsValue> {
        let image = crate::host::Image {
            width,
            height,
            pixels,
        };
        self.loading()?.image(uri, image);
        Ok(())
    }

    pub fn decode(&mut self, uri: &str, bytes: &[u8]) -> Result<(), JsValue> {
        Ok(self.loading()?.decode(uri, bytes)?)
    }

    pub fn finish(&mut self) -> Result<engine_scene::SceneFiles, JsValue> {
        let load = self.load.take().ok_or_else(ended)?;
        Ok(load.finish()?)
    }

    fn loading(&mut self) -> Result<&mut crate::host::SceneLoad, JsValue> {
        self.load.as_mut().ok_or_else(ended)
    }
}

fn ended() -> JsValue {
    crate::Error::new(engine_core::ErrorCode::InvalidRequest, "$")
        .expected(crate::Expected::Present)
        .into()
}

pub struct PageHost<G: Game> {
    host: GameHost<G>,
}

impl<G: Game> std::ops::Deref for PageHost<G> {
    type Target = GameHost<G>;

    fn deref(&self) -> &GameHost<G> {
        &self.host
    }
}

impl<G: Game> std::ops::DerefMut for PageHost<G> {
    fn deref_mut(&mut self) -> &mut GameHost<G> {
        &mut self.host
    }
}

impl<G: Game> PageHost<G> {
    pub async fn launch<E: Into<crate::Error>>(
        canvas: HtmlCanvasElement,
        bricks: impl FnOnce(BrickList) -> BrickList,
        build: impl FnOnce(&mut Engine) -> Result<G, E>,
    ) -> Result<Self, JsValue> {
        let target = engine_render::RenderTarget::Canvas(canvas);
        Ok(Self {
            host: GameHost::launch(target, bricks, build).await?,
        })
    }

    pub fn frame(&mut self, timestamp: f64, width: u32, height: u32) -> Result<(), JsValue> {
        Ok(self.host.frame(timestamp, width, height)?)
    }

    pub fn animation_frame(
        &mut self,
        timestamp: f64,
        width: u32,
        height: u32,
    ) -> Result<u32, JsValue> {
        Ok(self.host.animation_frame(timestamp, width, height)? as u32)
    }

    pub fn take_jobs(&mut self) -> js_sys::Array {
        let out = js_sys::Array::new();
        let set = |object: &js_sys::Object, field: &str, value: &JsValue| {
            let _ = js_sys::Reflect::set(object, &JsValue::from_str(field), value);
        };
        let bytes = |bytes: &[u8]| JsValue::from(js_sys::Uint8Array::from(bytes));
        let id = |id: JobId| JsValue::from_f64(id.0 as f64);
        for order in self.host.take_jobs() {
            let object = js_sys::Object::new();
            match order {
                JobOrder::Share(JobShare::Clear) => set(&object, "clear", &JsValue::TRUE),
                JobOrder::Share(JobShare::Set { key, bytes: shared }) => {
                    set(&object, "share", &JsValue::from_str(&key.hex()));
                    set(&object, "bytes", &bytes(&shared));
                }
                JobOrder::Share(JobShare::Remove { key }) => {
                    set(&object, "unshare", &JsValue::from_str(&key.hex()));
                }
                JobOrder::Run { worker, request } => {
                    let source: String = request
                        .kind
                        .source()
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect();
                    set(&object, "run", &id(request.id));
                    set(&object, "worker", &JsValue::from(worker));
                    set(&object, "kind", &JsValue::from_str(request.kind.name()));
                    set(&object, "source", &JsValue::from_str(&source));
                    set(&object, "input", &bytes(&request.input));
                    set(&object, "priority", &JsValue::from(request.priority));
                }
                JobOrder::Cancel { worker, id: job } => {
                    set(&object, "cancel", &id(job));
                    set(&object, "worker", &JsValue::from(worker));
                }
                JobOrder::Lookup { id: job, key, .. } => {
                    set(&object, "lookup", &id(job));
                    set(&object, "key", &JsValue::from_str(&key));
                }
                JobOrder::Keep { key, output, .. } => {
                    set(&object, "keep", &JsValue::from_str(&key));
                    set(&object, "output", &bytes(&output));
                }
                JobOrder::Mismatch { key, .. } => {
                    set(&object, "mismatch", &JsValue::from_str(&key));
                }
            }
            out.push(&object);
        }
        out
    }

    pub fn job_kept(&mut self, id: f64, key: String, output: Option<Vec<u8>>) {
        if let Some(id) = job_id(id) {
            self.host.job_kept(id, key, output);
        }
    }

    pub fn complete_job(&mut self, id: f64, output: Vec<u8>) {
        if let Some(id) = job_id(id) {
            self.host.complete_job(id, output);
        }
    }

    pub fn fail_job(&mut self, id: f64, error: String) {
        if let Some(id) = job_id(id) {
            self.host.fail_job(id, error);
        }
    }

    pub fn report_job_cache(&mut self, hits: f64, misses: f64, bytes: f64, mismatches: f64) {
        let count = |value: f64| {
            if value.is_finite() && value > 0.0 {
                value as u64
            } else {
                0
            }
        };
        self.host
            .report_job_cache(count(hits), count(misses), count(bytes), count(mismatches));
    }

    pub fn reload_scene(
        &mut self,
        source: &str,
        files: impl Into<engine_scene::SceneFiles>,
    ) -> Result<(), JsValue> {
        Ok(self.host.reload_scene(source, &files.into())?)
    }

    pub fn set_setting(&mut self, key: &str, value: &str) -> Result<(), JsValue> {
        let value = engine_settings::parse_json::<serde_json::Value>(value)
            .map_err(|error| error.within("$.value"))?;
        Ok(self.host.set_setting(key, value)?)
    }

    pub fn apply_host_settings(&mut self, query: &str) -> Result<(), JsValue> {
        let parameters = web_sys::UrlSearchParams::new_with_str(query)?;
        let mut settings = crate::host::HostSettings::default();
        for setting in crate::host::SETTINGS {
            if let Some(value) = parameters.get(setting.name) {
                settings
                    .set(setting.name, &value)
                    .map_err(|error| error.within(&format!("?{}", setting.name)))?;
            }
        }
        let overlay = crate::host::OVERLAY;
        for entry in parameters.get_all(overlay).iter() {
            let entry = entry.as_string().unwrap_or_default();
            settings
                .set(overlay, &entry)
                .map_err(|error| error.within(&format!("?{overlay}")))?;
        }
        let world = self.host.engine_mut().app.world_mut();
        if let Some(look) = parameters.get("look") {
            crate::host::show_look(world, &look, "?look")?;
        }
        if let Some(view) = parameters.get("view") {
            crate::host::show_view(world, &view, "?view")?;
        }
        settings.apply(&mut self.host)?;
        Ok(())
    }

    pub fn distribution(&self, values: Vec<f64>) -> String {
        serde_json::to_string(&crate::host::Distribution::of(&values))
            .expect("distribution_serializes")
    }
}

#[cfg(feature = "dev")]
impl<G: Game> PageHost<G> {
    pub fn dev_request(&mut self, line: &str, context: &str) -> String {
        let step = match read::<engine_dev::HostContext>(context, "$.context") {
            Ok(context) => self.host.dev_request(line, &context),
            Err(error) => self.unreadable(&error),
        };
        written(&step)
    }

    pub fn dev_resume(&mut self, outcome: &str, context: &str) -> String {
        let step = match (
            read::<engine_dev::Outcome>(outcome, "$.outcome"),
            read::<engine_dev::HostContext>(context, "$.context"),
        ) {
            (Ok(outcome), Ok(context)) => self.host.dev_resume(outcome, &context),
            (Err(error), _) | (_, Err(error)) => self.unreadable(&error),
        };
        written(&step)
    }

    pub fn dev_reloaded(
        &mut self,
        source: &str,
        context: &str,
        files: Option<impl Into<engine_scene::SceneFiles>>,
    ) -> String {
        let files = files.map(Into::into);
        let step = match read::<engine_dev::HostContext>(context, "$.context") {
            Ok(context) => self.host.dev_reloaded(source, files.as_ref(), &context),
            Err(error) => self.unreadable(&error),
        };
        written(&step)
    }

    pub fn dev_refusal(&mut self, error: &str) -> String {
        let error = engine_settings::parse_json::<crate::Error>(error)
            .unwrap_or_else(|error| error.within("$.error"));
        self.host.dev_refusal(&error).to_string()
    }

    fn unreadable(&mut self, error: &crate::Error) -> engine_dev::Step {
        engine_dev::Step::Reply(self.host.dev_refusal(error))
    }
}

#[cfg(feature = "dev")]
fn read<T: serde::de::DeserializeOwned>(json: &str, at: &str) -> crate::Result<T> {
    engine_settings::parse_json(json).map_err(|error| error.within(at))
}

#[cfg(feature = "dev")]
fn written(step: &engine_dev::Step) -> String {
    serde_json::to_string(step).expect("step_serializes")
}

#[cfg(feature = "profiling")]
impl<G: Game> PageHost<G> {
    pub fn profiling_start(&mut self, max_events: u32) -> Result<(), JsValue> {
        Ok(self.host.profiling_start(max_events as usize)?)
    }

    pub fn profiling_stop(&mut self) -> Result<String, JsValue> {
        Ok(self.host.profiling_stop()?.to_string())
    }

    pub fn profiling_report(&mut self) -> String {
        self.host.profiling_report().to_string()
    }

    pub fn profiling_status(&self) -> String {
        self.host.profiling_status().to_string()
    }
}

fn job_id(id: f64) -> Option<JobId> {
    (id.is_finite() && id >= 1.0).then_some(JobId(id as u64))
}
