use crate::GameHost;
use crate::game::{Game, GameEntry};
use crate::host::{JobId, JobOrder, JobShare};
use engine_core::BrickList;
use engine_scene::SceneFiles;
use engine_settings::js_error as js;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

#[doc(hidden)]
pub struct SceneLoading {
    load: Option<crate::host::SceneLoad>,
}

impl SceneLoading {
    pub fn new(source: &str, bricks: fn(BrickList) -> BrickList) -> Result<Self, JsValue> {
        let kinds = crate::scene_kinds(bricks).map_err(js)?;
        Ok(Self {
            load: Some(crate::host::SceneLoad::new(source, &kinds).map_err(js)?),
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
        Ok(self.loading()?.read(uri, bytes).map_err(js)?.is_some())
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
        self.loading()?.decode(uri, bytes).map_err(js)
    }

    pub fn finish(&mut self) -> Result<engine_scene::SceneFiles, JsValue> {
        let load = self.load.take().ok_or_else(ended)?;
        load.finish().map_err(js)
    }

    fn loading(&mut self) -> Result<&mut crate::host::SceneLoad, JsValue> {
        self.load.as_mut().ok_or_else(ended)
    }
}

fn ended() -> JsValue {
    js(
        crate::Error::new(engine_core::ErrorCode::InvalidRequest, "$")
            .expected(crate::Expected::Present),
    )
}

pub struct PageHost<G: Game> {
    host: GameHost<G>,
    #[cfg(feature = "dev")]
    pixels: Vec<u8>,
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
    pub async fn start(
        canvas: HtmlCanvasElement,
        game: GameEntry<G>,
        scene: Option<(String, SceneFiles)>,
    ) -> Result<Self, JsValue> {
        let bricks = (game.bricks)(BrickList::default());
        let target = engine_render::RenderTarget::Canvas(canvas);
        let mut host = GameHost::launch(target, move |_| bricks, game.build)
            .await
            .map_err(js)?;
        match scene {
            Some((source, files)) => host.open_scene(&source, &files).map_err(js)?,
            None => {
                if let Some(scene) = game.scene {
                    let kinds = engine_scene::SceneKinds::of(host.engine().app.world());
                    let (document, files) = crate::host::SceneLoad::new(scene.text, &kinds)
                        .and_then(crate::host::SceneLoad::opened)
                        .map_err(js)?;
                    host.open_document(&document, &files).map_err(js)?;
                }
            }
        }
        host.prepare_pipelines();
        host.engine_mut().exchange();
        #[cfg(feature = "dev")]
        {
            DECLARED.set(game.scene.map(|scene| scene.name));
            host.engine_mut()
                .app
                .world_mut()
                .insert_resource(engine_dev::record::RecordHost {
                    about,
                    open,
                    instruments: crate::host::instruments(),
                    instrument: crate::host::instrument,
                });
        }
        Ok(Self {
            host,
            #[cfg(feature = "dev")]
            pixels: Vec::new(),
        })
    }

    pub fn animation_frame(
        &mut self,
        timestamp: f64,
        width: u32,
        height: u32,
    ) -> Result<u32, JsValue> {
        Ok(self
            .host
            .animation_frame(timestamp, width, height)
            .map_err(js)? as u32)
    }

    pub fn mouse_button(&mut self, button: u32, down: bool) {
        if let Some(button) = engine_core::MouseButton::of(button) {
            self.host.mouse_button(button, down);
        }
    }

    pub fn pad(&mut self, index: u32, buttons: u32, axes: Vec<f32>) {
        if let Ok(axes) = <[f32; 6]>::try_from(axes) {
            self.host.pad(index, buttons, axes);
        }
    }

    pub fn touch(&mut self, id: u32, x: f32, y: f32, phase: u32) {
        if let Some(phase) = engine_core::TouchPhase::of(phase) {
            self.host.touch(id, x, y, phase);
        }
    }

    pub fn touch_motion(&mut self, id: u32, dx: f32, dy: f32) {
        self.host.touch_motion(id, dx, dy);
    }

    pub fn pause(&mut self, reason: u32) {
        if let Some(pause) = crate::host::HostPause::of(reason) {
            self.host.pause(pause);
        }
    }

    pub fn resume(&mut self, reason: u32) {
        if let Some(pause) = crate::host::HostPause::of(reason) {
            self.host.resume(pause);
        }
    }

    pub fn cursor_mode(&self) -> u32 {
        self.host.cursor_mode() as u32
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

    pub fn job_kept(&mut self, id: f64, key: String, output: Option<js_sys::Uint8Array>) {
        if let Some(id) = job_id(id) {
            self.host.job_kept(id, key, output.as_ref().map(shared));
        }
    }

    pub fn complete_job(&mut self, id: f64, output: js_sys::Uint8Array) {
        if let Some(id) = job_id(id) {
            self.host.complete_job(id, shared(&output));
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
        self.host.reload_scene(source, &files.into()).map_err(js)
    }

    pub fn apply_host_settings(&mut self, query: &str) -> Result<(), JsValue> {
        let parameters = web_sys::UrlSearchParams::new_with_str(query)?;
        #[cfg(feature = "dev")]
        ADDRESSED.set(parameters.get("scene"));
        let mut settings = crate::host::HostSettings::default();
        let overlay = crate::host::OVERLAY;
        for entry in parameters.get_all(overlay).iter() {
            let entry = entry.as_string().unwrap_or_default();
            settings
                .set(&entry)
                .map_err(|error| js(error.within(&format!("?{overlay}"))))?;
        }
        settings.apply(&mut self.host).map_err(js)
    }

    pub fn stats(&self) -> String {
        serde_json::to_string(self.host.frame_times()).expect("stats_serialize")
    }

    pub fn job_hash(&self) -> Vec<u8> {
        self.host.job_list().hash().to_vec()
    }
}

#[cfg(feature = "dev")]
impl<G: Game> PageHost<G> {
    pub fn dev_push(&mut self, line: &str) -> u32 {
        self.host.dev_push(line)
    }

    pub fn dev_take(&mut self) -> String {
        let taken = self.host.dev_take();
        self.pixels.clear();
        let pictures: Vec<serde_json::Value> = taken
            .pictures
            .into_iter()
            .map(|shot| {
                self.pixels.extend_from_slice(&shot.rgba);
                serde_json::json!({
                    "run": shot.run,
                    "file": shot.file,
                    "width": shot.width,
                    "height": shot.height,
                })
            })
            .collect();
        serde_json::json!({
            "replies": taken.replies,
            "pictures": pictures,
            "reload": taken.reload,
        })
        .to_string()
    }

    pub fn dev_pixels(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.pixels)
    }

    pub fn dev_reloaded(&mut self, source: &str, files: Option<impl Into<SceneFiles>>) {
        let files = files.map(Into::into).unwrap_or_default();
        self.host.dev_reloaded(source, &files);
    }

    pub fn dev_reload_failed(&mut self, error: &str) {
        let error = engine_settings::parse_json::<crate::Error>(error)
            .unwrap_or_else(|error| error.within("$.error"));
        self.host.dev_reload_failed(error);
    }
}

#[cfg(feature = "dev")]
thread_local! {
    static DECLARED: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
    static ADDRESSED: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

#[cfg(feature = "dev")]
fn binding(name: &str) -> Option<js_sys::Function> {
    js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str(name))
        .ok()
        .and_then(|value| value.dyn_into::<js_sys::Function>().ok())
}

#[cfg(feature = "dev")]
struct PageSink;

#[cfg(feature = "dev")]
impl engine_dev::record::RunSink for PageSink {
    fn line(&mut self, line: String) -> bool {
        binding("__engineRecord").is_some_and(|record| {
            record
                .call1(&JsValue::NULL, &JsValue::from_str(&line))
                .is_ok()
        })
    }
}

#[cfg(feature = "dev")]
fn open(_: &str, _: usize) -> Result<Box<dyn engine_dev::record::RunSink>, crate::Error> {
    Ok(Box::new(PageSink))
}

#[cfg(feature = "dev")]
fn about(_: &bevy_ecs::world::World) -> serde_json::Map<String, serde_json::Value> {
    use engine_registry::unavailable;
    let scene = ADDRESSED
        .with_borrow(Clone::clone)
        .or_else(|| DECLARED.get().map(str::to_owned))
        .map_or_else(|| unavailable("no_scene"), serde_json::Value::from);
    let features: Vec<&str> = [
        cfg!(feature = "dev").then_some("dev"),
        cfg!(feature = "profiling").then_some("profiling"),
    ]
    .into_iter()
    .flatten()
    .collect();
    serde_json::Map::from_iter([
        ("machine".to_owned(), unavailable("read_by_the_driver")),
        ("host".to_owned(), "browser".into()),
        (
            "build".to_owned(),
            serde_json::json!({"version": env!("CARGO_PKG_VERSION"), "features": features}),
        ),
        ("scene".to_owned(), scene),
        ("picture".to_owned(), unavailable("given_by_each_capture")),
        ("pack".to_owned(), unavailable("not_known_to_the_page")),
    ])
}

fn shared(bytes: &js_sys::Uint8Array) -> std::sync::Arc<[u8]> {
    let mut shared: std::sync::Arc<[u8]> =
        std::iter::repeat_n(0, bytes.length() as usize).collect();
    if let Some(held) = std::sync::Arc::get_mut(&mut shared) {
        bytes.copy_to(held);
    }
    shared
}

fn job_id(id: f64) -> Option<JobId> {
    (id.is_finite() && id >= 1.0).then_some(JobId(id as u64))
}
