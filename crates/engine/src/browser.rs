//! The page's side of a hosted game: the files a scene references, decoded
//! by the page, and [`PageHost`], a [`GameHost`] with the page's types,
//! which [`crate::export_game!`] exports.
use crate::GameHost;
use crate::game::Game;
use crate::host::JobMessage;
use engine_core::{Engine, InputState, JobId, JobShare};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

/// Decoded images and buffers of the files a scene references,
/// filled by the page before the scene loads.
#[wasm_bindgen]
#[derive(Default)]
pub struct SceneFileSet {
    files: engine_scene::SceneFiles,
}

#[wasm_bindgen]
impl SceneFileSet {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds image `uri`, decoded to `width` × `height` RGBA8 sRGB pixels.
    pub fn add_image(&mut self, uri: &str, width: u32, height: u32, pixels: Vec<u8>) {
        self.files.add_image(uri, width, height, pixels);
    }

    /// Adds image `uri` transcoded to block-compressed `format` (`bc7`,
    /// `etc2` or `astc`): `width` × `height`, its levels
    /// concatenated in `data` from the base down with their byte lengths in
    /// `levels`.
    pub fn add_compressed_image(
        &mut self,
        uri: &str,
        format: &str,
        width: u32,
        height: u32,
        data: Vec<u8>,
        levels: Vec<u32>,
    ) -> Result<(), JsValue> {
        let format = match format {
            "bc7" => engine_core::TextureFormat::Bc7,
            "etc2" => engine_core::TextureFormat::Etc2Rgba8,
            "astc" => engine_core::TextureFormat::Astc4x4,
            _ => {
                return Err(JsValue::from_str(
                    "compressed formats are bc7, etc2 and astc",
                ));
            }
        };
        let mut rest = &data[..];
        let mut split = Vec::with_capacity(levels.len());
        for length in levels {
            let Some((level, tail)) = rest.split_at_checked(length as usize) else {
                return Err(JsValue::from_str("level lengths exceed the data"));
            };
            split.push(level.to_vec());
            rest = tail;
        }
        if !rest.is_empty() {
            return Err(JsValue::from_str("data exceeds the level lengths"));
        }
        self.files
            .add_compressed_image(uri, format, width, height, split);
        Ok(())
    }

    /// Adds buffer `uri`'s bytes.
    pub fn add_buffer(&mut self, uri: &str, bytes: Vec<u8>) {
        self.files.add_buffer(uri, bytes);
    }

    /// Adds virtual mesh `uri`: its prefix
    /// ([`virtual_prefix_bytes`] long) and the URL renderers stream its
    /// other pages from.
    pub fn add_virtual_mesh(&mut self, uri: &str, bytes: Vec<u8>, url: Option<String>) {
        self.files.add_virtual_mesh(uri, bytes, url.as_deref());
    }

    /// Adds terrain file `uri`: its prefix
    /// ([`terrain_prefix_bytes`] long) and the URL its tiles stream from.
    pub fn add_terrain(&mut self, uri: &str, bytes: Vec<u8>, url: Option<String>) {
        self.files.add_terrain(uri, bytes, url.as_deref());
    }
}

impl SceneFileSet {
    pub fn files(&self) -> &engine_scene::SceneFiles {
        &self.files
    }
}

/// The files scene `source` references, as JSON `[{"uri", "kind"}]` with
/// kind `image`, `buffer`, `virtual_mesh` or `terrain`, for the page to
/// fetch; an unparsable source references none (loading reports its error).
#[wasm_bindgen]
pub fn scene_file_list(source: &str) -> String {
    let files = engine_scene::SceneDocument::from_json(source)
        .map(|document| document.files())
        .unwrap_or_default();
    // Validated URIs hold only letters, digits and `_./-`: no escaping.
    let entries: Vec<String> = files
        .iter()
        .map(|file| {
            let kind = match file.kind {
                engine_scene::SceneFileKind::Image => "image",
                engine_scene::SceneFileKind::Buffer => "buffer",
                engine_scene::SceneFileKind::VirtualMesh => "virtual_mesh",
                engine_scene::SceneFileKind::Terrain => "terrain",
            };
            format!(r#"{{"uri":"{}","kind":"{kind}"}}"#, file.uri)
        })
        .collect();
    format!("[{}]", entries.join(","))
}

/// Bytes of the prefix of the virtual mesh file whose first
/// 128 bytes are `header`: what hosts fetch before a scene loads; renderers
/// stream the rest. `None` when it is not a virtual mesh file of this
/// engine's version.
#[wasm_bindgen]
pub fn virtual_prefix_bytes(header: &[u8]) -> Option<u32> {
    engine_core::VirtualMesh::prefix_bytes(header).map(|bytes| bytes as u32)
}

/// Bytes of the prefix (header and directory) of the terrain file
/// whose first 64 bytes are `header`: what hosts fetch before a scene
/// loads; tiles stream by byte range as the camera needs them. `None` when
/// it is not a terrain file.
#[wasm_bindgen]
pub fn terrain_prefix_bytes(header: &[u8]) -> Option<u32> {
    engine_core::FileHeader::prefix_bytes(header)
        .ok()
        .map(|bytes| bytes as u32)
}

/// A hosted game as the page calls it: the game surface's methods whose
/// types are the page's convert here; the others are [`GameHost`]'s own,
/// which it dereferences to.
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
    /// Starts the renderer of `canvas` and, while the browser compiles its
    /// pipelines, the engine and game `build` makes.
    pub async fn launch<E: Into<crate::Error>>(
        canvas: HtmlCanvasElement,
        build: impl FnOnce() -> Result<(Engine, G), E>,
    ) -> Result<Self, JsValue> {
        console_error_panic_hook::set_once();
        let target = engine_render::RenderTarget::Canvas(canvas);
        Ok(Self {
            host: GameHost::launch(target, build).await?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn frame(
        &mut self,
        timestamp: f64,
        width: u32,
        height: u32,
        forward: f32,
        right: f32,
        up: f32,
        yaw: f32,
        pitch: f32,
        held: u32,
        pressed: u32,
        released: u32,
    ) -> Result<(), JsValue> {
        let input = InputState {
            move_forward: forward,
            move_right: right,
            move_up: up,
            look_yaw: yaw,
            look_pitch: pitch,
        };
        Ok(self
            .host
            .frame(timestamp, width, height, input, held, pressed, released)?)
    }

    /// [`GameHost::take_jobs`] as the page's workers read them:
    /// `{cancel}`, `{clear: true}`, `{share, bytes}`, `{unshare}` and
    /// `{id, kind, input, priority, identity?}`.
    pub fn take_jobs(&mut self) -> js_sys::Array {
        let out = js_sys::Array::new();
        let set = |object: &js_sys::Object, field: &str, value: &JsValue| {
            // Setting a field of a plain new object cannot fail.
            let _ = js_sys::Reflect::set(object, &JsValue::from_str(field), value);
        };
        for message in self.host.take_jobs() {
            let object = js_sys::Object::new();
            match message {
                JobMessage::Cancel(id) => set(&object, "cancel", &JsValue::from_f64(id.0 as f64)),
                JobMessage::Share(JobShare::Clear) => set(&object, "clear", &JsValue::TRUE),
                JobMessage::Share(JobShare::Set { key, bytes }) => {
                    set(&object, "share", &JsValue::from_str(&key));
                    set(
                        &object,
                        "bytes",
                        &js_sys::Uint8Array::from(bytes.as_slice()),
                    );
                }
                JobMessage::Share(JobShare::Remove { key }) => {
                    set(&object, "unshare", &JsValue::from_str(&key));
                }
                JobMessage::Run(request) => {
                    set(&object, "id", &JsValue::from_f64(request.id.0 as f64));
                    set(&object, "kind", &JsValue::from_str(request.kind));
                    set(
                        &object,
                        "input",
                        &js_sys::Uint8Array::from(request.input.as_slice()),
                    );
                    set(
                        &object,
                        "priority",
                        &JsValue::from_f64(f64::from(request.priority)),
                    );
                    // What hosts may keep its output by.
                    if let Some(identity) = &request.identity {
                        set(
                            &object,
                            "identity",
                            &js_sys::Uint8Array::from(identity.as_slice()),
                        );
                    }
                }
            }
            out.push(&object);
        }
        out
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

    pub fn part_job(&mut self, id: f64, part: Vec<u8>) {
        if let Some(id) = job_id(id) {
            self.host.part_job(id, part);
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

    pub fn reload_scene(&mut self, source: &str, files: SceneFileSet) -> Result<(), JsValue> {
        Ok(self.host.reload_scene(source, files.files())?)
    }
}

#[cfg(feature = "dev")]
impl<G: Game> PageHost<G> {
    pub fn dev_reload(
        &mut self,
        source: &str,
        expected_revision: &str,
        revision: &str,
        files: Option<SceneFileSet>,
    ) -> String {
        self.host.dev_reload(
            source,
            expected_revision,
            revision,
            files.as_ref().map(SceneFileSet::files),
        )
    }
}

#[cfg(feature = "profiling")]
impl<G: Game> PageHost<G> {
    pub fn profiling_start(&mut self, max_events: u32) -> Result<(), JsValue> {
        self.host
            .profiling_start(max_events as usize)
            .map_err(JsValue::from_str)
    }

    pub fn profiling_stop(&mut self) -> Result<String, JsValue> {
        self.host
            .profiling_stop()
            .map(|report| report.to_string())
            .map_err(JsValue::from_str)
    }

    pub fn profiling_report(&mut self) -> String {
        self.host.profiling_report().to_string()
    }

    pub fn profiling_status(&self) -> String {
        self.host.profiling_status().to_string()
    }
}

/// The job the page's `id` names: none unless finite and at least 1.
fn job_id(id: f64) -> Option<JobId> {
    (id.is_finite() && id >= 1.0).then_some(JobId(id as u64))
}
