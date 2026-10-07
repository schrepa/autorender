//! Coarse frame boundary between the TypeScript host and Rust engine.
use engine_core::{Engine, FrameReport, InputState, PresentationMode};
use engine_render::{Renderer, StartingRenderer};
use serde::Serialize;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

#[cfg(feature = "profiling")]
#[path = "browser_profiling.rs"]
mod profiling;

/// Milliseconds on the page's clock, which terrain times its work by
/// (decision 099); 0 without one.
fn page_clock() -> f64 {
    thread_local! {
        static CLOCK: Option<web_sys::Performance> =
            web_sys::window().and_then(|window| window.performance());
    }
    CLOCK.with(|clock| clock.as_ref().map_or(0.0, web_sys::Performance::now))
}

/// Decoded images and buffers of the files a scene references (decision
/// 030), filled by the page before the scene loads.
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
    /// `etc2` or `astc`, decision 037): `width` × `height`, its levels
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

    /// Adds virtual mesh `uri` (decision 044): its prefix
    /// ([`virtual_prefix_bytes`] long) and the URL renderers stream its
    /// other pages from.
    pub fn add_virtual_mesh(&mut self, uri: &str, bytes: Vec<u8>, url: Option<String>) {
        self.files.add_virtual_mesh(uri, bytes, url.as_deref());
    }

    /// Adds terrain file `uri` (decision 051): its prefix
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

/// Bytes of the prefix of the virtual mesh file (decision 044) whose first
/// 128 bytes are `header`: what hosts fetch before a scene loads; renderers
/// stream the rest. `None` when it is not a virtual mesh file of this
/// engine's version.
#[wasm_bindgen]
pub fn virtual_prefix_bytes(header: &[u8]) -> Option<u32> {
    engine_core::VirtualMesh::prefix_bytes(header).map(|bytes| bytes as u32)
}

/// Bytes of the prefix (header and directory) of the terrain file (decision
/// 051) whose first 64 bytes are `header`: what hosts fetch before a scene
/// loads; tiles stream by byte range as the camera needs them. `None` when
/// it is not a terrain file.
#[wasm_bindgen]
pub fn terrain_prefix_bytes(header: &[u8]) -> Option<u32> {
    engine_core::FileHeader::prefix_bytes(header)
        .ok()
        .map(|bytes| bytes as u32)
}

/// Terrain tiles in flight (decision 051): fetched by byte range from their
/// files' URLs and handed back to the engine's terrains the next frame.
#[derive(Default)]
struct TerrainFetches {
    inbox: std::rc::Rc<std::cell::RefCell<Vec<TerrainArrival>>>,
    requests: Vec<engine_core::TileRequest>,
}

/// A fetched tile: its terrain, key and bytes, `None` when the fetch
/// failed.
type TerrainArrival = (
    engine_core::bevy_ecs::entity::Entity,
    engine_core::TileKey,
    Option<Vec<u8>>,
);

impl TerrainFetches {
    /// Hands arrived tiles to `engine`'s terrains and starts fetching the
    /// tiles they request.
    fn exchange(&mut self, engine: &mut Engine) {
        let Some(mut terrains) = engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::Terrains>()
        else {
            return;
        };
        for (terrain, key, bytes) in self.inbox.borrow_mut().drain(..) {
            terrains.deliver(terrain, key, bytes);
        }
        terrains.take_requests(&mut self.requests);
        for request in self.requests.drain(..) {
            fetch_tile(request, std::rc::Rc::clone(&self.inbox));
        }
    }
}

/// Fetches `request`'s byte range into `inbox` (a server ignoring ranges
/// sends the whole file, which is cut to the range).
fn fetch_tile(
    request: engine_core::TileRequest,
    inbox: std::rc::Rc<std::cell::RefCell<Vec<TerrainArrival>>>,
) {
    let range = request.offset..request.offset + u64::from(request.length);
    let fetch = (|| {
        let init = web_sys::RequestInit::new();
        init.set_method("GET");
        let headers = web_sys::Headers::new().ok()?;
        headers
            .set("Range", &format!("bytes={}-{}", range.start, range.end - 1))
            .ok()?;
        init.set_headers(&headers);
        let request = web_sys::Request::new_with_str_and_init(&request.url, &init).ok()?;
        Some(web_sys::window()?.fetch_with_request(&request))
    })();
    wasm_bindgen_futures::spawn_local(async move {
        let bytes = async {
            let response: web_sys::Response = wasm_bindgen_futures::JsFuture::from(fetch?)
                .await
                .ok()?
                .dyn_into()
                .ok()?;
            if !response.ok() {
                return None;
            }
            let whole = response.status() == 200;
            let buffer = wasm_bindgen_futures::JsFuture::from(response.array_buffer().ok()?)
                .await
                .ok()?;
            let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
            if whole {
                bytes
                    .get(range.start as usize..range.end as usize)
                    .map(<[u8]>::to_vec)
            } else {
                Some(bytes)
            }
        }
        .await;
        inbox
            .borrow_mut()
            .push((request.terrain, request.key, bytes));
    });
}

/// CPU observations from the last submitted frame and the renderer's report
/// of it. Render time is not GPU time.
///
/// Hosts and agents read it as the flat JSON object [`BrowserApp::stats_json`]
/// returns: these fields and the renderer's [`engine_render::RenderStats`],
/// under their own names.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct FrameStats {
    pub simulation_ms: f64,
    pub extraction_ms: f64,
    pub render_ms: f64,
    pub invalid_transforms: u32,
    pub render_candidates: u32,
    pub visible_instances: u32,
    pub culled_instances: u32,
    pub dropped_instances: u32,
    pub missing_assets: u32,
    pub culling_enabled: bool,
    pub fixed_steps: u32,
    /// Total simulation time discarded by the fixed-step budget in this session.
    pub dropped_seconds: f64,
    /// Whether the GPU may still be compiling pipelines earlier frames
    /// created, read when the stats are (decision 049).
    pub pipelines_compiling: bool,
    /// Whether terrains have yet to load the camera's view whole, read
    /// when the stats are (decision 099): hosts keep preparing meanwhile.
    pub loading: bool,
    /// The host's workers and their jobs (decision 110), read when the
    /// stats are.
    pub jobs: engine_core::JobStats,
    #[serde(flatten)]
    pub render: engine_render::RenderStats,
}

/// Owns the runtime, extracted data, and GPU state for one canvas.
pub struct BrowserApp {
    pub engine: Engine,
    renderer: Renderer,
    last_timestamp_ms: Option<f64>,
    performance: web_sys::Performance,
    stats: FrameStats,
    size: (u32, u32),
    // Ordinary interpolated frames blend, compose and cull on the GPU.
    gpu_presentation: bool,
    // GPU-presented frames start the next fixed tick ahead of its due frame.
    tick_pipelining: bool,
    // The last GPU frame was not submitted, so the renderer missed its gather.
    gpu_inputs_stale: bool,
    // Terrain tiles fetched by byte range (decision 051).
    terrain_fetches: TerrainFetches,
    #[cfg(feature = "profiling")]
    profile_frame: u64,
}

/// What a submitted frame shows. Only development controls submit exact
/// frames and previews.
#[derive(Clone, Copy)]
#[cfg_attr(not(feature = "dev"), allow(dead_code))]
enum FrameKind {
    /// An ordinary frame at fraction `alpha` of the fixed tick: GPU-presented
    /// unless the game opted out, and blended by temporal antialiasing.
    Display { alpha: f32 },
    /// The exact current state: steps, renders and captures.
    Exact,
    /// Fraction `alpha` of the latest completed tick, presented on the GPU
    /// when `gpu`, else exactly.
    Preview { alpha: f32, gpu: bool },
}

/// A [`BrowserApp`] whose renderer's startup pipelines may still compile:
/// build the engine meanwhile, then [`Self::finish`].
pub struct StartingApp {
    renderer: StartingRenderer,
    performance: web_sys::Performance,
}

impl StartingApp {
    /// The app running `engine`, once the renderer's startup pipelines passed
    /// validation.
    pub async fn finish(self, engine: Engine) -> Result<BrowserApp, JsValue> {
        let renderer = self.renderer.finish().await?;
        Ok(BrowserApp::with(engine, renderer, self.performance))
    }
}

impl BrowserApp {
    pub async fn new(canvas: HtmlCanvasElement, engine: Engine) -> Result<Self, JsValue> {
        Self::start(canvas).await?.finish(engine).await
    }

    /// Starts the renderer of `canvas` before the engine exists: the browser
    /// compiles its startup pipelines while the caller builds the engine for
    /// [`StartingApp::finish`].
    pub async fn start(canvas: HtmlCanvasElement) -> Result<StartingApp, JsValue> {
        let performance = web_sys::window()
            .and_then(|window| window.performance())
            .ok_or_else(|| JsValue::from_str("A browser performance clock is required."))?;
        let renderer = Renderer::start(canvas).await?;
        Ok(StartingApp {
            renderer,
            performance,
        })
    }

    fn with(mut engine: Engine, renderer: Renderer, performance: web_sys::Performance) -> Self {
        // Terrain residency never exceeds the device's texture array layers
        // (decision 051), and its work runs for the settings' time a frame
        // (decision 099).
        if let Some(mut terrains) = engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::Terrains>()
        {
            terrains.set_device_tiles(Some(renderer.terrain_tile_capacity()));
            terrains.set_clock(page_clock);
        }
        Self {
            engine,
            renderer,
            last_timestamp_ms: None,
            performance,
            stats: FrameStats::default(),
            size: (0, 0),
            gpu_presentation: true,
            tick_pipelining: true,
            gpu_inputs_stale: false,
            terrain_fetches: TerrainFetches::default(),
            #[cfg(feature = "profiling")]
            profile_frame: 0,
        }
    }

    /// Selects GPU blending, composition and culling for ordinary interpolated
    /// frames (the default; decision 014) or the exact CPU path. Development
    /// steps, renders and captures always use the exact CPU path.
    pub fn set_gpu_presentation(&mut self, enabled: bool) {
        self.gpu_presentation = enabled;
        if !enabled {
            self.engine.complete_pending_tick();
        }
    }

    pub fn gpu_presentation(&self) -> bool {
        self.gpu_presentation
    }

    /// Splits each fixed tick across the frames around its due time when GPU
    /// presentation is enabled (the default): after a frame is submitted, the
    /// next tick's gameplay stage starts if the tick will be due by the next
    /// frame, and its propagation stage runs when due. See
    /// [`Engine::run_ahead`]. Disabling completes a started tick.
    pub fn set_tick_pipelining(&mut self, enabled: bool) {
        self.tick_pipelining = enabled;
        if !enabled {
            self.engine.complete_pending_tick();
        }
    }

    pub fn tick_pipelining(&self) -> bool {
        self.tick_pipelining
    }

    /// Temporally antialiases ordinary GPU-presented frames (decision 017):
    /// each rasterizes with a sub-pixel jitter and blends with the previous
    /// frame's output, reprojected by per-pixel motion. Off by default.
    /// Development steps, renders, previews and captures stay exact, as do
    /// CPU-presented frames. Enabling creates its pipelines; disabling
    /// releases its targets.
    pub fn set_temporal_antialiasing(&mut self, enabled: bool) {
        self.renderer.set_temporal_antialiasing(enabled);
    }

    pub fn temporal_antialiasing(&self) -> bool {
        self.renderer.temporal_antialiasing()
    }

    /// Occlusion culling of GPU-presented frames (decision 040), on by
    /// default: frames that cull without an instance budget draw what the
    /// last frame showed, test every other frustum-visible instance against
    /// a depth pyramid of those depths, then draw only what it does not hide.
    /// Output is unchanged; hidden instances cost no drawing.
    pub fn set_occlusion_culling(&mut self, enabled: bool) {
        self.renderer.set_occlusion_culling(enabled);
    }

    pub fn occlusion_culling(&self) -> bool {
        self.renderer.occlusion_culling()
    }

    /// Sub-pixel culling of GPU-presented frames (decision 041), on by
    /// default: frames that cull without an instance budget leave visible
    /// instances whose projected bounds cover no pixel center undrawn.
    /// Output is unchanged; they could produce no fragments.
    pub fn set_subpixel_culling(&mut self, enabled: bool) {
        self.renderer.set_subpixel_culling(enabled);
    }

    pub fn subpixel_culling(&self) -> bool {
        self.renderer.subpixel_culling()
    }

    /// Streams terrains for a frame's work time on a `width` × `height`
    /// canvas without simulating or rendering (decision 099): hosts call it
    /// in place of frames while the GPU compiles pipelines, which frames
    /// would only queue behind, so loading goes on meanwhile.
    pub fn work(&mut self, width: u32, height: u32) {
        let viewport = engine_core::Viewport {
            width: width.max(1),
            height: height.max(1),
        };
        let world = self.engine.app.world_mut();
        if world.get_resource::<engine_core::Viewport>() != Some(&viewport) {
            world.insert_resource(viewport);
        }
        self.terrain_fetches.exchange(&mut self.engine);
        self.engine.stream_terrain();
    }

    /// Loads terrain for `milliseconds` between frames (decision 108): hosts
    /// call it after a frame with the time left before the next one is due,
    /// while the last frame found tiles to load. Returns whether tiles are
    /// still to load.
    pub fn work_between_frames(&mut self, milliseconds: f64) -> bool {
        if !milliseconds.is_finite() || milliseconds <= 0.0 {
            return false;
        }
        self.terrain_fetches.exchange(&mut self.engine);
        self.engine.work_terrain(milliseconds)
    }

    /// Workers the host runs jobs on (decision 110): with any, terrains bake
    /// and generate tiles and agents search walks there; with none, the
    /// default, that work stays in the engine, which takes back every job
    /// handed out.
    pub fn set_job_workers(&mut self, workers: u32) {
        if let Some(mut jobs) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::Jobs>()
        {
            jobs.set_workers(workers);
        }
    }

    /// What to hand workers since the last call, in order (decision 110):
    /// `{cancel}` for jobs handed out that their owners cancelled, which
    /// their workers stop (decision 112), `{clear: true}` to drop every
    /// share an earlier engine made, `{share, bytes}` to hold bytes as a key
    /// and `{unshare}` to drop one, to every worker, then
    /// `{id, kind, input, priority}` jobs, each to one, lower priorities
    /// first.
    pub fn take_jobs(&mut self) -> js_sys::Array {
        let out = js_sys::Array::new();
        let Some(mut jobs) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::Jobs>()
        else {
            return out;
        };
        let mut cancels = Vec::new();
        let mut shares = Vec::new();
        let mut requests = Vec::new();
        jobs.take_cancels(&mut cancels);
        jobs.take_shares(&mut shares);
        jobs.take_requests(&mut requests);
        let set = |object: &js_sys::Object, field: &str, value: &JsValue| {
            // Setting a field of a plain new object cannot fail.
            let _ = js_sys::Reflect::set(object, &JsValue::from_str(field), value);
        };
        for id in cancels {
            let object = js_sys::Object::new();
            set(&object, "cancel", &JsValue::from_f64(id.0 as f64));
            out.push(&object);
        }
        for share in shares {
            let object = js_sys::Object::new();
            match share {
                engine_core::JobShare::Clear => set(&object, "clear", &JsValue::TRUE),
                engine_core::JobShare::Set { key, bytes } => {
                    set(&object, "share", &JsValue::from_str(&key));
                    set(
                        &object,
                        "bytes",
                        &js_sys::Uint8Array::from(bytes.as_slice()),
                    );
                }
                engine_core::JobShare::Remove { key } => {
                    set(&object, "unshare", &JsValue::from_str(&key));
                }
            }
            out.push(&object);
        }
        for request in requests {
            let object = js_sys::Object::new();
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
            // What hosts may keep its output by (decision 116).
            if let Some(identity) = &request.identity {
                set(
                    &object,
                    "identity",
                    &js_sys::Uint8Array::from(identity.as_slice()),
                );
            }
            out.push(&object);
        }
        out
    }

    /// Job `id`'s output from its worker (decision 110).
    pub fn complete_job(&mut self, id: f64, output: Vec<u8>) {
        self.finish_job(id, Ok(output));
    }

    /// Job `id` failed on its worker, for `error` (decision 110): its owner
    /// takes the work back.
    pub fn fail_job(&mut self, id: f64, error: String) {
        self.finish_job(id, Err(error));
    }

    /// The host's output cache's counters (decision 116): outputs handed
    /// back from it, jobs it lacked, its bytes and, verifying, outputs that
    /// differed from a fresh run.
    pub fn report_job_cache(&mut self, hits: f64, misses: f64, bytes: f64, mismatches: f64) {
        let count = |value: f64| {
            if value.is_finite() && value > 0.0 {
                value as u64
            } else {
                0
            }
        };
        if let Some(mut jobs) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::Jobs>()
        {
            jobs.report_cache(count(hits), count(misses), count(bytes), count(mismatches));
        }
    }

    /// A part of job `id`'s output its worker handed back before the rest
    /// (decision 114), such as a tile's shape.
    pub fn part_job(&mut self, id: f64, part: Vec<u8>) {
        if !(id.is_finite() && id >= 1.0) {
            return;
        }
        if let Some(mut jobs) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::Jobs>()
        {
            jobs.part(engine_core::JobId(id as u64), part);
        }
    }

    fn finish_job(&mut self, id: f64, output: Result<Vec<u8>, String>) {
        if !(id.is_finite() && id >= 1.0) {
            return;
        }
        if let Some(mut jobs) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::Jobs>()
        {
            jobs.complete(engine_core::JobId(id as u64), output);
        }
    }

    /// Advance once per animation frame; input is a bounded, owned value.
    pub fn frame(
        &mut self,
        timestamp_ms: f64,
        width: u32,
        height: u32,
        input: InputState,
    ) -> Result<(), JsValue> {
        if !timestamp_ms.is_finite() {
            return Err(JsValue::from_str("Frame timestamp must be finite."));
        }
        #[cfg(feature = "profiling")]
        self.begin_profile_frame();
        let _profile = engine_profile::span("cpu", "browser.frame");
        let delta = self
            .last_timestamp_ms
            .map_or(0.0, |last| (timestamp_ms - last).max(0.0) * 0.001);
        self.last_timestamp_ms = Some(timestamp_ms);
        self.set_input(input);
        let start = self.performance.now();
        let report = self.engine.advance(delta);
        let simulation_ms = self.performance.now() - start;
        self.render_frame(
            width,
            height,
            simulation_ms,
            report,
            FrameKind::Display {
                alpha: report.interpolation,
            },
        )?;
        // After submission, so the GPU draws this frame while the next tick's
        // first stage runs; the frame interval predicts the next one.
        if self.gpu_presentation && self.tick_pipelining {
            let start = self.performance.now();
            if self.engine.run_ahead(delta) {
                // A tick just completed ahead: stage its current poses now, so
                // the frame that gathers them only records their GPU copy.
                if self.engine.tick_completed_ahead()
                    && let Some(current) = self.engine.gpu_current()
                {
                    self.renderer.prestage_gpu_current(current);
                }
                self.stats.simulation_ms += self.performance.now() - start;
            }
        }
        Ok(())
    }

    /// Runs exact fixed ticks and submits the resulting state while paused.
    ///
    /// Uses [`Engine::step_fixed`]'s count and presentation contract. Rejected
    /// tick counts leave the engine, input, timestamp, and renderer untouched.
    /// Merge action input before this call, after validating the count. This
    /// method preserves those edges and clears the real-time timestamp baseline
    /// so the next ordinary frame excludes time spent in development control.
    #[cfg(feature = "dev")]
    pub fn dev_step(
        &mut self,
        ticks: u32,
        width: u32,
        height: u32,
        input: InputState,
    ) -> Result<(), JsValue> {
        engine_core::DevStepError::validate(ticks)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        #[cfg(feature = "profiling")]
        self.begin_profile_frame();
        let _profile = engine_profile::span("cpu", "browser.step");
        self.engine.complete_pending_tick();
        self.last_timestamp_ms = None;
        self.set_input(input);
        let start = self.performance.now();
        let report = self
            .engine
            .step_fixed(ticks)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        let simulation_ms = self.performance.now() - start;
        self.render_frame(width, height, simulation_ms, report, FrameKind::Exact)
    }

    /// Extracts and submits current state without running either schedule.
    ///
    /// Intended for captures and scene replacement while paused. Frame/tick
    /// counters, time, and input are unchanged; submission diagnostics report
    /// zero simulation time and fixed steps, retaining cumulative dropped time.
    #[cfg(feature = "dev")]
    pub fn dev_render(&mut self, width: u32, height: u32) -> Result<(), JsValue> {
        #[cfg(feature = "profiling")]
        self.begin_profile_frame();
        self.render_frame(width, height, 0.0, FrameReport::default(), FrameKind::Exact)
    }

    /// Render a selected fraction of the last fixed tick without running schedules.
    /// Invalid fractions are rejected before touching the engine or renderer.
    /// Ordinary development rendering and captures use current simulation state.
    /// `gpu` presents the fraction through GPU presentation instead of the
    /// exact CPU path, e.g. to compare both.
    #[cfg(feature = "dev")]
    pub fn dev_present(
        &mut self,
        alpha: f32,
        width: u32,
        height: u32,
        gpu: bool,
    ) -> Result<(), JsValue> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(JsValue::from_str(
                "Presentation alpha must be finite and in [0, 1].",
            ));
        }
        #[cfg(feature = "profiling")]
        self.begin_profile_frame();
        // Previews show the latest completed tick.
        self.engine.complete_pending_tick();
        self.engine.reveal_ticks_ahead();
        self.render_frame(
            width,
            height,
            0.0,
            FrameReport::default(),
            FrameKind::Preview { alpha, gpu },
        )
    }

    fn set_input(&mut self, input: InputState) {
        let _profile = engine_profile::span("cpu", "browser.input");
        fn bounded(value: f32, limit: f32) -> f32 {
            if value.is_finite() {
                engine_core::math::clamp(value, -limit, limit)
            } else {
                0.0
            }
        }
        *self.engine.app.world_mut().resource_mut::<InputState>() = InputState {
            move_forward: bounded(input.move_forward, 1.0),
            move_right: bounded(input.move_right, 1.0),
            move_up: bounded(input.move_up, 1.0),
            look_yaw: bounded(input.look_yaw, 1000.0),
            look_pitch: bounded(input.look_pitch, 1000.0),
        };
    }

    fn render_frame(
        &mut self,
        width: u32,
        height: u32,
        simulation_ms: f64,
        report: FrameReport,
        kind: FrameKind,
    ) -> Result<(), JsValue> {
        if self.size != (width, height) {
            self.renderer.resize(width, height);
            self.size = (width, height);
            // Terrain level of detail measures pixels (decision 051).
            self.engine
                .app
                .world_mut()
                .insert_resource(engine_core::Viewport {
                    width: width.max(1),
                    height: height.max(1),
                });
        }
        // Tiles that arrived stream in with this frame's extraction.
        self.terrain_fetches.exchange(&mut self.engine);
        let before_extraction = self.performance.now();
        let culling_enabled = self.engine.frustum_culling();
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        let gpu = match kind {
            FrameKind::Display { .. } => self.gpu_presentation,
            FrameKind::Exact => false,
            FrameKind::Preview { gpu, .. } => gpu,
        };
        if gpu && self.gpu_inputs_stale {
            // The renderer missed a gather. Current poses are the hierarchy's
            // own locals, so an earlier interval can no longer be uploaded:
            // complete and present the latest tick instead.
            self.engine.complete_pending_tick();
            self.engine.reveal_ticks_ahead();
        }
        let frame = match kind {
            FrameKind::Display { alpha } | FrameKind::Preview { alpha, .. } if gpu => {
                self.engine.extract_gpu_frame(aspect, alpha)
            }
            FrameKind::Display { alpha } | FrameKind::Preview { alpha, .. } => self
                .engine
                .extract_frame(aspect, PresentationMode::Interpolated { alpha }),
            FrameKind::Exact => self.engine.extract_frame(aspect, PresentationMode::Current),
        }
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
        let after_extraction = self.performance.now();
        let rendered = if let FrameKind::Display { .. } = kind {
            self.renderer.render_display_frame(frame)?
        } else {
            self.renderer.render_frame(frame)?
        };
        if frame.gpu().is_some() {
            self.gpu_inputs_stale = !rendered.submitted;
        }
        let snapshot = frame.snapshot();
        // GPU-culled frames list their valid candidates as visible; culling and
        // budget counts come from a GPU read-back a few frames old.
        let listed = snapshot.visible_instances.min(u32::MAX as usize) as u32;
        let (visible, culled, dropped) = if frame.gpu().is_some() {
            let visible = rendered.gpu_visible.unwrap_or(listed);
            (
                visible,
                listed.saturating_sub(visible),
                visible.saturating_sub(rendered.instances),
            )
        } else {
            (
                listed,
                snapshot.culled_instances.min(u32::MAX as usize) as u32,
                snapshot.dropped_instances.min(u32::MAX as usize) as u32,
            )
        };
        self.stats = FrameStats {
            simulation_ms,
            extraction_ms: after_extraction - before_extraction,
            render_ms: self.performance.now() - after_extraction,
            invalid_transforms: snapshot.invalid_transforms.min(u32::MAX as usize) as u32,
            render_candidates: snapshot.render_candidates.min(u32::MAX as usize) as u32,
            visible_instances: visible,
            culled_instances: culled,
            dropped_instances: dropped,
            missing_assets: snapshot.missing_assets.min(u32::MAX as usize) as u32,
            culling_enabled,
            fixed_steps: report.fixed_steps,
            dropped_seconds: self.stats.dropped_seconds + report.dropped_seconds,
            pipelines_compiling: false,
            loading: false,
            jobs: engine_core::JobStats::default(),
            render: rendered,
        };
        Ok(())
    }

    pub fn stats(&self) -> FrameStats {
        FrameStats {
            pipelines_compiling: self.renderer.pipelines_compiling(),
            loading: self
                .engine
                .app
                .world()
                .get_resource::<engine_core::Terrains>()
                .is_some_and(engine_core::Terrains::loading),
            jobs: self
                .engine
                .app
                .world()
                .get_resource::<engine_core::Jobs>()
                .map(engine_core::Jobs::stats)
                .unwrap_or_default(),
            ..self.stats
        }
    }

    /// [`Self::stats`] as the JSON object hosts and agents read.
    pub fn stats_json(&self) -> String {
        serde_json::to_string(&self.stats()).expect("frame stats serialize")
    }

    /// GPU residency reflects the last valid render boundary, not queue completion.
    #[cfg(feature = "dev")]
    pub(crate) fn texture_resident(&self, handle: engine_core::TextureHandle) -> bool {
        self.renderer.is_texture_resident(handle)
    }

    /// Drop suspension time and held input before the next frame.
    pub fn suspend(&mut self) {
        self.last_timestamp_ms = None;
        self.engine.reset_clock();
        *self.engine.app.world_mut().resource_mut::<InputState>() = InputState::default();
    }
}
