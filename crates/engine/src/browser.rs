//! Coarse frame boundary between the TypeScript host and Rust engine.
use engine_core::{Engine, FrameReport, InputState, PresentationMode};
use engine_render::Renderer;
use serde::Serialize;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

#[cfg(feature = "profiling")]
#[path = "browser_profiling.rs"]
mod profiling;

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
}

impl SceneFileSet {
    pub fn files(&self) -> &engine_scene::SceneFiles {
        &self.files
    }
}

/// The files scene `source` references, as JSON `[{"uri", "kind"}]` with
/// kind `image`, `buffer` or `virtual_mesh`, for the page to fetch; an
/// unparsable source references none (loading reports its error).
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
    #[cfg(feature = "profiling")]
    profile_frame: u64,
}

impl BrowserApp {
    pub async fn new(canvas: HtmlCanvasElement, engine: Engine) -> Result<Self, JsValue> {
        let performance = web_sys::window()
            .and_then(|window| window.performance())
            .ok_or_else(|| JsValue::from_str("A browser performance clock is required."))?;
        let renderer = Renderer::new(canvas).await?;
        Ok(Self {
            engine,
            renderer,
            last_timestamp_ms: None,
            performance,
            stats: FrameStats::default(),
            size: (0, 0),
            gpu_presentation: true,
            tick_pipelining: true,
            gpu_inputs_stale: false,
            #[cfg(feature = "profiling")]
            profile_frame: 0,
        })
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
            PresentationMode::Interpolated {
                alpha: report.interpolation,
            },
            self.gpu_presentation,
            true,
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
        self.render_frame(
            width,
            height,
            simulation_ms,
            report,
            PresentationMode::Current,
            false,
            false,
        )
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
        self.render_frame(
            width,
            height,
            0.0,
            FrameReport::default(),
            PresentationMode::Current,
            false,
            false,
        )
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
            PresentationMode::Interpolated { alpha },
            gpu,
            false,
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

    /// `display` marks ordinary frames, which temporal antialiasing may
    /// blend; others render exactly.
    #[allow(clippy::too_many_arguments)]
    fn render_frame(
        &mut self,
        width: u32,
        height: u32,
        simulation_ms: f64,
        report: FrameReport,
        presentation: PresentationMode,
        gpu: bool,
        display: bool,
    ) -> Result<(), JsValue> {
        if self.size != (width, height) {
            self.renderer.resize(width, height);
            self.size = (width, height);
        }
        let before_extraction = self.performance.now();
        let culling_enabled = self.engine.frustum_culling();
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        if gpu && self.gpu_inputs_stale {
            // The renderer missed a gather. Current poses are the hierarchy's
            // own locals, so an earlier interval can no longer be uploaded:
            // complete and present the latest tick instead.
            self.engine.complete_pending_tick();
            self.engine.reveal_ticks_ahead();
        }
        let frame = match presentation {
            PresentationMode::Interpolated { alpha } if gpu => {
                self.engine.extract_gpu_frame(aspect, alpha)
            }
            mode => self.engine.extract_frame(aspect, mode),
        }
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
        let after_extraction = self.performance.now();
        let rendered = if display {
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
            render: rendered,
        };
        Ok(())
    }

    pub fn stats(&self) -> FrameStats {
        FrameStats {
            pipelines_compiling: self.renderer.pipelines_compiling(),
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
