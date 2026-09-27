//! Coarse frame boundary between the TypeScript host and Rust engine.
use engine_core::{Engine, FrameReport, InputState, PresentationMode};
use engine_render::Renderer;
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
}

impl SceneFileSet {
    pub fn files(&self) -> &engine_scene::SceneFiles {
        &self.files
    }
}

/// The files scene `source` references, as JSON `[{"uri", "kind"}]` with
/// kind `image` or `buffer`, for the page to fetch; an unparsable source
/// references none (loading reports its error).
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
            };
            format!(r#"{{"uri":"{}","kind":"{kind}"}}"#, file.uri)
        })
        .collect();
    format!("[{}]", entries.join(","))
}

/// CPU observations from the last submitted frame. Render time is not GPU time.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub simulation_ms: f64,
    pub extraction_ms: f64,
    pub render_ms: f64,
    pub instances: u32,
    pub invalid_transforms: u32,
    pub render_candidates: u32,
    pub visible_instances: u32,
    pub culled_instances: u32,
    pub dropped_instances: u32,
    pub missing_assets: u32,
    pub culling_enabled: bool,
    pub draw_calls: u32,
    pub upload_bytes: u64,
    pub fixed_steps: u32,
    /// Total simulation time discarded by the fixed-step budget in this session.
    pub dropped_seconds: f64,
    pub resident_meshes: u32,
    pub mesh_upload_bytes: u64,
    pub resident_mesh_bytes: u64,
    pub resident_textures: u32,
    pub texture_upload_bytes: u64,
    pub resident_texture_bytes: u64,
    /// Sun shadow work (decision 020); `None` when shadows are off.
    pub shadows: Option<engine_render::ShadowStats>,
    /// Screen effects applied (decision 021).
    pub ambient_occlusion: bool,
    pub bloom: bool,
    /// Local lights (decision 023); `None` when the frame shaded none.
    pub lights: Option<engine_render::LightStats>,
    /// Sun shafts (decision 024) and reflections (decision 025) applied.
    pub sun_shafts: bool,
    pub reflections: bool,
    /// Glass refracted the opaque scene (decision 038).
    pub refraction: bool,
    /// Transparent draws, particles drawn and particles a budget or the
    /// GPU's bindings left out (decision 026).
    pub transparent_draws: u32,
    pub particles: u32,
    pub dropped_particles: u32,
    /// Decals drawn and decals a budget or the GPU's binding left out
    /// (decision 039).
    pub decals: u32,
    pub dropped_decals: u32,
}

/// Owned diagnostics returned at low frequency across the browser boundary.
/// JavaScript callers free this generated binding object after reading it.
#[wasm_bindgen]
pub struct BrowserStats {
    pub simulation_ms: f64,
    pub extraction_ms: f64,
    pub render_ms: f64,
    pub instances: u32,
    pub invalid_transforms: u32,
    pub render_candidates: u32,
    pub visible_instances: u32,
    pub culled_instances: u32,
    pub dropped_instances: u32,
    pub missing_assets: u32,
    pub culling_enabled: bool,
    pub draw_calls: u32,
    pub upload_bytes: f64,
    pub fixed_steps: u32,
    pub dropped_seconds: f64,
    pub resident_meshes: u32,
    pub mesh_upload_bytes: f64,
    pub resident_mesh_bytes: f64,
    pub resident_textures: u32,
    pub texture_upload_bytes: f64,
    pub resident_texture_bytes: f64,
    /// Whether the sun cast shadows (decision 020).
    pub shadows_enabled: bool,
    /// Indexed shadow-map draws, in addition to the camera pass's `draw_calls`.
    pub shadow_draws: u32,
    /// Kept-map redraw: 0 none, 1 one strip, 2 the whole map.
    pub shadow_kept_redraw: u32,
    /// Casters the near, moving and kept shadow maps drew, read back a few
    /// frames late; undefined before the first read-back.
    pub shadow_near_casters: Option<u32>,
    pub shadow_moving_casters: Option<u32>,
    pub shadow_kept_casters: Option<u32>,
    /// Whether ambient occlusion and bloom ran (decision 021).
    pub ambient_occlusion: bool,
    pub bloom: bool,
    /// Local lights shaded (decision 023) and usable lights beyond the budget.
    pub local_lights: u32,
    pub dropped_lights: u32,
    /// Lights that reached a view cluster and cluster entries lost to full
    /// lists, read back a few frames late; undefined before then.
    pub lights_visible: Option<u32>,
    pub light_cluster_overflow: Option<u32>,
    /// Whether sun shafts (decision 024) and reflections (decision 025) ran.
    pub sun_shafts: bool,
    pub reflections: bool,
    /// Whether glass refracted the opaque scene (decision 038).
    pub refraction: bool,
    /// Transparent draws, included in `draw_calls`, particles drawn and
    /// particles a budget or the GPU's bindings left out (decision 026).
    pub transparent_draws: u32,
    pub particles: u32,
    pub dropped_particles: u32,
    /// Decals drawn and decals a budget or the GPU's binding left out
    /// (decision 039).
    pub decals: u32,
    pub dropped_decals: u32,
}

impl From<FrameStats> for BrowserStats {
    fn from(stats: FrameStats) -> Self {
        Self {
            simulation_ms: stats.simulation_ms,
            extraction_ms: stats.extraction_ms,
            render_ms: stats.render_ms,
            instances: stats.instances,
            invalid_transforms: stats.invalid_transforms,
            render_candidates: stats.render_candidates,
            visible_instances: stats.visible_instances,
            culled_instances: stats.culled_instances,
            dropped_instances: stats.dropped_instances,
            missing_assets: stats.missing_assets,
            culling_enabled: stats.culling_enabled,
            draw_calls: stats.draw_calls,
            upload_bytes: stats.upload_bytes as f64,
            fixed_steps: stats.fixed_steps,
            dropped_seconds: stats.dropped_seconds,
            resident_meshes: stats.resident_meshes,
            mesh_upload_bytes: stats.mesh_upload_bytes as f64,
            resident_mesh_bytes: stats.resident_mesh_bytes as f64,
            resident_textures: stats.resident_textures,
            texture_upload_bytes: stats.texture_upload_bytes as f64,
            resident_texture_bytes: stats.resident_texture_bytes as f64,
            shadows_enabled: stats.shadows.is_some(),
            shadow_draws: stats.shadows.map_or(0, |shadows| shadows.draws),
            shadow_kept_redraw: stats
                .shadows
                .map_or(0, |shadows| u32::from(shadows.kept_redraw)),
            shadow_near_casters: stats
                .shadows
                .and_then(|shadows| shadows.casters)
                .map(|c| c[0]),
            shadow_moving_casters: stats
                .shadows
                .and_then(|shadows| shadows.casters)
                .map(|c| c[1]),
            shadow_kept_casters: stats
                .shadows
                .and_then(|shadows| shadows.casters)
                .map(|c| c[2]),
            ambient_occlusion: stats.ambient_occlusion,
            bloom: stats.bloom,
            local_lights: stats.lights.map_or(0, |lights| lights.lights),
            dropped_lights: stats.lights.map_or(0, |lights| lights.dropped),
            lights_visible: stats.lights.and_then(|lights| lights.visible),
            light_cluster_overflow: stats.lights.and_then(|lights| lights.overflow),
            sun_shafts: stats.sun_shafts,
            reflections: stats.reflections,
            refraction: stats.refraction,
            transparent_draws: stats.transparent_draws,
            particles: stats.particles,
            dropped_particles: stats.dropped_particles,
            decals: stats.decals,
            dropped_decals: stats.dropped_decals,
        }
    }
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
            instances: rendered.instances,
            invalid_transforms: snapshot.invalid_transforms.min(u32::MAX as usize) as u32,
            render_candidates: snapshot.render_candidates.min(u32::MAX as usize) as u32,
            visible_instances: visible,
            culled_instances: culled,
            dropped_instances: dropped,
            missing_assets: snapshot.missing_assets.min(u32::MAX as usize) as u32,
            culling_enabled,
            draw_calls: rendered.draw_calls,
            upload_bytes: rendered.upload_bytes,
            fixed_steps: report.fixed_steps,
            dropped_seconds: self.stats.dropped_seconds + report.dropped_seconds,
            resident_meshes: rendered.resident_meshes,
            mesh_upload_bytes: rendered.mesh_upload_bytes,
            resident_mesh_bytes: rendered.resident_mesh_bytes,
            resident_textures: rendered.resident_textures,
            texture_upload_bytes: rendered.texture_upload_bytes,
            resident_texture_bytes: rendered.resident_texture_bytes,
            shadows: rendered.shadows,
            ambient_occlusion: rendered.ambient_occlusion,
            bloom: rendered.bloom,
            lights: rendered.lights,
            sun_shafts: rendered.sun_shafts,
            reflections: rendered.reflections,
            refraction: rendered.refraction,
            transparent_draws: rendered.transparent_draws,
            particles: rendered.particles,
            dropped_particles: rendered.dropped_particles,
            decals: rendered.decals,
            dropped_decals: rendered.dropped_decals,
        };
        Ok(())
    }

    pub fn stats(&self) -> FrameStats {
        self.stats
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
