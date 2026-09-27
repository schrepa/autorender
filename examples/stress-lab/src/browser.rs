use engine::browser::{BrowserApp, BrowserStats};
use engine::prelude::*;
use wasm_bindgen::prelude::*;

use crate::{
    FrameMetrics, StressConfig, StressScene, create_workload, lamp_intensity, look, set_view,
    stress_observation, stress_shafts,
};

#[wasm_bindgen]
pub struct StressGame {
    app: BrowserApp,
    generation: u32,
    // The lighting preset and effect selection, kept across workload changes.
    look: String,
    effects: String,
    #[cfg(feature = "dev")]
    development: engine::dev_browser::DevBrowser,
}

#[wasm_bindgen]
impl StressGame {
    pub async fn create(
        canvas: web_sys::HtmlCanvasElement,
        config_json: &str,
    ) -> Result<Self, JsValue> {
        console_error_panic_hook::set_once();
        let engine = create_workload(StressConfig::from_json(config_json).map_err(js_error)?)
            .map_err(js_error)?;
        Ok(Self {
            app: BrowserApp::new(canvas, engine).await?,
            generation: 1,
            look: "scene".into(),
            effects: "all".into(),
            #[cfg(feature = "dev")]
            development: development_tools(),
        })
    }

    /// Build first, then replace the complete Engine at a browser frame boundary.
    /// Asset library IDs separate old renderer snapshots from the new world.
    pub fn configure(&mut self, config_json: &str) -> Result<(), JsValue> {
        let engine = create_workload(StressConfig::from_json(config_json).map_err(js_error)?)
            .map_err(js_error)?;
        self.app.engine = engine;
        self.generation = self.generation.saturating_add(1);
        self.set_look(&self.look.clone())?;
        self.set_effects(&self.effects.clone())?;
        self.suspend();
        Ok(())
    }

    /// Rendering effects: `all`, `none`, or comma-separated `no-<effect>`
    /// exclusions (decisions 020 to 026 and 038).
    pub fn set_effects(&mut self, name: &str) -> Result<(), JsValue> {
        // `all`, `none`, or comma-separated `no-<effect>` exclusions.
        const EFFECTS: [&str; 8] = [
            "shadows",
            "occlusion",
            "bloom",
            "lights",
            "shafts",
            "reflections",
            "particles",
            "refraction",
        ];
        let mut on = [true; EFFECTS.len()];
        match name {
            "all" => {}
            "none" => on = [false; EFFECTS.len()],
            _ => {
                for token in name.split(',') {
                    let excluded = token
                        .strip_prefix("no-")
                        .and_then(|effect| EFFECTS.iter().position(|name| *name == effect));
                    match excluded {
                        Some(index) => on[index] = false,
                        None => {
                            return Err(js_error(format!(
                                "effects must be all, none or comma-separated no-<effect> for {}",
                                EFFECTS.join(", ")
                            )));
                        }
                    }
                }
            }
        }
        let [
            shadows,
            occlusion,
            bloom,
            lights,
            shafts,
            reflections,
            particles,
            refraction,
        ] = on;
        let scene = self.app.engine.app.world().resource::<StressConfig>().scene;
        let lamps = lamp_intensity(scene, &self.look).map_err(js_error)?;
        let world = self.app.engine.app.world_mut();
        world.resource_mut::<Shadows>().enabled = shadows;
        world.resource_mut::<AmbientOcclusion>().intensity = if occlusion {
            AmbientOcclusion::default().intensity
        } else {
            0.0
        };
        world.resource_mut::<Bloom>().intensity = if bloom {
            Bloom::default().intensity
        } else {
            0.0
        };
        // Opt-in effects are removed entirely when off.
        if shafts {
            world.insert_resource(stress_shafts());
        } else {
            world.remove_resource::<SunShafts>();
        }
        if reflections {
            world.insert_resource(Reflections::default());
        } else {
            world.remove_resource::<Reflections>();
        }
        {
            let mut local = world.resource_mut::<LocalLights>();
            local.enabled = lights;
            local.intensity = lamps;
        }
        world.resource_mut::<Particles>().enabled = particles;
        // Only the metro has glass that refracts (decision 038).
        if refraction && scene == StressScene::Metro {
            world.insert_resource(Refraction::default());
        } else {
            world.remove_resource::<Refraction>();
        }
        self.effects = name.into();
        Ok(())
    }

    /// Lighting preset: `scene` (the scene's own), `golden`, `dusk`,
    /// `night` or `day`.
    pub fn set_look(&mut self, name: &str) -> Result<(), JsValue> {
        let scene = self.app.engine.app.world().resource::<StressConfig>().scene;
        let atmosphere = look(scene, name).map_err(js_error)?;
        let lamps = lamp_intensity(scene, name).map_err(js_error)?;
        self.app.engine.app.insert_resource(atmosphere);
        if let Some(mut local) = self
            .app
            .engine
            .app
            .world_mut()
            .get_resource_mut::<LocalLights>()
        {
            local.intensity = lamps;
        }
        self.look = name.into();
        Ok(())
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
        _held: u32,
        _pressed: u32,
        _released: u32,
    ) -> Result<(), JsValue> {
        self.app
            .engine
            .app
            .world_mut()
            .resource_mut::<FrameMetrics>()
            .begin_frame();
        self.app.frame(
            timestamp,
            width,
            height,
            InputState {
                move_forward: forward,
                move_right: right,
                move_up: up,
                look_yaw: yaw,
                look_pitch: pitch,
            },
        )
    }

    pub fn stats(&self) -> BrowserStats {
        self.app.stats().into()
    }
    pub fn stress_stats(&self) -> String {
        stress_observation(&self.app.engine, self.generation).to_string()
    }
    pub fn set_view(&mut self, view: &str) -> Result<(), JsValue> {
        set_view(&mut self.app.engine, view).map_err(js_error)
    }
    pub fn set_culling(&mut self, enabled: bool) {
        self.app.engine.set_frustum_culling(enabled);
        self.app
            .engine
            .app
            .world_mut()
            .resource_mut::<StressConfig>()
            .culling = enabled;
    }
    /// GPU (default) or exact CPU presentation of ordinary frames.
    pub fn set_gpu_presentation(&mut self, enabled: bool) {
        self.app.set_gpu_presentation(enabled);
    }
    pub fn gpu_presentation(&self) -> bool {
        self.app.gpu_presentation()
    }
    /// Splits fixed ticks across frames around their due time (default).
    pub fn set_tick_pipelining(&mut self, enabled: bool) {
        self.app.set_tick_pipelining(enabled);
    }
    pub fn tick_pipelining(&self) -> bool {
        self.app.tick_pipelining()
    }
    /// Temporal antialiasing of GPU-presented frames (off by default).
    pub fn set_temporal_antialiasing(&mut self, enabled: bool) {
        self.app.set_temporal_antialiasing(enabled);
    }
    pub fn temporal_antialiasing(&self) -> bool {
        self.app.temporal_antialiasing()
    }
    /// Occlusion culling of GPU-presented frames (on by default).
    pub fn set_occlusion_culling(&mut self, enabled: bool) {
        self.app.set_occlusion_culling(enabled);
    }
    pub fn occlusion_culling(&self) -> bool {
        self.app.occlusion_culling()
    }
    pub fn reset_camera(&mut self) {
        set_view(&mut self.app.engine, "overview").expect("known view");
    }
    pub fn suspend(&mut self) {
        self.app.suspend();
        #[cfg(feature = "dev")]
        self.development.clear_input();
    }
}

#[cfg(feature = "dev")]
#[wasm_bindgen]
impl StressGame {
    pub fn dev_command(&mut self, request: &str, width: u32, height: u32) -> String {
        // Read-only polls retain the timings belonging to the last frame.
        let submitted = serde_json::from_str::<serde_json::Value>(request)
            .ok()
            .and_then(|value| {
                value
                    .get("op")
                    .and_then(|op| op.as_str())
                    .map(|op| matches!(op, "step" | "render" | "present" | "camera" | "culling"))
            })
            == Some(true);
        let previous = *self.app.engine.app.world().resource::<FrameMetrics>();
        if submitted {
            self.app
                .engine
                .app
                .world_mut()
                .resource_mut::<FrameMetrics>()
                .begin_frame();
        }
        let reply = self
            .development
            .command(&mut self.app, None, request, width, height);
        if submitted {
            if serde_json::from_str::<serde_json::Value>(&reply)
                .ok()
                .is_some_and(|value| value["ok"] == true)
            {
                let enabled = self.app.engine.frustum_culling();
                self.app
                    .engine
                    .app
                    .world_mut()
                    .resource_mut::<StressConfig>()
                    .culling = enabled;
            } else {
                *self
                    .app
                    .engine
                    .app
                    .world_mut()
                    .resource_mut::<FrameMetrics>() = previous;
            }
        }
        reply
    }
}

fn js_error(message: String) -> JsValue {
    JsValue::from_str(&message)
}

#[cfg(feature = "profiling")]
#[wasm_bindgen]
impl StressGame {
    pub fn profiling_start(&mut self, max_events: u32) -> Result<(), JsValue> {
        self.app
            .profiling_start(max_events as usize)
            .map_err(JsValue::from_str)
    }
    pub fn profiling_stop(&mut self) -> Result<String, JsValue> {
        self.app
            .profiling_stop()
            .map(|report| report.to_string())
            .map_err(JsValue::from_str)
    }
    pub fn profiling_poll(&mut self) {
        self.app.profiling_poll();
    }
    pub fn profiling_status(&self) -> String {
        self.app.profiling_status().to_string()
    }
    pub fn profiling_report(&mut self) -> String {
        self.app.profiling_report().to_string()
    }
}

#[cfg(feature = "dev")]
fn development_tools() -> engine::dev_browser::DevBrowser {
    use engine::dev::InspectorDescriptor;
    let mut development = engine::dev_browser::DevBrowser::default();
    development.tools.register_resource::<StressConfig>(
        InspectorDescriptor::new("StressConfig", "Applied deterministic stress workload; changes rebuild its runtime", serde_json::json!({
            "type":"object", "additionalProperties":false,
            "properties":{
                "seed":{"type":"integer","minimum":0,"maximum":4294967295u64},
                "renderables":{"type":"integer","minimum":1,"maximum":4000000},
                "moving_percent":{"type":"integer","minimum":0,"maximum":100},
                "hierarchy":{"type":"boolean"}, "interpolation":{"type":"boolean"},
                "colliders":{"type":"integer","minimum":0,"maximum":10000},
                "sensor_percent":{"type":"integer","minimum":0,"maximum":100},
                "dense_colliders":{"type":"boolean"},
                "queries_per_tick":{"type":"integer","minimum":0,"maximum":512},
                "mesh_variety":{"enum":[1,8,32,64]}, "textured":{"type":"boolean"}, "culling":{"type":"boolean"},
                "scene":{"enum":["field","city","metro"]}, "compressed_textures":{"type":"boolean"}
            },
            "required":["seed","renderables","moving_percent","hierarchy","interpolation","colliders",
                "sensor_percent","dense_colliders","queries_per_tick","mesh_variety","textured","culling",
                "scene","compressed_textures"]
        })),
        |config| serde_json::to_value(config).expect("configuration contains finite integer values"),
    ).expect("stress inspector registration is valid");
    development.tools.register_resource::<FrameMetrics>(
        InspectorDescriptor::new("StressMetrics", "Last frame CPU simulation subsets and bounded spatial query counters", serde_json::json!({
            "type":"object", "additionalProperties":false,
            "properties":{
                "sync_pre_ms":{"type":"number","minimum":0},
                "sync_post_ms":{"type":"number","minimum":0},
                "sensor_ms":{"type":"number","minimum":0},
                "queries_ms":{"type":"number","minimum":0},
                "animation_ms":{"type":"number","minimum":0},
                "query_calls":{"type":"integer","minimum":0},
                "query_hits":{"type":"integer","minimum":0},
                "query_errors":{"type":"integer","minimum":0},
                "total_calls":{"type":"string","pattern":"^[0-9]+$"},
                "total_hits":{"type":"string","pattern":"^[0-9]+$"},
                "total_errors":{"type":"string","pattern":"^[0-9]+$"},
                "last_error":{"type":["string","null"]}
            },
            "required":["sync_pre_ms","sync_post_ms","sensor_ms","queries_ms","animation_ms",
                "query_calls","query_hits","query_errors","total_calls","total_hits","total_errors","last_error"]
        })),
        |metrics| serde_json::json!({
            "sync_pre_ms":metrics.sync_pre_ms,"sync_post_ms":metrics.sync_post_ms,
            "sensor_ms":metrics.sensor_ms,"queries_ms":metrics.queries_ms,"animation_ms":metrics.animation_ms,
            "query_calls":metrics.query_calls,"query_hits":metrics.query_hits,"query_errors":metrics.query_errors,
            "total_calls":metrics.total_calls.to_string(),"total_hits":metrics.total_hits.to_string(),
            "total_errors":metrics.total_errors.to_string(),"last_error":metrics.last_error,
        }),
    ).expect("stress metrics registration is valid");
    development
}
