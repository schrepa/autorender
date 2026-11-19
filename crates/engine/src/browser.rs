use crate::GameHost;
use crate::game::{Game, GameEntry};
use engine_core::BrickList;
use engine_scene::SceneFiles;
use engine_settings::js_error as js;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

pub type SoundOrderList = js_sys::Array;

#[doc(hidden)]
pub struct SceneLoading {
    load: Option<crate::host::SceneLoad>,
}

impl SceneLoading {
    pub fn new(source: &str, bricks: fn(BrickList) -> BrickList) -> Result<Self, JsValue> {
        let kinds = engine_scene::SceneKinds::listed(&bricks(BrickList::default()));
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

    pub fn read(&mut self, uri: &str, bytes: Vec<u8>) -> Result<(), JsValue> {
        self.loading()?.read(uri, bytes).map_err(js)
    }

    pub fn finish(&mut self) -> Result<SceneFiles, JsValue> {
        let load = self.load.take().ok_or_else(ended)?;
        let (_, files) = load.opened().map_err(js)?;
        Ok(files)
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
        host.keep_in(game.name);
        match scene {
            Some((source, files)) => host.open_scene(&source, &files).map_err(js)?,
            None => {
                let kinds = engine_scene::SceneKinds::of(host.engine().app.world());
                let (document, files) = crate::host::SceneLoad::new(game.scene.text, &kinds)
                    .and_then(crate::host::SceneLoad::opened)
                    .map_err(js)?;
                host.open_document(&document, &files).map_err(js)?;
            }
        }
        host.engine_mut().exchange();
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

    pub fn take_sound(&mut self) -> SoundOrderList {
        use engine_core::SoundOrder;
        let out = js_sys::Array::new();
        let set = |object: &js_sys::Object, field: &str, value: JsValue| {
            let _ = js_sys::Reflect::set(object, &JsValue::from_str(field), &value);
        };
        for order in self.host.take_sound() {
            let object = js_sys::Object::new();
            match order {
                SoundOrder::Open { latency_seconds } => {
                    set(&object, "open", JsValue::TRUE);
                    set(
                        &object,
                        "latency",
                        latency_seconds.map_or(JsValue::NULL, |seconds| {
                            JsValue::from_f64(f64::from(seconds))
                        }),
                    );
                }
                SoundOrder::Decode { sound, bytes } => {
                    set(&object, "decode", JsValue::from(sound.0));
                    set(
                        &object,
                        "bytes",
                        js_sys::Uint8Array::from(&bytes[..]).into(),
                    );
                }
                SoundOrder::Start { voice, play } => {
                    set(&object, "start", JsValue::from(voice.0));
                    set(&object, "sound", JsValue::from(play.sound.0));
                    set(&object, "gain", JsValue::from(play.gain));
                    set(&object, "pan", JsValue::from(play.pan));
                    set(&object, "rate", JsValue::from(play.rate));
                    set(&object, "looping", JsValue::from(play.looping));
                    set(&object, "offset", JsValue::from(play.offset_seconds));
                }
                SoundOrder::Tune { voice, tune } => {
                    set(&object, "tune", JsValue::from(voice.0));
                    set(&object, "gain", JsValue::from(tune.gain));
                    set(&object, "pan", JsValue::from(tune.pan));
                    set(&object, "rate", JsValue::from(tune.rate));
                    set(&object, "seconds", JsValue::from(tune.seconds));
                }
                SoundOrder::Stop {
                    voice,
                    fade_seconds,
                } => {
                    set(&object, "stop", JsValue::from(voice.0));
                    set(&object, "seconds", JsValue::from(fade_seconds));
                }
                SoundOrder::Master { gain, seconds } => {
                    set(&object, "master", JsValue::from(gain));
                    set(&object, "seconds", JsValue::from(seconds));
                }
            }
            out.push(&object);
        }
        out
    }

    pub fn sound_opened(&mut self, running: bool, rate: u32, channels: u32, latency_seconds: f32) {
        self.host.hear_sound(engine_core::SoundReply::Opened {
            running,
            rate,
            channels,
            latency_seconds,
        });
    }

    pub fn sound_open_failed(&mut self, error: String) {
        self.host
            .hear_sound(engine_core::SoundReply::OpenFailed { error });
    }

    pub fn sound_decoded(&mut self, sound: u32, frames: f64, rate: u32, channels: u32) {
        let frames = if frames.is_finite() && frames >= 0.0 {
            frames as u64
        } else {
            0
        };
        self.host.hear_sound(engine_core::SoundReply::Decoded {
            sound: engine_core::SoundId(sound),
            length: engine_core::Length {
                frames,
                rate,
                channels,
            },
        });
    }

    pub fn sound_failed(&mut self, sound: u32, error: String) {
        self.host.hear_sound(engine_core::SoundReply::DecodeFailed {
            sound: engine_core::SoundId(sound),
            error,
        });
    }

    pub fn voice_ended(&mut self, voice: u32) {
        self.host.hear_sound(engine_core::SoundReply::Ended {
            voice: engine_core::VoiceId(voice),
        });
    }

    pub fn voice_dropped(&mut self, voice: u32, error: String) {
        self.host.hear_sound(engine_core::SoundReply::Dropped {
            voice: engine_core::VoiceId(voice),
            error,
        });
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
