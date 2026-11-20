use crate::GameHost;
use crate::game::{Game, GameEntry};
use engine_core::{BrickList, IoAct, IoOrder, IoReply};
use engine_settings::js_error as js;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

pub struct PageHost<G: Game> {
    host: GameHost<G>,
    landed: Landed,
    sound_bytes: Vec<u8>,
    #[cfg(feature = "dev")]
    pixels: Vec<u8>,
}

type Landed = std::rc::Rc<std::cell::RefCell<Vec<IoReply>>>;

const KEPT_PREFIX: &str = "kept-";

fn run_io(IoOrder { id, act }: IoOrder, landed: Landed) {
    wasm_bindgen_futures::spawn_local(async move {
        let done = match act {
            IoAct::Read(at) => fetched(&at)
                .await
                .map(Some)
                .map_err(|error| format!("{at}: {error}")),
            IoAct::Load(place, name) => load(&place, &name),
            IoAct::Store(place, name, bytes) => store(&place, &name, &bytes).map(|()| None),
        };
        landed.borrow_mut().push(IoReply { id, done });
    });
}

async fn fetched(at: &str) -> Result<Vec<u8>, String> {
    use wasm_bindgen_futures::JsFuture;
    let window = web_sys::window().ok_or("no window")?;
    let response = JsFuture::from(window.fetch_with_str(at))
        .await
        .map_err(|error| format!("{error:?}"))?;
    let response: web_sys::Response = response.dyn_into().map_err(|error| format!("{error:?}"))?;
    if !response.ok() {
        return Err(format!("status {}", response.status()));
    }
    let buffer = response
        .array_buffer()
        .map_err(|error| format!("{error:?}"))?;
    let buffer = JsFuture::from(buffer)
        .await
        .map_err(|error| format!("{error:?}"))?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

fn storage() -> Result<web_sys::Storage, String> {
    web_sys::window()
        .ok_or("no window")?
        .local_storage()
        .map_err(|error| format!("{error:?}"))?
        .ok_or_else(|| "no local storage".to_owned())
}

fn key(place: &str, name: &str) -> String {
    format!("{place}/{KEPT_PREFIX}{name}")
}

fn store(place: &str, name: &str, bytes: &[u8]) -> Result<(), String> {
    let text: String = bytes.iter().map(|&byte| char::from(byte)).collect();
    storage()?
        .set_item(&key(place, name), &text)
        .map_err(|error| format!("{}: {error:?}", key(place, name)))
}

fn load(place: &str, name: &str) -> Result<Option<Vec<u8>>, String> {
    let at = key(place, name);
    let Some(text) = storage()?
        .get_item(&at)
        .map_err(|error| format!("{at}: {error:?}"))?
    else {
        return Ok(None);
    };
    text.chars()
        .map(|unit| u8::try_from(u32::from(unit)).map_err(|_| format!("{at}: not bytes")))
        .collect::<Result<Vec<u8>, String>>()
        .map(Some)
}

#[cfg(feature = "dev")]
fn kept_listing(place: &str) -> Result<Vec<(String, u64)>, String> {
    let storage = storage()?;
    let failed = |error: JsValue| format!("{place}: {error:?}");
    let mut listed = Vec::new();
    for index in 0..storage.length().map_err(failed)? {
        let at = storage.key(index).map_err(failed)?.unwrap_or_default();
        if let Some(name) = at.strip_prefix(&key(place, "")) {
            let text = storage.get_item(&at).map_err(failed)?.unwrap_or_default();
            listed.push((name.to_owned(), text.chars().count() as u64));
        }
    }
    listed.sort();
    Ok(listed)
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
        game: GameEntry,
        scene: &str,
    ) -> Result<Self, JsValue> {
        let bricks = (game.bricks)(BrickList::default());
        let target = engine_render::RenderTarget::Canvas(canvas);
        let mut host = GameHost::launch(target, G::start(), bricks, game.linked)
            .await
            .map_err(js)?;
        #[cfg(feature = "dev")]
        host.engine_mut()
            .app
            .insert_resource(engine_dev::Instruments {
                names: crate::host::instruments(),
                switch: crate::host::instrument,
            });
        #[cfg(feature = "dev")]
        engine_registry::register(
            host.engine_mut().app.world_mut(),
            crate::host::kept_entry(kept_listing),
        );
        host.keep_in(game.name);
        host.open_scene(scene).map_err(js)?;
        Ok(Self {
            host,
            landed: Landed::default(),
            sound_bytes: Vec::new(),
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
        for reply in self.landed.take() {
            self.host.land_io(reply);
        }
        let phase = self
            .host
            .animation_frame(timestamp, width, height)
            .map_err(js)?;
        Ok(phase as u32)
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

    pub fn cursor_mode(&self) -> u32 {
        self.host.cursor_mode() as u32
    }

    pub fn take_sound(&mut self) -> Vec<f64> {
        use engine_core::SoundOrder;
        let mut orders = Vec::new();
        self.sound_bytes.clear();
        for order in self.host.take_sound() {
            match order {
                SoundOrder::Open { latency_seconds } => orders.extend([
                    0.0,
                    f64::from(u8::from(latency_seconds.is_some())),
                    latency_seconds.map_or(0.0, f64::from),
                ]),
                SoundOrder::Decode { sound, bytes } => {
                    self.sound_bytes.extend_from_slice(&bytes);
                    orders.extend([1.0, f64::from(sound.0), bytes.len() as f64]);
                }
                SoundOrder::Start { voice, play } => orders.extend([
                    2.0,
                    f64::from(voice.0),
                    f64::from(play.sound.0),
                    f64::from(play.gain),
                    f64::from(play.pan),
                    f64::from(play.rate),
                    f64::from(u8::from(play.looping)),
                    f64::from(play.offset_seconds),
                ]),
                SoundOrder::Tune { voice, tune } => orders.extend([
                    3.0,
                    f64::from(voice.0),
                    f64::from(tune.gain),
                    f64::from(tune.pan),
                    f64::from(tune.rate),
                    f64::from(tune.seconds),
                ]),
                SoundOrder::Stop {
                    voice,
                    fade_seconds,
                } => orders.extend([4.0, f64::from(voice.0), f64::from(fade_seconds)]),
                SoundOrder::Master { gain, seconds } => {
                    orders.extend([5.0, f64::from(gain), f64::from(seconds)]);
                }
            }
        }
        for order in self.host.take_io() {
            run_io(order, std::rc::Rc::clone(&self.landed));
        }
        orders
    }

    pub fn take_sound_bytes(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.sound_bytes)
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

    pub fn reload_scene(&mut self, source: &str) -> Result<(), JsValue> {
        self.host.open_scene(source).map_err(js)
    }

    pub fn files(&self) -> String {
        let listed: Vec<serde_json::Value> = self
            .host
            .files()
            .into_iter()
            .map(|(uri, group)| serde_json::json!({"uri": uri, "group": group}))
            .collect();
        serde_json::Value::from(listed).to_string()
    }

    pub fn file(&mut self, uri: &str, bytes: Vec<u8>) -> Result<(), JsValue> {
        self.host.file(uri, Ok(bytes)).map_err(js)
    }

    pub fn file_failed(&mut self, uri: &str, error: &str) -> Result<(), JsValue> {
        self.host.file(uri, Err(page_error(error))).map_err(js)
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
        self.host.time_frame();
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

    pub fn dev_reloaded(&mut self, source: &str) {
        self.host.dev_reloaded(source);
    }

    pub fn dev_reload_ended(&mut self, error: Option<String>) {
        let outcome = error.map_or(Ok(()), |error| Err(page_error(&error)));
        self.host.dev_reload_ended(outcome);
    }
}

fn page_error(error: &str) -> crate::Error {
    engine_settings::parse_json::<crate::Error>(error).unwrap_or_else(|error| error.within("$.error"))
}
