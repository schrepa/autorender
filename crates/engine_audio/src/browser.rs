use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Arc;

use engine_core::Ticket;
use engine_places_rendering::audio::Bus;
use glam::Vec3;
use js_sys::Uint8Array;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::{
    AudioBuffer, AudioBufferSourceNode, AudioContext, AudioContextState, AudioParam,
    DistanceModelType, GainNode, PannerNode, PanningModelType,
};

use crate::mixer::Start;

const GLIDE: f64 = 0.003;
const FADE_OUT: f64 = 0.02;

thread_local! {
    static PAGE: RefCell<Page> = RefCell::new(Page::default());
}

fn text(failure: &JsValue) -> String {
    format!("{failure:?}")
}

fn glide(param: &AudioParam, value: f32, now: f64) {
    if param.set_target_at_time(value, now, GLIDE).is_err() {
        param.set_value(value);
    }
}

type Arrival = ((u32, Ticket), Result<Decoded, String>);

pub(crate) struct Decoded(u32);

impl Drop for Decoded {
    fn drop(&mut self) {
        let _ = PAGE.try_with(|page| {
            if let Ok(mut page) = page.try_borrow_mut() {
                page.buffers.remove(&self.0);
            }
        });
    }
}

struct Voice {
    source: AudioBufferSourceNode,
    gain: GainNode,
    panner: Option<PannerNode>,
    bus: Bus,
    volume: f32,
    rate: f32,
    heard: bool,
    paused: bool,
    ended: bool,
    _on_end: Closure<dyn FnMut()>,
}

impl Voice {
    fn apply(&self, now: f64) {
        let volume = if self.heard { self.volume } else { 0.0 };
        let rate = if self.paused { 0.0 } else { self.rate };
        glide(&self.gain.gain(), volume, now);
        self.source.playback_rate().set_value(rate);
    }
}

struct Graph {
    context: AudioContext,
    buses: [GainNode; Bus::ALL.len()],
}

fn bus_node(context: &AudioContext, master: &GainNode) -> Result<GainNode, JsValue> {
    let node = context.create_gain()?;
    node.connect_with_audio_node(master)?;
    Ok(node)
}

fn spatial_node(context: &AudioContext, radius: f32) -> Result<PannerNode, JsValue> {
    let panner = context.create_panner()?;
    panner.set_panning_model(PanningModelType::Equalpower);
    panner.set_distance_model(DistanceModelType::Linear);
    panner.set_ref_distance(0.0);
    panner.set_max_distance(f64::from(radius));
    panner.set_rolloff_factor(1.0);
    Ok(panner)
}

impl Graph {
    fn new() -> Result<Self, JsValue> {
        let context = AudioContext::new()?;
        let master = context.create_gain()?;
        master.connect_with_audio_node(&context.destination())?;
        let buses = [
            master.clone(),
            bus_node(&context, &master)?,
            bus_node(&context, &master)?,
            bus_node(&context, &master)?,
            bus_node(&context, &master)?,
        ];
        Ok(Self { context, buses })
    }
}

#[derive(Default)]
struct Page {
    graph: Option<Graph>,
    buffers: BTreeMap<u32, AudioBuffer>,
    arrived: Vec<Arrival>,
    voices: BTreeMap<u32, Voice>,
    paused: [bool; Bus::ALL.len()],
}

impl Page {
    fn graph(&mut self) -> Result<&Graph, JsValue> {
        if self.graph.is_none() {
            self.graph = Some(Graph::new()?);
        }
        self.graph.as_ref().ok_or(JsValue::NULL)
    }

    fn now(&self) -> f64 {
        self.graph
            .as_ref()
            .map_or(0.0, |graph| graph.context.current_time())
    }

    fn start(&mut self, voice: u32, sound: &Decoded, play: &Start) -> Result<(), JsValue> {
        let graph = self
            .graph
            .as_ref()
            .ok_or_else(|| JsValue::from_str("no_device"))?;
        let Some(buffer) = self.buffers.get(&sound.0) else {
            return Err(JsValue::from_str("sound_missing"));
        };
        let context = &graph.context;
        let out = &graph.buses[play.bus.index()];
        let paused = self.paused[play.bus.index()];
        let heard = play.radius.is_none();
        let source = context.create_buffer_source()?;
        source.set_buffer(Some(buffer));
        source.set_loop(play.looping);
        source
            .playback_rate()
            .set_value(if paused { 0.0 } else { play.pitch });
        let gain = context.create_gain()?;
        gain.gain().set_value(if heard { play.volume } else { 0.0 });
        source.connect_with_audio_node(&gain)?;
        let panner = match play.radius {
            Some(radius) => {
                let panner = spatial_node(context, radius)?;
                gain.connect_with_audio_node(&panner)?;
                panner.connect_with_audio_node(out)?;
                Some(panner)
            }
            None => {
                gain.connect_with_audio_node(out)?;
                None
            }
        };
        let on_end = Closure::<dyn FnMut()>::new(move || {
            PAGE.with_borrow_mut(|page| {
                if let Some(held) = page.voices.get_mut(&voice) {
                    held.ended = true;
                }
            });
        });
        source.set_onended(Some(on_end.as_ref().unchecked_ref()));
        source.start()?;
        self.voices.insert(
            voice,
            Voice {
                source,
                gain,
                panner,
                bus: play.bus,
                volume: play.volume,
                rate: play.pitch,
                heard,
                paused,
                ended: false,
                _on_end: on_end,
            },
        );
        Ok(())
    }

    fn tick(&mut self) {
        let Some(graph) = self.graph.as_ref() else {
            return;
        };
        if graph.context.state() == AudioContextState::Suspended {
            let _ = graph.context.resume();
        }
    }
}

fn begin_decode(sound: (u32, Ticket), bytes: &[u8]) -> Result<(), JsValue> {
    let promise = PAGE.with_borrow_mut(|page| {
        let graph = page.graph()?;
        graph
            .context
            .decode_audio_data(&Uint8Array::from(bytes).buffer())
    })?;
    spawn_local(async move {
        let decoded = JsFuture::from(promise).await;
        PAGE.with_borrow_mut(|page| {
            let arrived = match decoded {
                Ok(buffer) => {
                    page.buffers.insert(sound.0, buffer.unchecked_into());
                    Ok(Decoded(sound.0))
                }
                Err(failure) => Err(text(&failure)),
            };
            page.arrived.push((sound, arrived));
        });
    });
    Ok(())
}

#[derive(Default)]
pub(crate) struct Backend {
    opened: bool,
}

impl Backend {
    pub(crate) fn open(&mut self) -> Result<(), String> {
        PAGE.with_borrow_mut(|page| page.graph().map(|_| ()))
            .map_err(|failure| text(&failure))?;
        self.opened = true;
        Ok(())
    }

    pub(crate) fn opened(&self) -> bool {
        self.opened
    }

    pub(crate) fn close(&mut self) {
        self.opened = false;
    }

    pub(crate) fn decode(
        &mut self,
        sound: (u32, Ticket),
        bytes: Arc<[u8]>,
        _stream: bool,
        may_open: bool,
    ) -> Option<Result<Decoded, String>> {
        if !may_open {
            return Some(Err("device_refused".to_owned()));
        }
        begin_decode(sound, &bytes)
            .err()
            .map(|failure| Err(text(&failure)))
    }

    pub(crate) fn arrivals(&mut self) -> Vec<Arrival> {
        PAGE.with_borrow_mut(|page| std::mem::take(&mut page.arrived))
    }

    pub(crate) fn start(&mut self, voice: u32, sound: &Decoded, play: &Start) -> Result<(), String> {
        PAGE.with_borrow_mut(|page| page.start(voice, sound, play))
            .map_err(|failure| text(&failure))
    }

    pub(crate) fn stop(&mut self, voice: u32) {
        PAGE.with_borrow_mut(|page| {
            let now = page.now();
            if let Some(voice) = page.voices.remove(&voice) {
                voice.source.set_onended(None);
                glide(&voice.gain.gain(), 0.0, now);
                let _ = web_sys::AudioScheduledSourceNode::stop_with_when(
                    &voice.source,
                    now + FADE_OUT,
                );
            }
        });
    }

    pub(crate) fn finished(&self, voice: u32) -> bool {
        PAGE.with_borrow(|page| {
            page.voices
                .get(&voice)
                .is_none_or(|voice| voice.ended)
        })
    }

    pub(crate) fn set_voice(&mut self, voice: u32, volume: f32, pitch: f32) {
        PAGE.with_borrow_mut(|page| {
            let now = page.now();
            if let Some(voice) = page.voices.get_mut(&voice) {
                voice.volume = volume;
                voice.rate = pitch;
                voice.apply(now);
            }
        });
    }

    pub(crate) fn place(&mut self, voice: u32, at: Option<Vec3>, _eye: Vec3) {
        PAGE.with_borrow_mut(|page| {
            let now = page.now();
            let Some(voice) = page.voices.get_mut(&voice) else {
                return;
            };
            if let (Some(panner), Some(at)) = (voice.panner.as_ref(), at) {
                panner.set_position(f64::from(at.x), f64::from(at.y), f64::from(at.z));
            }
            if voice.heard != at.is_some() {
                voice.heard = at.is_some();
                voice.apply(now);
            }
        });
    }

    pub(crate) fn listen(&mut self, eye: Vec3, forward: Vec3, up: Vec3) {
        PAGE.with_borrow(|page| {
            let Some(graph) = page.graph.as_ref() else {
                return;
            };
            let listener = graph.context.listener();
            listener.set_position(f64::from(eye.x), f64::from(eye.y), f64::from(eye.z));
            listener.set_orientation(
                f64::from(forward.x),
                f64::from(forward.y),
                f64::from(forward.z),
                f64::from(up.x),
                f64::from(up.y),
                f64::from(up.z),
            );
        });
    }

    pub(crate) fn set_bus(&mut self, bus: Bus, volume: f32, paused: bool) {
        PAGE.with_borrow_mut(|page| {
            let now = page.now();
            page.paused[bus.index()] = paused;
            if let Some(graph) = page.graph.as_ref() {
                let volume = if paused { 0.0 } else { volume };
                glide(&graph.buses[bus.index()].gain(), volume, now);
            }
            for voice in page.voices.values_mut().filter(|voice| voice.bus == bus) {
                voice.paused = paused;
                voice.apply(now);
            }
        });
    }

    pub(crate) fn tick(&mut self) {
        PAGE.with_borrow_mut(Page::tick);
    }
}
