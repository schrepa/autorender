use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use bevy_ecs::prelude::*;
use engine_core::math::clamp;
use engine_core::{
    Error, ErrorCode, FrameTime, GlobalTransforms, Interest, Pause, Poses, Request, Requests,
    SceneIds, Spawner, Ticket, Views,
};
use engine_places_rendering::audio::{
    Bus, OneShot, SoundHandle, SoundOrder, SoundSource, Sounds, VoiceHandle,
};
use glam::{Mat4, Vec3};
use serde::Serialize;

use crate::AudioBrick;
use crate::backend::{Backend, Decoded};
use crate::scene::{MAX_PITCH, MAX_RADIUS, MIN_PITCH, MIN_RADIUS};
use crate::settings::{Audio, AudioStats};

pub const AUDIO_FAILED: ErrorCode = ErrorCode::brick("audio_failed");

static NEXT_ID: AtomicU32 = AtomicU32::new(1);

fn fresh() -> u32 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

fn bounded(value: f32, low: f32, high: f32) -> f32 {
    if value.is_nan() {
        low
    } else {
        clamp(value, low, high)
    }
}

fn level(volume: f32) -> f32 {
    bounded(volume, 0.0, 1.0)
}

fn rate(pitch: f32) -> f32 {
    bounded(pitch, MIN_PITCH, MAX_PITCH)
}

fn same(held: &Arc<[u8]>, wanted: &Arc<[u8]>) -> bool {
    Arc::ptr_eq(held, wanted) || held[..] == wanted[..]
}

const fn pauses(bus: Bus) -> bool {
    match bus {
        Bus::Effects | Bus::Ambient | Bus::Voice => true,
        Bus::Master | Bus::Music => false,
    }
}

pub(crate) struct Start {
    pub(crate) bus: Bus,
    pub(crate) volume: f32,
    pub(crate) pitch: f32,
    pub(crate) looping: bool,
    pub(crate) radius: Option<f32>,
}

impl Start {
    fn of_source(source: &SoundSource) -> Self {
        Self {
            bus: source.bus,
            volume: level(source.volume),
            pitch: rate(source.pitch),
            looping: source.looping,
            radius: source
                .spatial
                .then(|| bounded(source.radius, MIN_RADIUS, MAX_RADIUS)),
        }
    }

    fn of_shot(shot: &OneShot) -> Self {
        Self {
            bus: shot.bus,
            volume: level(shot.volume),
            pitch: rate(shot.pitch),
            looping: false,
            radius: shot
                .at
                .map(|_| bounded(shot.radius, MIN_RADIUS, MAX_RADIUS)),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Device {
    #[default]
    Allowed,
    Off,
    Refused,
    Failed,
}

struct File {
    bytes: Arc<[u8]>,
    sound: Interest<u32>,
}

enum Playing {
    Waiting(Interest<u32>),
    Started(Interest<u32>),
    Ended,
}

impl Playing {
    fn started(self) -> Self {
        match self {
            Self::Waiting(sound) | Self::Started(sound) => Self::Started(sound),
            Self::Ended => Self::Ended,
        }
    }
}

#[derive(Clone, Copy, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum VoiceState {
    Waiting,
    Playing,
    Ended,
}

#[derive(PartialEq)]
struct Key {
    file: String,
    bus: Bus,
    looping: bool,
    spatial: bool,
    radius: f32,
    stream: bool,
}

impl Key {
    fn of(source: &SoundSource) -> Self {
        Self {
            file: source.file.clone(),
            bus: source.bus,
            looping: source.looping,
            spatial: source.spatial,
            radius: source.radius,
            stream: source.stream,
        }
    }
}

enum Owner {
    Source(Entity, Key),
    Handle(VoiceHandle),
}

struct Voice {
    id: u32,
    owner: Owner,
    play: Start,
    at: Option<Vec3>,
    playing: Playing,
}

impl Voice {
    fn new(owner: Owner, sound: Interest<u32>, play: Start, at: Option<Vec3>) -> Self {
        Self {
            id: fresh(),
            owner,
            play,
            at,
            playing: Playing::Waiting(sound),
        }
    }

    fn shot(voice: VoiceHandle, sound: Interest<u32>, shot: &OneShot) -> Self {
        Self::new(Owner::Handle(voice), sound, Start::of_shot(shot), shot.at)
    }

    fn started(&self) -> bool {
        matches!(self.playing, Playing::Started(_))
    }

    fn live(&self) -> bool {
        !matches!(self.playing, Playing::Ended)
    }

    fn sound(&self) -> Option<u32> {
        match &self.playing {
            Playing::Waiting(sound) | Playing::Started(sound) => Some(*sound.key()),
            Playing::Ended => None,
        }
    }

    fn state(&self) -> VoiceState {
        match self.playing {
            Playing::Waiting(_) => VoiceState::Waiting,
            Playing::Started(_) => VoiceState::Playing,
            Playing::Ended => VoiceState::Ended,
        }
    }

    fn ends(&self) -> bool {
        match self.owner {
            Owner::Source(..) => !self.play.looping,
            Owner::Handle(_) => false,
        }
    }

    fn of_entity(&self, entity: Entity) -> bool {
        matches!(&self.owner, Owner::Source(owner, _) if *owner == entity)
    }

    fn of_handle(&self, handle: VoiceHandle) -> bool {
        matches!(&self.owner, Owner::Handle(owner) if *owner == handle)
    }

    fn holds(&self, key: &Key) -> bool {
        matches!(&self.owner, Owner::Source(_, held) if held == key)
    }
}

#[derive(Serialize, schemars::JsonSchema)]
pub(crate) struct VoiceView {
    entity: Option<String>,
    bus: Bus,
    volume: f32,
    pitch: f32,
    looping: bool,
    spatial: bool,
    state: VoiceState,
    at: Option<[f32; 3]>,
}

#[derive(Resource, Default)]
pub(crate) struct Mixer {
    backend: Backend,
    device: Device,
    voices: Vec<Voice>,
    sounds: Requests<u32, Decoded>,
    files: BTreeMap<(String, bool), File>,
    loaded: BTreeMap<SoundHandle, Interest<u32>>,
    cut: u32,
    failures: u32,
    latest: Option<String>,
}

impl Drop for Mixer {
    fn drop(&mut self) {
        for voice in &self.voices {
            self.backend.stop(voice.id);
        }
    }
}

impl Mixer {
    fn live(&self) -> usize {
        self.voices.iter().filter(|voice| voice.live()).count()
    }

    fn room(&self, limit: u32) -> bool {
        self.device == Device::Allowed && self.live() < limit as usize
    }

    fn turn(&mut self, next: Device) {
        if next == self.device {
            return;
        }
        match next {
            Device::Off | Device::Refused => self.shut(),
            Device::Allowed | Device::Failed => {}
        }
        self.device = next;
    }

    pub(crate) fn set_available(&mut self, available: bool) {
        let next = match (available, self.device) {
            (_, Device::Off) => Device::Off,
            (false, Device::Allowed | Device::Refused | Device::Failed) => Device::Refused,
            (true, Device::Allowed | Device::Refused | Device::Failed) => Device::Allowed,
        };
        self.turn(next);
    }

    fn set_enabled(&mut self, enabled: bool) {
        let next = match (enabled, self.device) {
            (false, _) => Device::Off,
            (true, Device::Off) => Device::Allowed,
            (true, held @ (Device::Allowed | Device::Refused | Device::Failed)) => held,
        };
        self.turn(next);
    }

    pub(crate) fn listed(&self, ids: &SceneIds<'_, '_>) -> Vec<VoiceView> {
        self.voices
            .iter()
            .map(|voice| VoiceView {
                entity: match &voice.owner {
                    Owner::Source(entity, _) => ids.id(*entity).map(str::to_owned),
                    Owner::Handle(_) => None,
                },
                bus: voice.play.bus,
                volume: voice.play.volume,
                pitch: voice.play.pitch,
                looping: voice.play.looping,
                spatial: voice.play.radius.is_some(),
                state: voice.state(),
                at: voice.at.map(|at| at.to_array()),
            })
            .collect()
    }

    fn device(&mut self) -> bool {
        if self.backend.opened() {
            return true;
        }
        match self.device {
            Device::Off | Device::Refused | Device::Failed => false,
            Device::Allowed => match self.backend.open() {
                Ok(()) => true,
                Err(failure) => {
                    self.latest = Some(failure);
                    self.turn(Device::Failed);
                    false
                }
            },
        }
    }

    fn shut(&mut self) {
        for voice in &self.voices {
            if voice.live() {
                self.backend.stop(voice.id);
            }
        }
        self.voices.retain(|voice| !voice.live());
        self.backend.close();
    }

    fn fail(&mut self, failure: String) {
        self.failures = self.failures.saturating_add(1);
        self.latest = Some(failure);
    }

    fn cut_one(&mut self) {
        self.cut = self.cut.saturating_add(1);
    }

    fn decode(&mut self, bytes: Arc<[u8]>, stream: bool) -> Interest<u32> {
        let sound = self.sounds.ask(fresh());
        let key = *sound.key();
        if let Some(ticket) = self.sounds.ticket(&key) {
            let may_open = !matches!(self.device, Device::Refused | Device::Failed);
            if let Some(decoded) = self.backend.decode((key, ticket), bytes, stream, may_open) {
                self.arrive((key, ticket), decoded);
            }
        }
        sound
    }

    fn arrive(&mut self, (key, ticket): (u32, Ticket), decoded: Result<Decoded, String>) {
        match decoded {
            Ok(decoded) => {
                self.sounds.ready(&key, ticket, decoded);
            }
            Err(failure) => {
                let error =
                    Error::new(AUDIO_FAILED, "$.audio.sounds").got(failure.clone());
                if self.sounds.fail(&key, ticket, error) {
                    self.fail(failure);
                }
            }
        }
    }

    fn prune(&mut self, named: &BTreeSet<(&str, bool)>) {
        self.files
            .retain(|(file, stream), _| named.contains(&(file.as_str(), *stream)));
    }

    fn sound_of(&mut self, source: &SoundSource) -> Option<Interest<u32>> {
        let key = (source.file.clone(), source.stream);
        if let Some(held) = self.files.get(&key)
            && source
                .bytes
                .as_ref()
                .is_none_or(|wanted| same(&held.bytes, wanted))
        {
            return Some(held.sound.clone());
        }
        let Some(bytes) = source.bytes.clone() else {
            self.fail(format!("no_bytes:{}", source.file));
            return None;
        };
        let sound = self.decode(Arc::clone(&bytes), source.stream);
        self.files.insert(
            key,
            File {
                bytes,
                sound: sound.clone(),
            },
        );
        Some(sound)
    }

    fn add(&mut self, voice: Voice, limit: u32) {
        let failed = voice
            .sound()
            .is_some_and(|sound| matches!(self.sounds.get(&sound), Some(Request::Failed(_))));
        if failed {
            return;
        }
        if self.live() >= limit as usize {
            self.cut_one();
        } else if self.device() {
            self.voices.push(voice);
        }
    }

    fn begin(&mut self, entity: Entity, source: &SoundSource, limit: u32) {
        let Some(sound) = self.sound_of(source) else {
            return;
        };
        let owner = Owner::Source(entity, Key::of(source));
        let voice = Voice::new(owner, sound, Start::of_source(source), None);
        self.add(voice, limit);
    }

    fn halt(&mut self, at: usize) {
        let voice = self.voices.swap_remove(at);
        self.backend.stop(voice.id);
    }

    fn adjust(&mut self, at: usize, volume: f32, pitch: f32) {
        let voice = &mut self.voices[at];
        voice.play.volume = volume;
        voice.play.pitch = pitch;
        if voice.started() {
            self.backend.set_voice(voice.id, volume, pitch);
        }
    }

    fn reconcile(&mut self, entity: Entity, source: &SoundSource, limit: u32) {
        let at = self.voices.iter().position(|voice| voice.of_entity(entity));
        if !source.autoplay {
            if let Some(at) = at {
                self.halt(at);
            }
            return;
        }
        let key = Key::of(source);
        match at {
            Some(at) if self.voices[at].holds(&key) => {
                if self.voices[at].live() {
                    self.adjust(at, level(source.volume), rate(source.pitch));
                }
            }
            Some(at) => {
                self.halt(at);
                self.begin(entity, source, limit);
            }
            None => self.begin(entity, source, limit),
        }
    }

    fn order(&mut self, order: SoundOrder, limit: u32) {
        match order {
            SoundOrder::Load { sound, bytes } => {
                let decoded = self.decode(bytes, false);
                self.loaded.insert(sound, decoded);
            }
            SoundOrder::Play { voice, sound, shot } => match self.loaded.get(&sound).cloned() {
                Some(held) => self.add(Voice::shot(voice, held, &shot), limit),
                None => self.cut_one(),
            },
            SoundOrder::PlayBytes { voice, bytes, shot } => {
                if self.device != Device::Off {
                    let sound = self.decode(bytes, false);
                    self.add(Voice::shot(voice, sound, &shot), limit);
                }
            }
            SoundOrder::Stop(voice) => {
                if let Some(at) = self.voices.iter().position(|held| held.of_handle(voice)) {
                    self.halt(at);
                }
            }
            SoundOrder::Volume(voice, volume) => {
                if let Some(at) = self.voices.iter().position(|held| held.of_handle(voice)) {
                    let pitch = self.voices[at].play.pitch;
                    self.adjust(at, level(volume), pitch);
                }
            }
        }
    }

    fn trim(&mut self, limit: u32) {
        while self.live() > limit as usize {
            let Some(at) = self.voices.iter().rposition(Voice::live) else {
                return;
            };
            self.halt(at);
            self.cut_one();
        }
    }

    fn collect_decoded(&mut self) {
        for (sound, decoded) in self.backend.arrivals() {
            self.arrive(sound, decoded);
        }
    }

    fn apply_buses(&mut self, settings: &Audio, paused: bool) {
        if !self.backend.opened() {
            return;
        }
        for bus in Bus::ALL {
            self.backend
                .set_bus(bus, settings.volume_of(bus), paused && pauses(bus));
        }
    }

    fn running(&mut self, at: usize) -> bool {
        let Self {
            backend,
            sounds,
            voices,
            cut,
            latest,
            ..
        } = self;
        let voice = &mut voices[at];
        let sound = match &voice.playing {
            Playing::Started(_) => return !backend.finished(voice.id),
            Playing::Ended => return true,
            Playing::Waiting(sound) => *sound.key(),
        };
        match sounds.get(&sound) {
            Some(Request::Asked(_)) => true,
            Some(Request::Ready(decoded)) => match backend.start(voice.id, decoded, &voice.play) {
                Ok(()) => {
                    voice.playing = std::mem::replace(&mut voice.playing, Playing::Ended).started();
                    true
                }
                Err(failure) => {
                    *cut = cut.saturating_add(1);
                    *latest = Some(failure);
                    false
                }
            },
            Some(Request::Failed(_) | Request::Missing) | None => false,
        }
    }

    fn advance(&mut self) {
        if !self.voices.iter().any(Voice::live) {
            return;
        }
        self.backend.tick();
        let mut at = 0;
        while at < self.voices.len() {
            if !self.voices[at].live() || self.running(at) {
                at += 1;
                continue;
            }
            let voice = &mut self.voices[at];
            self.backend.stop(voice.id);
            if voice.ends() {
                voice.playing = Playing::Ended;
                at += 1;
            } else {
                self.voices.swap_remove(at);
            }
        }
    }

    fn hear(&mut self, listener: Option<Mat4>, globals: &GlobalTransforms) {
        let Some(listener) = listener.filter(|_| self.backend.opened()) else {
            return;
        };
        let eye = listener.w_axis.truncate();
        let forward = (-listener.z_axis.truncate())
            .try_normalize()
            .unwrap_or(Vec3::NEG_Z);
        let up = listener.y_axis.truncate().try_normalize().unwrap_or(Vec3::Y);
        self.backend.listen(eye, forward, up);
        for voice in &mut self.voices {
            if !voice.started() || voice.play.radius.is_none() {
                continue;
            }
            let posed = match &voice.owner {
                Owner::Source(entity, _) => globals.pose(*entity),
                Owner::Handle(_) => None,
            };
            if let Some(pose) = posed {
                voice.at = Some(Vec3::from(pose.translation));
            }
            self.backend
                .place(voice.id, voice.at.filter(|at| at.is_finite()), eye);
        }
    }

    fn publish(&self, stats: &mut AudioStats) {
        let census = self.sounds.census();
        stats.device = self.backend.opened();
        stats.sounds_loaded = census.ready;
        stats.voices = self.live() as u32;
        stats.voices_cut = self.cut;
        stats.decode_failures = self.failures;
        stats.latest_failure.clone_from(&self.latest);
    }
}

pub(crate) fn mix(
    (settings, time): (Res<Audio>, Res<FrameTime>),
    (views, poses, globals, pause): (
        Res<Views>,
        Poses,
        Res<GlobalTransforms>,
        Option<Res<Pause>>,
    ),
    sounds: Res<Sounds>,
    sources: Query<(Entity, Ref<SoundSource>)>,
    (mut stats, mut mixer, mut spawner): (ResMut<AudioStats>, ResMut<Mixer>, Spawner<AudioBrick>),
) {
    let was = mixer.device;
    let closed = !mixer.backend.opened();
    let limit = settings.max_voices;
    mixer.set_enabled(settings.enabled);
    if !mixer.files.is_empty() {
        let named: BTreeSet<(&str, bool)> = sources
            .iter()
            .map(|(_, source)| {
                let source = source.into_inner();
                (source.file.as_str(), source.stream)
            })
            .collect();
        mixer.prune(&named);
    }
    mixer.cut = mixer.cut.saturating_add(sounds.take_cut());
    for order in sounds.take_orders() {
        mixer.order(order, limit);
    }
    for (entity, source) in &sources {
        if source.is_changed() {
            mixer.reconcile(entity, &source, limit);
        } else if source.looping
            && source.autoplay
            && source.bytes.is_some()
            && mixer.room(limit)
            && !mixer.voices.iter().any(|voice| voice.of_entity(entity))
        {
            mixer.begin(entity, &source, limit);
        }
    }
    for at in (0..mixer.voices.len()).rev() {
        if let Owner::Source(entity, _) = &mixer.voices[at].owner
            && !sources.contains(*entity)
        {
            mixer.halt(at);
        }
    }
    mixer.trim(limit);
    mixer.collect_decoded();
    let pause_moved = pause.as_ref().is_some_and(|pause| pause.is_changed());
    if settings.is_changed() || pause_moved || (closed && mixer.backend.opened()) {
        mixer.apply_buses(&settings, pause.is_some_and(|pause| pause.paused()));
    }
    mixer.advance();
    mixer.hear(views.main().and_then(|main| poses.drawn_pose(main.camera)), &globals);
    mixer.sounds.settle(time.real_elapsed);
    mixer.publish(&mut stats);
    if was != Device::Failed && mixer.device == Device::Failed {
        let error = Error::new(AUDIO_FAILED, "$.audio.device").got(mixer.latest.clone());
        spawner.fault(error);
    }
}

pub(crate) fn release(In(entity): In<Entity>, mut mixer: ResMut<Mixer>) {
    if let Some(at) = mixer.voices.iter().position(|held| held.of_entity(entity)) {
        mixer.halt(at);
    }
}
