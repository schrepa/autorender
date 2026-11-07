use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use bevy_ecs::prelude::*;
use engine_core::math::clamp;
use engine_core::{Camera, GlobalTransforms, SceneIds, TimeControl};
use engine_places_rendering::audio::{
    Bus, OneShot, SoundHandle, SoundOrder, SoundSource, Sounds, VoiceHandle,
};
use glam::Vec3;
use serde::Serialize;

use crate::backend::Backend;
use crate::scene::{MAX_PITCH, MAX_RADIUS, MIN_PITCH, MIN_RADIUS};
use crate::settings::{Audio, AudioStats};

const STRAIGHT_UP: f32 = 0.999;

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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Decoding,
    Ready,
    Failed,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Device {
    #[default]
    Allowed,
    Off,
    Refused,
    Failed,
}

enum Origin {
    File((String, bool), Arc<[u8]>),
    Loaded(SoundHandle),
    Once,
}

struct Sound {
    stage: Stage,
    origin: Origin,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Playing {
    Waiting,
    Started,
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
    Shot(VoiceHandle),
    Bytes(VoiceHandle),
}

struct Voice {
    id: u32,
    owner: Owner,
    sound: u32,
    play: Start,
    at: Option<Vec3>,
    playing: Playing,
}

impl Voice {
    fn new(owner: Owner, sound: u32, play: Start, at: Option<Vec3>) -> Self {
        Self {
            id: fresh(),
            owner,
            sound,
            play,
            at,
            playing: Playing::Waiting,
        }
    }

    fn shot(owner: Owner, sound: u32, shot: &OneShot) -> Self {
        Self::new(owner, sound, Start::of_shot(shot), shot.at)
    }

    fn started(&self) -> bool {
        self.playing == Playing::Started
    }

    fn of_entity(&self, entity: Entity) -> bool {
        matches!(&self.owner, Owner::Source(owner, _) if *owner == entity)
    }

    fn of_handle(&self, handle: VoiceHandle) -> bool {
        matches!(&self.owner, Owner::Shot(owner) | Owner::Bytes(owner) if *owner == handle)
    }

    fn holds(&self, key: &Key) -> bool {
        matches!(&self.owner, Owner::Source(_, held) if held == key)
    }
}

#[derive(Serialize)]
pub(crate) struct VoiceView<'a> {
    entity: Option<&'a str>,
    bus: Bus,
    volume: f32,
    pitch: f32,
    looping: bool,
    spatial: bool,
    playing: bool,
    at: Option<[f32; 3]>,
}

#[derive(Resource, Default)]
pub(crate) struct Mixer {
    backend: Backend,
    device: Device,
    voices: Vec<Voice>,
    sounds: BTreeMap<u32, Sound>,
    ended: BTreeMap<Entity, Key>,
    cut: u32,
    failures: u32,
    latest: Option<String>,
}

impl Drop for Mixer {
    fn drop(&mut self) {
        for voice in &self.voices {
            self.backend.stop(voice.id);
        }
        for sound in self.sounds.keys() {
            self.backend.forget(*sound);
        }
    }
}

impl Mixer {
    fn room(&self, limit: u32) -> bool {
        self.device == Device::Allowed && self.voices.len() < limit as usize
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

    pub(crate) fn listed<'a>(&self, ids: &'a SceneIds<'_, '_>) -> Vec<VoiceView<'a>> {
        self.voices
            .iter()
            .map(|voice| VoiceView {
                entity: match &voice.owner {
                    Owner::Source(entity, _) => ids.id(*entity),
                    Owner::Shot(_) | Owner::Bytes(_) => None,
                },
                bus: voice.play.bus,
                volume: voice.play.volume,
                pitch: voice.play.pitch,
                looping: voice.play.looping,
                spatial: voice.play.radius.is_some(),
                playing: voice.started(),
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
        for voice in std::mem::take(&mut self.voices) {
            self.backend.stop(voice.id);
            self.discard(&voice);
        }
        self.backend.close();
    }

    fn fail(&mut self, failure: String) {
        self.failures = self.failures.saturating_add(1);
        self.latest = Some(failure);
    }

    fn cut_one(&mut self) {
        self.cut = self.cut.saturating_add(1);
    }

    fn decode(&mut self, id: u32, bytes: Arc<[u8]>, (stream, origin): (bool, Origin)) {
        self.sounds.insert(
            id,
            Sound {
                stage: Stage::Decoding,
                origin,
            },
        );
        let may_open = !matches!(self.device, Device::Refused | Device::Failed);
        self.backend.decode(id, bytes, stream, may_open);
    }

    fn forget(&mut self, id: u32) {
        self.backend.forget(id);
        self.sounds.remove(&id);
    }

    fn stage(&self, id: u32) -> Option<Stage> {
        self.sounds.get(&id).map(|sound| sound.stage)
    }

    fn discard(&mut self, voice: &Voice) {
        match voice.owner {
            Owner::Bytes(_) => self.forget(voice.sound),
            Owner::Source(..) | Owner::Shot(_) => {}
        }
    }

    fn files(&self, kept: impl Fn(&(String, bool), &Arc<[u8]>) -> bool) -> Vec<u32> {
        self.sounds
            .iter()
            .filter_map(|(&id, sound)| match &sound.origin {
                Origin::File(key, bytes) => kept(key, bytes).then_some(id),
                Origin::Loaded(_) | Origin::Once => None,
            })
            .collect()
    }

    fn prune(&mut self) {
        for id in self.files(|_, bytes| Arc::strong_count(bytes) == 1) {
            self.forget(id);
        }
    }

    fn loaded(&self, handle: SoundHandle) -> Option<u32> {
        self.sounds.iter().find_map(|(&id, sound)| match sound.origin {
            Origin::Loaded(held) => (held == handle).then_some(id),
            Origin::File(..) | Origin::Once => None,
        })
    }

    fn sound_of(&mut self, source: &SoundSource) -> Option<u32> {
        let key = (source.file.clone(), source.stream);
        let current = self
            .files(|held, bytes| {
                *held == key
                    && source
                        .bytes
                        .as_ref()
                        .is_none_or(|wanted| Arc::ptr_eq(wanted, bytes))
            })
            .first()
            .copied();
        if current.is_some() {
            return current;
        }
        let Some(bytes) = source.bytes.clone() else {
            self.fail(format!("no_bytes:{}", source.file));
            return None;
        };
        let replaced = self.files(|held, _| *held == key);
        let id = fresh();
        let origin = Origin::File(key, Arc::clone(&bytes));
        self.decode(id, bytes, (source.stream, origin));
        for old in replaced {
            self.forget(old);
        }
        Some(id)
    }

    fn add(&mut self, voice: Voice, limit: u32) {
        if self.stage(voice.sound) == Some(Stage::Failed) {
            self.discard(&voice);
        } else if self.voices.len() >= limit as usize {
            self.cut_one();
            self.discard(&voice);
        } else if self.device() {
            self.voices.push(voice);
        } else {
            self.discard(&voice);
        }
    }

    fn begin(&mut self, entity: Entity, source: &SoundSource, limit: u32) {
        self.ended.remove(&entity);
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
        self.discard(&voice);
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
            self.ended.remove(&entity);
            if let Some(at) = at {
                self.halt(at);
            }
            return;
        }
        let key = Key::of(source);
        match at {
            Some(at) if self.voices[at].holds(&key) => {
                self.adjust(at, level(source.volume), rate(source.pitch));
            }
            Some(at) => {
                self.halt(at);
                self.begin(entity, source, limit);
            }
            None if self.ended.get(&entity) == Some(&key) => {}
            None => self.begin(entity, source, limit),
        }
    }

    fn order(&mut self, order: SoundOrder, limit: u32) {
        match order {
            SoundOrder::Load { sound, bytes } => {
                self.decode(fresh(), bytes, (false, Origin::Loaded(sound)));
            }
            SoundOrder::Play { voice, sound, shot } => match self.loaded(sound) {
                Some(id) => self.add(Voice::shot(Owner::Shot(voice), id, &shot), limit),
                None => self.cut_one(),
            },
            SoundOrder::PlayBytes { voice, bytes, shot } => {
                if self.device != Device::Off {
                    let id = fresh();
                    self.decode(id, bytes, (false, Origin::Once));
                    self.add(Voice::shot(Owner::Bytes(voice), id, &shot), limit);
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
        while self.voices.len() > limit as usize {
            self.halt(self.voices.len() - 1);
            self.cut_one();
        }
    }

    fn collect_decoded(&mut self) {
        for (id, sound) in &mut self.sounds {
            if sound.stage != Stage::Decoding {
                continue;
            }
            match self.backend.arrival(*id) {
                None => {}
                Some(Ok(())) => sound.stage = Stage::Ready,
                Some(Err(failure)) => {
                    sound.stage = Stage::Failed;
                    self.backend.forget(*id);
                    self.failures = self.failures.saturating_add(1);
                    self.latest = Some(failure);
                }
            }
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
        let stage = self.stage(self.voices[at].sound);
        let voice = &mut self.voices[at];
        if voice.started() {
            return !self.backend.finished(voice.id);
        }
        match stage {
            Some(Stage::Decoding) => true,
            Some(Stage::Ready) => match self.backend.start(voice.id, voice.sound, &voice.play) {
                Ok(()) => {
                    voice.playing = Playing::Started;
                    true
                }
                Err(failure) => {
                    self.cut = self.cut.saturating_add(1);
                    self.latest = Some(failure);
                    false
                }
            },
            Some(Stage::Failed) | None => false,
        }
    }

    fn retire(&mut self, voice: Voice) {
        self.backend.stop(voice.id);
        self.discard(&voice);
        match voice.owner {
            Owner::Source(entity, key) if !voice.play.looping => {
                self.ended.insert(entity, key);
            }
            Owner::Source(..) | Owner::Shot(_) | Owner::Bytes(_) => {}
        }
    }

    fn advance(&mut self) {
        if self.voices.is_empty() {
            return;
        }
        self.backend.tick();
        let mut at = 0;
        while at < self.voices.len() {
            if self.running(at) {
                at += 1;
            } else {
                let voice = self.voices.swap_remove(at);
                self.retire(voice);
            }
        }
    }

    fn hear(&mut self, camera: &Camera, globals: &GlobalTransforms) {
        if !self.backend.opened() {
            return;
        }
        let eye = camera.eye();
        let forward = (camera.target - eye).try_normalize().unwrap_or(Vec3::NEG_Z);
        let up = if forward.dot(Vec3::Y).abs() > STRAIGHT_UP {
            Vec3::Z
        } else {
            Vec3::Y
        };
        self.backend.listen(eye, forward, up);
        for voice in &mut self.voices {
            if !voice.started() || voice.play.radius.is_none() {
                continue;
            }
            let posed = match &voice.owner {
                Owner::Source(entity, _) => globals.pose(*entity),
                Owner::Shot(_) | Owner::Bytes(_) => None,
            };
            if let Some(pose) = posed {
                voice.at = Some(Vec3::from(pose.translation));
            }
            self.backend
                .place(voice.id, voice.at.filter(|at| at.is_finite()), eye);
        }
    }

    fn count(&self, stage: Stage) -> u32 {
        self.sounds
            .values()
            .filter(|sound| sound.stage == stage)
            .count() as u32
    }

    fn publish(&self, stats: &mut AudioStats) {
        stats.device = self.backend.opened();
        stats.sounds_loaded = self.count(Stage::Ready);
        stats.decoding = self.count(Stage::Decoding);
        stats.voices = self.voices.len() as u32;
        stats.voices_cut = self.cut;
        stats.decode_failures = self.failures;
        stats.latest_failure.clone_from(&self.latest);
    }
}

pub(crate) fn mix(
    settings: Res<Audio>,
    (camera, globals, time): (Res<Camera>, Res<GlobalTransforms>, Option<Res<TimeControl>>),
    sounds: Res<Sounds>,
    sources: Query<(Entity, Ref<SoundSource>)>,
    mut stats: ResMut<AudioStats>,
    mut mixer: ResMut<Mixer>,
) {
    let closed = !mixer.backend.opened();
    let limit = settings.max_voices;
    mixer.set_enabled(settings.enabled);
    mixer.prune();
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
    let time_moved = time.as_ref().is_some_and(|time| time.is_changed());
    if settings.is_changed() || time_moved || (closed && mixer.backend.opened()) {
        mixer.apply_buses(&settings, time.is_some_and(|time| time.paused));
    }
    mixer.advance();
    mixer.hear(&camera, &globals);
    mixer.publish(&mut stats);
}

pub(crate) fn release(In(entity): In<Entity>, mut mixer: ResMut<Mixer>) {
    mixer.ended.remove(&entity);
    if let Some(at) = mixer.voices.iter().position(|held| held.of_entity(entity)) {
        mixer.halt(at);
    }
}
