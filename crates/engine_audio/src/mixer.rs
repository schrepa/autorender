use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use bevy_ecs::prelude::*;
use engine_core::math::clamp;
use engine_core::{Camera, Poses, SceneIds, TimeControl};
use engine_places::audio::{
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
}

struct Voice {
    id: u32,
    owner: Owner,
    sound: u32,
    play: Start,
    at: Option<Vec3>,
    started: bool,
    once: bool,
}

impl Voice {
    fn new(owner: Owner, sound: u32, play: Start, at: Option<Vec3>, once: bool) -> Self {
        Self {
            id: fresh(),
            owner,
            sound,
            play,
            at,
            started: false,
            once,
        }
    }

    fn shot(handle: VoiceHandle, sound: u32, shot: &OneShot, once: bool) -> Self {
        Self::new(
            Owner::Shot(handle),
            sound,
            Start::of_shot(shot),
            shot.at,
            once,
        )
    }

    fn of_entity(&self, entity: Entity) -> bool {
        matches!(&self.owner, Owner::Source(owner, _) if *owner == entity)
    }

    fn of_handle(&self, handle: VoiceHandle) -> bool {
        matches!(&self.owner, Owner::Shot(owner) if *owner == handle)
    }

    fn holds(&self, key: &Key) -> bool {
        matches!(&self.owner, Owner::Source(_, held) if held == key)
    }
}

struct Held {
    id: u32,
    bytes: Arc<[u8]>,
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
    voices: Vec<Voice>,
    stages: BTreeMap<u32, Stage>,
    handles: BTreeMap<SoundHandle, u32>,
    files: BTreeMap<(String, bool), Held>,
    ended: BTreeMap<Entity, Key>,
    applied: [Option<(f32, bool)>; Bus::ALL.len()],
    disabled: bool,
    refused: bool,
    open_failed: bool,
    dirty: bool,
    cut: u32,
    failures: u32,
    latest: Option<String>,
}

impl Drop for Mixer {
    fn drop(&mut self) {
        for voice in &self.voices {
            self.backend.stop(voice.id);
        }
        for sound in self.stages.keys() {
            self.backend.forget(*sound);
        }
    }
}

impl Mixer {
    fn is_idle(&self) -> bool {
        !self.dirty
            && self.voices.is_empty()
            && !self.stages.values().any(|stage| *stage == Stage::Decoding)
    }

    pub(crate) fn set_available(&mut self, available: bool) {
        self.refused = !available;
        self.open_failed = false;
        self.dirty = true;
        if !available {
            self.shut();
        }
    }

    pub(crate) fn listed<'a>(&self, ids: &'a SceneIds<'_, '_>) -> Vec<VoiceView<'a>> {
        self.voices
            .iter()
            .map(|voice| VoiceView {
                entity: match &voice.owner {
                    Owner::Source(entity, _) => ids.id(*entity),
                    Owner::Shot(_) => None,
                },
                bus: voice.play.bus,
                volume: voice.play.volume,
                pitch: voice.play.pitch,
                looping: voice.play.looping,
                spatial: voice.play.radius.is_some(),
                playing: voice.started,
                at: voice.at.map(|at| at.to_array()),
            })
            .collect()
    }

    fn device(&mut self) -> bool {
        if self.backend.opened() {
            return true;
        }
        if self.disabled || self.refused || self.open_failed {
            return false;
        }
        match self.backend.open() {
            Ok(()) => {
                self.applied = [None; Bus::ALL.len()];
                true
            }
            Err(failure) => {
                self.open_failed = true;
                self.latest = Some(failure);
                false
            }
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

    fn decode(&mut self, id: u32, bytes: Arc<[u8]>, stream: bool) {
        self.stages.insert(id, Stage::Decoding);
        let may_open = !self.refused && !self.open_failed;
        self.backend.decode(id, bytes, stream, may_open);
    }

    fn forget(&mut self, id: u32) {
        self.backend.forget(id);
        self.stages.remove(&id);
    }

    fn discard(&mut self, voice: &Voice) {
        if voice.once {
            self.forget(voice.sound);
        }
    }

    fn prune(&mut self) {
        let (backend, stages) = (&mut self.backend, &mut self.stages);
        self.files.retain(|_, held| {
            let kept = Arc::strong_count(&held.bytes) > 1;
            if !kept {
                backend.forget(held.id);
                stages.remove(&held.id);
            }
            kept
        });
    }

    fn sound_of(&mut self, source: &SoundSource) -> Option<u32> {
        let key = (source.file.clone(), source.stream);
        let current = self
            .files
            .get(&key)
            .filter(|held| {
                source
                    .bytes
                    .as_ref()
                    .is_none_or(|bytes| Arc::ptr_eq(bytes, &held.bytes))
            })
            .map(|held| held.id);
        if current.is_some() {
            return current;
        }
        let Some(bytes) = source.bytes.clone() else {
            self.fail(format!("no_bytes:{}", source.file));
            return None;
        };
        let id = fresh();
        self.decode(id, Arc::clone(&bytes), source.stream);
        if let Some(replaced) = self.files.insert(key, Held { id, bytes }) {
            self.forget(replaced.id);
        }
        Some(id)
    }

    fn add(&mut self, voice: Voice, limit: u32) {
        if self.stages.get(&voice.sound) == Some(&Stage::Failed) {
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
        let voice = Voice::new(owner, sound, Start::of_source(source), None, false);
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
        if voice.started {
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
                let id = fresh();
                self.decode(id, bytes, false);
                self.handles.insert(sound, id);
            }
            SoundOrder::Play { voice, sound, shot } => match self.handles.get(&sound).copied() {
                Some(id) => self.add(Voice::shot(voice, id, &shot, false), limit),
                None => self.cut_one(),
            },
            SoundOrder::PlayBytes { voice, bytes, shot } => {
                if !self.disabled {
                    let id = fresh();
                    self.decode(id, bytes, false);
                    self.add(Voice::shot(voice, id, &shot, true), limit);
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
        for (id, stage) in &mut self.stages {
            if *stage != Stage::Decoding {
                continue;
            }
            match self.backend.arrival(*id) {
                None => {}
                Some(Ok(())) => *stage = Stage::Ready,
                Some(Err(failure)) => {
                    *stage = Stage::Failed;
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
            let state = (settings.volume_of(bus), paused && pauses(bus));
            if self.applied[bus.index()] != Some(state) {
                self.applied[bus.index()] = Some(state);
                self.backend.set_bus(bus, state.0, state.1);
            }
        }
    }

    fn running(&mut self, at: usize) -> bool {
        let voice = &mut self.voices[at];
        if voice.started {
            return !self.backend.finished(voice.id);
        }
        match self.stages.get(&voice.sound) {
            Some(Stage::Decoding) => true,
            Some(Stage::Ready) => match self.backend.start(voice.id, voice.sound, &voice.play) {
                Ok(()) => {
                    voice.started = true;
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
        self.dirty = true;
        match voice.owner {
            Owner::Source(entity, key) if !voice.play.looping => {
                self.ended.insert(entity, key);
            }
            Owner::Source(..) | Owner::Shot(_) => {}
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

    fn hear(&mut self, camera: &Camera, poses: &Poses) {
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
            if !voice.started || voice.play.radius.is_none() {
                continue;
            }
            let posed = match &voice.owner {
                Owner::Source(entity, _) => poses.of(*entity),
                Owner::Shot(_) => None,
            };
            if let Some(pose) = posed {
                voice.at = Some(Vec3::from(pose.translation));
            }
            self.backend
                .place(voice.id, voice.at.filter(|at| at.is_finite()), eye);
        }
    }

    fn count(&self, stage: Stage) -> u32 {
        self.stages.values().filter(|held| **held == stage).count() as u32
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
    (camera, poses, time): (Res<Camera>, Res<Poses>, Option<Res<TimeControl>>),
    mut sounds: ResMut<Sounds>,
    sources: Query<(Entity, Ref<SoundSource>)>,
    mut stats: ResMut<AudioStats>,
    mut mixer: ResMut<Mixer>,
) {
    if sources.is_empty() && sounds.is_idle() && mixer.is_idle() && !settings.is_changed() {
        return;
    }
    let limit = settings.max_voices;
    mixer.disabled = !settings.enabled;
    if mixer.disabled {
        mixer.shut();
    }
    let dirty = mixer.dirty || settings.is_changed();
    mixer.dirty = false;
    if dirty {
        mixer.prune();
    }
    mixer.cut = mixer.cut.saturating_add(sounds.take_cut());
    for order in sounds.take_orders() {
        mixer.order(order, limit);
    }
    for (entity, source) in &sources {
        if source.is_changed() {
            mixer.reconcile(entity, &source, limit);
        } else if dirty
            && source.looping
            && source.autoplay
            && !mixer.voices.iter().any(|voice| voice.of_entity(entity))
        {
            mixer.begin(entity, &source, limit);
        }
    }
    mixer.trim(limit);
    mixer.collect_decoded();
    mixer.apply_buses(&settings, time.is_some_and(|time| time.paused));
    mixer.advance();
    mixer.hear(&camera, &poses);
    mixer.publish(&mut stats);
}

pub(crate) fn release(In(entity): In<Entity>, mut mixer: ResMut<Mixer>) {
    mixer.ended.remove(&entity);
    mixer.dirty = true;
    if let Some(at) = mixer.voices.iter().position(|held| held.of_entity(entity)) {
        mixer.halt(at);
    }
}
