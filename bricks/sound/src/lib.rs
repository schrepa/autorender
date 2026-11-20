#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

engine::brick!();

mod spatial;
mod table;
mod voices;

use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::sync::Arc;

use engine::prelude::*;
use engine_core::{
    Asset, AssetKind, AssetKinds, Assets, Loaded, Output, Play, SoundId, Sounds, Stats, Tune,
};
use engine_math::Random;
use glam::Vec3;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use table::{Curve, Named, Table};
use voices::{Admit, Voice};

pub trait Occluder {
    fn clear(&self, from: Vec3, to: Vec3) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Anchor {
    Listener,
    Point(Vec3),
    Follow(u64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cue {
    pub event: String,
    pub anchor: Anchor,
    pub source: Option<u64>,
    pub gain: f32,
    pub hold_ticks: Option<u32>,
    pub tag: Option<u64>,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct SoundCues {
    pub tick: u64,
    pub cues: Vec<Cue>,
    pub stops: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ear {
    pub eye: Vec3,
    pub right: Vec3,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct Hearing {
    pub ear: Option<Ear>,
    pub own: Option<u64>,
    pub follows: BTreeMap<u64, Vec3>,
    pub master: Option<f32>,
    pub groups: BTreeMap<String, f32>,
    pub reach: Option<BTreeMap<String, f32>>,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct SoundLoad {
    pub ready: bool,
    pub asked: u32,
    pub loaded: u32,
    pub failed: u32,
    pub refusal: Option<String>,
    pub events: Arc<BTreeSet<String>>,
}

#[derive(Component, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SoundTable {
    table: Asset<Table>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DoorStats {
    pub output: String,
    pub voices: u32,
    pub ready: u32,
    pub failed: u32,
    pub bytes_decoded: u64,
    pub refused: u64,
    pub dropped: u64,
    pub underruns: Option<u64>,
    pub latency_seconds: Option<f32>,
}

stats! {
    #[section("sound")]
    pub struct SoundStats {
        pub table: String,
        pub cues_tick: u32,
        pub cues: u64,
        pub voices: u32,
        pub started: u64,
        pub stolen: u64,
        pub dropped: u64,
        pub inaudible: u64,
        pub unmapped: u64,
        pub refused: u64,
        pub asked: u32,
        pub loaded: u32,
        pub ready: bool,
        pub waiting: Vec<String>,
        pub unset_gains: Vec<String>,
        pub heard: BTreeMap<String, u64>,
        pub door: DoorStats,
    }
}

#[derive(Resource, Default)]
struct Bank {
    table: Option<Arc<Table>>,
    refusal: Option<String>,
    files: BTreeMap<String, SoundId>,
    opened: bool,
    master: Option<f32>,
    reached: Option<BTreeMap<String, f32>>,
    events: Arc<BTreeSet<String>>,
}

impl Bank {
    fn playable(&self) -> Option<&Table> {
        let table = self.table.as_deref()?;
        let far = self.reached.is_some()
            || table.curves.iter().all(|(_, curve)| curve.far_from.is_none());
        (far && self.refusal.is_none()).then_some(table)
    }
}

#[derive(Resource, Default)]
struct Mixdown {
    voices: Vec<Voice>,
    rng: Option<Random>,
    last: Vec<Option<usize>>,
    started_at: Vec<Option<u64>>,
    gains: Vec<f32>,
}

pub struct SoundBrick<O: Occluder + Resource>(PhantomData<fn() -> O>);

impl<O: Occluder + Resource> Brick for SoundBrick<O> {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<SoundTable>("sound")
                    .schema(engine_core::schema_of::<SoundTable>),
            )
            .join::<AssetKinds>(AssetKind::of("sound_table", Table::read))
            .resource::<SoundLoad>()
            .resource::<SoundCues>()
            .resource::<Hearing>()
            .resource::<Bank>()
            .resource::<Mixdown>()
            .systems(Schedule::FixedSimulation, mix::<O>)
            .systems(Schedule::Presentation, prepare)
            .systems(Schedule::Presentation, flush_tunes)
            .systems(Schedule::Presentation, load)
            .dev(|dev| {
                dev.stats::<SoundStats>()
                    .systems(Schedule::Presentation, report);
            });
    }
}

fn prepare(
    (tables, assets): (Query<Ref<SoundTable>>, Res<Assets>),
    hearing: Res<Hearing>,
    (mut bank, mut mixdown): (ResMut<Bank>, ResMut<Mixdown>),
    mut sounds: ResMut<Sounds>,
) {
    for table in &tables {
        if table.is_changed()
            && let Ok(read) = assets.get(table.table)
        {
            adopt(&mut bank, read.clone(), &mut mixdown, &mut sounds);
        }
    }
    reach(&mut bank, &hearing);
    let Some(table) = bank.table.as_ref() else {
        return;
    };
    let master = table.master_gain * hearing.master.unwrap_or(1.0);
    let glide = table.glide_seconds;
    if bank.master != Some(master) {
        sounds.master(master, glide);
        bank.master = Some(master);
    }
}

fn reach(bank: &mut Bank, hearing: &Hearing) {
    let (Some(table), Some(reach)) = (bank.table.as_mut(), hearing.reach.as_ref()) else {
        return;
    };
    if bank.reached.as_ref() == Some(reach) {
        return;
    }
    bank.reached = Some(reach.clone());
    bank.refusal = Arc::make_mut(table).reach(reach).err();
}

fn adopt(bank: &mut Bank, mut table: Table, mixdown: &mut Mixdown, sounds: &mut Sounds) {
    if !bank.opened {
        sounds.open(table.latency_seconds);
        bank.opened = true;
    }
    bank.master = None;
    bank.events = Arc::new(table.events.keys().cloned().collect());
    for (_, sound) in &mut table.sounds {
        for file in &mut sound.files {
            let id = *(bank.files)
                .entry(file.file.clone())
                .or_insert_with(|| sounds.load(&file.file));
            file.sound = Some(id);
        }
    }
    for mut voice in mixdown.voices.drain(..) {
        release(&mut voice, &table, table.steal_fade_seconds, sounds);
    }
    mixdown.last = vec![None; table.sounds.len()];
    mixdown.started_at = vec![None; table.sounds.len()];
    bank.table = Some(Arc::new(table));
    bank.refusal = None;
    bank.reached = None;
}

fn mix<O: Occluder + Resource>(
    time: Res<FixedTime>,
    bank: Res<Bank>,
    occluder: Option<Res<O>>,
    (fed, hearing): (Res<SoundCues>, Res<Hearing>),
    (mut mixdown, mut stats): (ResMut<Mixdown>, Stats<SoundStats>),
    mut sounds: ResMut<Sounds>,
) {
    let Some(table) = bank.playable() else {
        return;
    };
    let mixdown = &mut *mixdown;
    let now = time.tick;
    let (fresh, stops): (&[Cue], &[u64]) = if fed.tick == now {
        (&fed.cues, &fed.stops)
    } else {
        (&[], &[])
    };
    stop_tagged(&mut mixdown.voices, stops, now);
    stats.edit(|stats| {
        for cue in fresh {
            *stats.heard.entry(cue.event.clone()).or_default() += 1;
        }
        stats.cues_tick = u32::try_from(fresh.len()).unwrap_or(u32::MAX);
        stats.cues += fresh.len() as u64;
    });
    let Some(ear) = hearing.ear else {
        return;
    };
    let mut place = Place {
        table,
        ear,
        hearing: &hearing,
        gains: std::mem::take(&mut mixdown.gains),
        occluder: occluder.as_deref().map(|occluder| occluder as &dyn Occluder),
        now,
        tps: time.rate,
    };
    place.gains.clear();
    place.gains.extend(
        (table.groups.iter()).map(|group| hearing.groups.get(group).copied().unwrap_or(1.0)),
    );
    follow(mixdown, &place, &mut sounds);
    let rng = mixdown.rng.get_or_insert_with(|| Random::new(table.seed));
    let mut rng = *rng;
    for cue in fresh {
        start(mixdown, &place, &mut rng, cue, &mut sounds, &mut stats);
    }
    mixdown.rng = Some(rng);
    mixdown.gains = place.gains;
}

fn stop_tagged(voices: &mut [Voice], stops: &[u64], now: u64) {
    for voice in voices {
        if voice.tag.is_some_and(|tag| stops.contains(&tag)) {
            voice.ends = now;
        }
    }
}

struct Place<'a> {
    table: &'a Table,
    ear: Ear,
    hearing: &'a Hearing,
    gains: Vec<f32>,
    occluder: Option<&'a dyn Occluder>,
    now: u64,
    tps: u32,
}

impl Place<'_> {
    fn at(&self, anchor: Anchor, last: Vec3) -> Vec3 {
        match anchor {
            Anchor::Listener => self.ear.eye,
            Anchor::Point(point) => point,
            Anchor::Follow(id) => self.hearing.follows.get(&id).copied().unwrap_or(last),
        }
    }

    fn group(&self, group: &Named) -> f32 {
        self.gains[group.at]
    }

    fn own(&self, source: Option<u64>) -> bool {
        source.is_some() && source == self.hearing.own
    }

    fn occlusion(&self, curve: &Curve, at: Vec3) -> f32 {
        self.occluder.map_or(1.0, |occluder| {
            self.ear
                .occlusion(occluder, curve, at, self.table.occlusion_pullback_m)
        })
    }

    fn shape(&self, curve: &Curve, own: bool, anchor: Anchor, at: Vec3) -> (f32, f32) {
        if own || anchor == Anchor::Listener || curve.flat {
            return (1.0, 0.0);
        }
        (
            self.ear.attenuation(curve, at),
            self.ear.pan(at, self.table.pan_width),
        )
    }
}

fn follow(mixdown: &mut Mixdown, place: &Place<'_>, sounds: &mut Sounds) {
    let table = place.table;
    let refresh = place
        .now
        .is_multiple_of(u64::from(table.occlusion_every_ticks));
    mixdown.voices.retain_mut(|voice| {
        if voice.ends <= place.now {
            flush(voice, table.glide_seconds, sounds);
            if let (true, Some(door)) = (voice.looping, voice.door) {
                sounds.stop(door, table.hold_fade_seconds);
            }
            return false;
        }
        let (_, sound) = &table.sounds[voice.sound];
        let (_, curve) = &table.curves[sound.curve.at];
        let group = place.group(&sound.group);
        voice.at = place.at(voice.anchor, voice.at);
        let own = place.own(voice.source);
        if own {
            voice.occlusion = 1.0;
        } else if refresh {
            voice.occlusion = place.occlusion(curve, voice.at);
        }
        let (attenuation, pan) = place.shape(curve, own, voice.anchor, voice.at);
        voice.gain = voice.base * group * attenuation * voice.occlusion;
        voice.pan = pan;
        let moved = (voice.gain - voice.sent_gain).abs() > table.tune_step
            || (voice.pan - voice.sent_pan).abs() > table.tune_step;
        if moved && voice.door.is_some() {
            voice.sent_gain = voice.gain;
            voice.sent_pan = voice.pan;
            voice.owed = true;
        }
        true
    });
}

fn flush_tunes(bank: Res<Bank>, mut mixdown: ResMut<Mixdown>, mut sounds: ResMut<Sounds>) {
    let Some(table) = bank.table.as_deref() else {
        return;
    };
    for voice in &mut mixdown.voices {
        flush(voice, table.glide_seconds, &mut sounds);
    }
}

fn flush(voice: &mut Voice, glide_seconds: f32, sounds: &mut Sounds) {
    if std::mem::take(&mut voice.owed)
        && let Some(door) = voice.door
    {
        sounds.tune(
            door,
            Tune {
                gain: voice.sent_gain,
                pan: voice.sent_pan,
                rate: voice.rate,
                seconds: glide_seconds,
            },
        );
    }
}

fn release(voice: &mut Voice, table: &Table, fade_seconds: f32, sounds: &mut Sounds) {
    flush(voice, table.glide_seconds, sounds);
    if let Some(door) = voice.door {
        sounds.stop(door, fade_seconds);
    }
}

fn start(
    mixdown: &mut Mixdown,
    place: &Place<'_>,
    rng: &mut Random,
    cue: &Cue,
    sounds: &mut Sounds,
    stats: &mut Stats<'_, SoundStats>,
) {
    let table = place.table;
    if let (Some(tag), Some(hold)) = (cue.tag, cue.hold_ticks)
        && let Some(voice) = mixdown
            .voices
            .iter_mut()
            .find(|voice| voice.tag == Some(tag))
    {
        voice.ends = place.now + u64::from(hold);
        return;
    }
    let Some(named) = table.events.get(&cue.event) else {
        stats.edit(|stats| stats.unmapped += 1);
        return;
    };
    let (_, sound) = &table.sounds[named.at];
    let (_, curve) = &table.curves[sound.curve.at];
    let group = place.group(&sound.group);
    if clashed(mixdown, place, named.at, sounds) {
        stats.edit(|stats| stats.dropped += 1);
        return;
    }
    let pick = voices::variation(rng, sound.files.len(), mixdown.last[named.at]);
    mixdown.last[named.at] = Some(pick);
    let Some(file) = sound.files.get(pick) else {
        stats.edit(|stats| stats.unmapped += 1);
        return;
    };
    let own = place.own(cue.source);
    let at = place.at(cue.anchor, place.ear.eye);
    let occlusion = if own { 1.0 } else { place.occlusion(curve, at) };
    let (attenuation, pan) = place.shape(curve, own, cue.anchor, at);
    let level = 10.0_f32.powf(voices::signed(rng) * sound.gain_jitter_db / 20.0);
    let base = sound.gain * level * cue.gain * sound.own_gain.filter(|_| own).unwrap_or(1.0);
    let gain = base * group * attenuation * occlusion;
    if gain < table.audible_gain {
        stats.edit(|stats| stats.inaudible += 1);
        return;
    }
    match voices::admit(&mixdown.voices, table.voices, sound.priority, gain) {
        Admit::Free => {}
        Admit::Steal(index) => {
            let mut stolen = mixdown.voices.swap_remove(index);
            release(&mut stolen, table, table.steal_fade_seconds, sounds);
            stats.edit(|stats| stats.stolen += 1);
        }
        Admit::Drop => {
            stats.edit(|stats| stats.dropped += 1);
            return;
        }
    }
    let rate = 1.0 + voices::signed(rng) * sound.jitter;
    let looping = sound.loops && cue.hold_ticks.is_some();
    let ends = match cue.hold_ticks.filter(|_| looping) {
        Some(hold) => place.now + u64::from(hold),
        None => place.now + (f64::from(file.seconds / rate) * f64::from(place.tps)).ceil() as u64,
    };
    let door = file.sound.and_then(|id| {
        let played = sounds.play(Play {
            sound: id,
            gain,
            pan,
            rate,
            looping,
            offset_seconds: 0.0,
        });
        played.ok()
    });
    stats.edit(|stats| {
        stats.refused += u64::from(door.is_none());
        stats.started += 1;
    });
    mixdown.voices.push(Voice {
        door,
        sound: named.at,
        priority: sound.priority,
        base,
        gain,
        pan,
        rate,
        sent_gain: gain,
        sent_pan: pan,
        owed: false,
        anchor: cue.anchor,
        at,
        source: cue.source,
        tag: cue.tag,
        occlusion,
        looping,
        ends,
        started: place.now,
    });
    mixdown.started_at[named.at] = Some(place.now);
}

fn clashed(mixdown: &mut Mixdown, place: &Place<'_>, sound: usize, sounds: &mut Sounds) -> bool {
    let table = place.table;
    let span = |seconds: f32| (f64::from(seconds) * f64::from(place.tps)).round() as u64;
    let mut lost = false;
    for clash in &table.clashes {
        let window = span(clash.seconds);
        if clash.loser.at == sound {
            lost |= mixdown.started_at[clash.winner.at].is_some_and(|at| place.now < at + window);
        }
        if clash.winner.at == sound {
            mixdown.voices.retain_mut(|voice| {
                let beaten = voice.sound == clash.loser.at && place.now < voice.started + window;
                if beaten {
                    release(voice, table, table.steal_fade_seconds, sounds);
                }
                !beaten
            });
        }
    }
    lost
}

fn load(bank: Res<Bank>, sounds: Res<Sounds>, mut load: ResMut<SoundLoad>) {
    let asked = u32::try_from(bank.files.len()).unwrap_or(u32::MAX);
    let (mut loaded, mut failed) = (load.loaded, load.failed);
    if loaded + failed < asked {
        (loaded, failed) = (0, 0);
        for id in bank.files.values() {
            match sounds.loaded(*id) {
                Some(Loaded::Ready(_)) => loaded += 1,
                Some(Loaded::Failed(_)) => failed += 1,
                Some(Loaded::Reading | Loaded::Decoding) | None => {}
            }
        }
    }
    load.set_if_neq(SoundLoad {
        ready: bank.refusal.is_some() || (bank.playable().is_some() && loaded + failed == asked),
        asked,
        loaded,
        failed,
        refusal: bank.refusal.clone(),
        events: Arc::clone(&bank.events),
    });
}

fn report(
    (bank, hearing): (Res<Bank>, Res<Hearing>),
    mixdown: Res<Mixdown>,
    sounds: Res<Sounds>,
    load: Res<SoundLoad>,
    mut stats: Stats<SoundStats>,
) {
    let door = sounds.report();
    let waiting: Vec<String> = bank
        .files
        .iter()
        .filter_map(|(file, id)| match sounds.loaded(*id) {
            Some(Loaded::Reading) => Some(format!("{file} reading")),
            Some(Loaded::Decoding) => Some(format!("{file} decoding")),
            None => Some(format!("{file} unknown")),
            Some(Loaded::Ready(_) | Loaded::Failed(_)) => None,
        })
        .take(10)
        .collect();
    let table = match (&bank.refusal, bank.playable(), &bank.table) {
        (Some(refusal), ..) => refusal.clone(),
        (None, Some(_), _) => "ready".to_owned(),
        (None, None, Some(_)) => "reaching".to_owned(),
        (None, None, None) => "reading".to_owned(),
    };
    let mut unset_gains: BTreeSet<&str> = (bank.table.iter())
        .flat_map(|table| &table.groups)
        .map(String::as_str)
        .filter(|group| !hearing.groups.contains_key(*group))
        .collect();
    if hearing.master.is_none() {
        unset_gains.insert("master");
    }
    let latency_seconds = match sounds.output() {
        Output::Open {
            latency_seconds, ..
        } => Some(*latency_seconds),
        Output::Closed | Output::Asked | Output::Failed(_) | Output::Absent(_) => None,
    };
    stats.edit(|stats| {
        *stats = SoundStats {
            table,
            voices: u32::try_from(mixdown.voices.len()).unwrap_or(u32::MAX),
            asked: load.asked,
            loaded: load.loaded,
            ready: load.ready,
            waiting,
            unset_gains: unset_gains.into_iter().map(str::to_owned).collect(),
            door: DoorStats {
                output: door.output.to_owned(),
                voices: door.voices.live,
                ready: door.sounds.ready,
                failed: door.sounds.failed,
                bytes_decoded: door.bytes_decoded,
                refused: door.refused.total(),
                dropped: door.voices.dropped,
                underruns: door.underruns,
                latency_seconds,
            },
            ..std::mem::take(stats)
        };
    });
}
