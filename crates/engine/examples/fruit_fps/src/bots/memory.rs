use engine_math::clamp;
use glam::Vec3;

use super::aim::{PI, angle_between, look_angles, radians, sin_cos};
use super::numbers::{BotNumbers, LevelNumbers};
use super::rng::Rng;
use super::sense::{CallKind, Sense, SoundKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    Seen,
    Heard,
    Called,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Track {
    pub id: Option<u32>,
    pub at: Vec3,
    pub head: Vec3,
    pub chest: Vec3,
    pub velocity: Vec3,
    pub seen_tick: u64,
    pub visible: bool,
    pub visible_since: u64,
    pub react_tick: u64,
    pub source: Source,
    pub carries_device: bool,
    pub called_tick: Option<u64>,
    pub sound: Option<SoundKind>,
}

pub struct Look {
    pub eye: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Memory {
    pub tracks: Vec<Track>,
}

impl Memory {
    pub fn new() -> Memory {
        Memory {
            tracks: Vec::with_capacity(16),
        }
    }

    pub fn clear(&mut self) {
        self.tracks.clear();
    }

    pub fn observe(
        &mut self,
        sense: &Sense<'_>,
        level: &LevelNumbers,
        numbers: &BotNumbers,
        look: &Look,
        rng: &mut Rng,
        tick: u64,
    ) {
        for track in &mut self.tracks {
            track.visible = false;
        }
        self.see(sense, level, numbers, look, rng, tick);
        self.hear(sense, level, numbers, rng, tick);
        self.listen(sense, numbers, tick);
        let keep = numbers.ticks(level.memory_seconds);
        self.tracks
            .retain(|t| t.visible || tick.saturating_sub(t.seen_tick) <= keep);
    }

    fn see(
        &mut self,
        sense: &Sense<'_>,
        level: &LevelNumbers,
        numbers: &BotNumbers,
        look: &Look,
        rng: &mut Rng,
        tick: u64,
    ) {
        let play = &numbers.play;
        let reset = numbers.ticks(play.reaction_reset_seconds);
        let blinded = sense.blind >= play.blind_sight;
        for seen in sense.enemies.iter().filter(|_| !blinded) {
            let (yaw, pitch) = look_angles(look.eye, seen.chest);
            let preaimed =
                angle_between(look.yaw, look.pitch, yaw, pitch) < radians(play.preaim_cone_deg);
            let jitter = 1.0 + play.reaction_jitter * (2.0 * rng.unit() - 1.0);
            let aimed = if preaimed {
                play.preaimed_reaction
            } else {
                1.0
            };
            let reaction = level.reaction_seconds * jitter * aimed * (1.0 + sense.blind);
            let react_tick = tick + numbers.ticks(reaction);
            let slot = self.tracks.iter().position(|t| t.id == Some(seen.id));
            let track = match slot {
                Some(index) => &mut self.tracks[index],
                None => {
                    self.tracks.push(Track {
                        id: Some(seen.id),
                        at: seen.feet,
                        head: seen.head,
                        chest: seen.chest,
                        velocity: seen.velocity,
                        seen_tick: tick,
                        visible: true,
                        visible_since: tick,
                        react_tick,
                        source: Source::Seen,
                        carries_device: seen.carries_device,
                        called_tick: None,
                        sound: None,
                    });
                    continue;
                }
            };
            let was_seen =
                track.source == Source::Seen && tick.saturating_sub(track.seen_tick) <= reset;
            if !was_seen {
                track.visible_since = tick;
                track.react_tick = react_tick;
            }
            track.at = seen.feet;
            track.head = seen.head;
            track.chest = seen.chest;
            track.velocity = seen.velocity;
            track.seen_tick = tick;
            track.visible = true;
            track.source = Source::Seen;
            track.carries_device = seen.carries_device;
            track.sound = None;
        }
    }

    fn merge_at(&mut self, at: Vec3, merge_m: f32) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (index, track) in self.tracks.iter().enumerate() {
            if track.visible {
                continue;
            }
            let distance = (track.at - at).length();
            if distance <= merge_m && best.is_none_or(|(_, d)| distance < d) {
                best = Some((index, distance));
            }
        }
        best.map(|(index, _)| index)
    }

    fn remember(
        &mut self,
        at: Vec3,
        source: Source,
        sound: Option<SoundKind>,
        merge_m: f32,
        tick: u64,
        eye_up: f32,
    ) {
        let head = at + Vec3::new(0.0, eye_up, 0.0);
        if let Some(index) = self.merge_at(at, merge_m) {
            let track = &mut self.tracks[index];
            if track.source == Source::Seen && track.seen_tick >= tick {
                return;
            }
            track.at = at;
            track.head = head;
            track.chest = (at + head) * 0.5;
            track.seen_tick = track.seen_tick.max(tick);
            if track.source != Source::Seen || source != Source::Heard {
                track.source = source;
            }
            track.sound = sound;
            return;
        }
        self.tracks.push(Track {
            id: None,
            at,
            head,
            chest: (at + head) * 0.5,
            velocity: Vec3::ZERO,
            seen_tick: tick,
            visible: false,
            visible_since: tick,
            react_tick: tick,
            source,
            carries_device: false,
            called_tick: None,
            sound,
        });
    }

    fn hear(
        &mut self,
        sense: &Sense<'_>,
        level: &LevelNumbers,
        numbers: &BotNumbers,
        rng: &mut Rng,
        tick: u64,
    ) {
        let eye_up = numbers.player.eye_height_m;
        for heard in sense.sounds {
            if heard.loudness < level.hearing_loudness {
                continue;
            }
            let spread = level.hearing_error_m * clamp(1.0 - heard.loudness, 0.0, 1.0);
            let (sin, cos) = sin_cos(rng.unit() * 2.0 * PI);
            let off = Vec3::new(cos, 0.0, sin) * spread * rng.unit().sqrt();
            let at = heard.at + off;
            self.remember(
                at,
                Source::Heard,
                Some(heard.kind),
                numbers.play.merge_m,
                tick,
                eye_up,
            );
        }
    }

    fn listen(&mut self, sense: &Sense<'_>, numbers: &BotNumbers, tick: u64) {
        let eye_up = numbers.player.eye_height_m;
        for call in sense.calls {
            match call.kind {
                CallKind::Spotted | CallKind::Device | CallKind::Down | CallKind::Heard => {
                    self.remember(
                        call.at,
                        Source::Called,
                        None,
                        numbers.play.merge_m,
                        call.tick.min(tick),
                        eye_up,
                    );
                }
                CallKind::Hurt | CallKind::Flash => {}
            }
        }
    }

    pub fn target(&self, eye: Vec3, tick: u64) -> Option<Track> {
        self.tracks
            .iter()
            .filter(|t| t.visible && t.react_tick <= tick)
            .min_by(|a, b| {
                let da = (a.chest - eye).length();
                let db = (b.chest - eye).length();
                da.total_cmp(&db).then(a.id.cmp(&b.id))
            })
            .copied()
    }

    pub fn visible_count(&self) -> usize {
        self.tracks.iter().filter(|t| t.visible).count()
    }

    pub fn lost(&self, tick: u64, within: u64) -> Option<Track> {
        self.tracks
            .iter()
            .filter(|t| {
                !t.visible
                    && t.source != Source::Heard
                    && tick.saturating_sub(t.seen_tick) <= within
            })
            .max_by(|a, b| a.seen_tick.cmp(&b.seen_tick).then(b.id.cmp(&a.id)))
            .copied()
    }

    pub fn heard(&self, tick: u64, within: u64) -> Option<Track> {
        self.tracks
            .iter()
            .filter(|t| {
                !t.visible
                    && t.source == Source::Heard
                    && tick.saturating_sub(t.seen_tick) <= within
            })
            .max_by(|a, b| a.seen_tick.cmp(&b.seen_tick))
            .copied()
    }

    pub fn nearest(&self, p: Vec3) -> Option<f32> {
        self.tracks
            .iter()
            .map(|t| (t.at - p).length())
            .min_by(f32::total_cmp)
    }

    pub fn take_call(&mut self, tick: u64, repeat: u64) -> Option<Track> {
        let index = self.tracks.iter().position(|t| {
            t.visible
                && t.called_tick
                    .is_none_or(|c| tick.saturating_sub(c) >= repeat)
        })?;
        self.tracks[index].called_tick = Some(tick);
        Some(self.tracks[index])
    }
}
