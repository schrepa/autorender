use std::collections::BTreeSet;
use std::sync::Arc;

use bevy_ecs::resource::Resource;
use serde::Serialize;

use crate::{Background, Files, Taken};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct SoundId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct VoiceId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Play {
    pub sound: SoundId,
    pub gain: f32,
    pub pan: f32,
    pub rate: f32,
    pub looping: bool,
    pub offset_seconds: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Tune {
    pub gain: f32,
    pub pan: f32,
    pub rate: f32,
    pub seconds: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Length {
    pub frames: u64,
    pub rate: u32,
    pub channels: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Loaded {
    Reading,
    Decoding,
    Ready(Length),
    Failed(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Refusal {
    Unknown,
    NotReady,
    Failed,
    NotRunning,
    Invalid,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum Output {
    #[default]
    Closed,
    Asked,
    Open {
        running: bool,
        rate: u32,
        channels: u32,
        latency_seconds: f32,
    },
    Failed(String),
    Absent(String),
}

#[derive(Clone, Debug)]
pub enum SoundOrder {
    Open { latency_seconds: Option<f32> },
    Decode { sound: SoundId, bytes: Arc<[u8]> },
    Start { voice: VoiceId, play: Play },
    Tune { voice: VoiceId, tune: Tune },
    Stop { voice: VoiceId, fade_seconds: f32 },
    Master { gain: f32, seconds: f32 },
}

#[derive(Clone, Debug, PartialEq)]
pub enum SoundReply {
    Opened {
        running: bool,
        rate: u32,
        channels: u32,
        latency_seconds: f32,
    },
    OpenFailed {
        error: String,
    },
    Absent {
        reason: String,
    },
    Decoded {
        sound: SoundId,
        length: Length,
    },
    DecodeFailed {
        sound: SoundId,
        error: String,
    },
    Ended {
        voice: VoiceId,
    },
    Dropped {
        voice: VoiceId,
        error: String,
    },
    Underruns {
        total: u64,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SoundCounts {
    pub asked: u32,
    pub reading: u32,
    pub decoding: u32,
    pub ready: u32,
    pub failed: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct VoiceCounts {
    pub started: u64,
    pub live: u32,
    pub ended: u64,
    pub dropped: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct RefusalCounts {
    pub unknown: u64,
    pub not_ready: u64,
    pub failed: u64,
    pub not_running: u64,
    pub invalid: u64,
}

impl RefusalCounts {
    fn count(&mut self, refusal: Refusal) {
        let slot = match refusal {
            Refusal::Unknown => &mut self.unknown,
            Refusal::NotReady => &mut self.not_ready,
            Refusal::Failed => &mut self.failed,
            Refusal::NotRunning => &mut self.not_running,
            Refusal::Invalid => &mut self.invalid,
        };
        *slot += 1;
    }

    pub fn total(&self) -> u64 {
        self.unknown + self.not_ready + self.failed + self.not_running + self.invalid
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SoundReport {
    pub output: &'static str,
    pub rate: Option<u32>,
    pub channels: Option<u32>,
    pub latency_seconds: Option<f32>,
    pub error: Option<String>,
    pub sounds: SoundCounts,
    pub bytes_read: u64,
    pub bytes_decoded: u64,
    pub voices: VoiceCounts,
    pub refused: RefusalCounts,
    pub orders_waiting: u32,
    pub underruns: Option<u64>,
}

type Reading = Background<Arc<[u8]>, String>;

#[derive(Resource, Default)]
pub struct Sounds {
    sounds: Vec<Loaded>,
    unasked: Vec<(SoundId, String)>,
    reading: Vec<(SoundId, Reading)>,
    orders: Vec<SoundOrder>,
    output: Output,
    voices: u32,
    live: BTreeSet<VoiceId>,
    started: u64,
    ended: u64,
    dropped: u64,
    refused: RefusalCounts,
    bytes_read: u64,
    bytes_decoded: u64,
    last_error: Option<String>,
    underruns: Option<u64>,
}

fn finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

impl Sounds {
    pub fn open(&mut self, latency_seconds: Option<f32>) {
        if self.output != Output::Closed {
            return;
        }
        if latency_seconds.is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0) {
            self.refused.count(Refusal::Invalid);
            return;
        }
        self.output = Output::Asked;
        self.orders.push(SoundOrder::Open { latency_seconds });
    }

    pub fn load(&mut self, path: &str) -> SoundId {
        let sound = SoundId(u32::try_from(self.sounds.len()).unwrap_or(u32::MAX));
        self.sounds.push(Loaded::Reading);
        self.unasked.push((sound, path.to_owned()));
        sound
    }

    pub fn loaded(&self, sound: SoundId) -> Option<&Loaded> {
        self.sounds.get(sound.0 as usize)
    }

    pub fn play(&mut self, play: Play) -> Result<VoiceId, Refusal> {
        let refusal = if !finite(&[play.gain, play.pan, play.rate, play.offset_seconds])
            || play.rate <= 0.0
            || play.offset_seconds < 0.0
        {
            Some(Refusal::Invalid)
        } else {
            match self.sounds.get(play.sound.0 as usize) {
                None => Some(Refusal::Unknown),
                Some(Loaded::Reading | Loaded::Decoding) => Some(Refusal::NotReady),
                Some(Loaded::Failed(_)) => Some(Refusal::Failed),
                Some(Loaded::Ready(_)) => match self.output {
                    Output::Open { running: true, .. } => None,
                    Output::Open { running: false, .. }
                    | Output::Closed
                    | Output::Asked
                    | Output::Failed(_)
                    | Output::Absent(_) => Some(Refusal::NotRunning),
                },
            }
        };
        if let Some(refusal) = refusal {
            self.refused.count(refusal);
            return Err(refusal);
        }
        self.voices += 1;
        let voice = VoiceId(self.voices);
        self.live.insert(voice);
        self.started += 1;
        self.orders.push(SoundOrder::Start { voice, play });
        Ok(voice)
    }

    pub fn tune(&mut self, voice: VoiceId, tune: Tune) {
        if !finite(&[tune.gain, tune.pan, tune.rate, tune.seconds])
            || tune.rate <= 0.0
            || tune.seconds < 0.0
        {
            self.refused.count(Refusal::Invalid);
            return;
        }
        if self.live.contains(&voice) {
            self.orders.push(SoundOrder::Tune { voice, tune });
        }
    }

    pub fn stop(&mut self, voice: VoiceId, fade_seconds: f32) {
        if !fade_seconds.is_finite() || fade_seconds < 0.0 {
            self.refused.count(Refusal::Invalid);
            return;
        }
        if self.live.contains(&voice) {
            self.orders.push(SoundOrder::Stop {
                voice,
                fade_seconds,
            });
        }
    }

    pub fn master(&mut self, gain: f32, seconds: f32) {
        if !finite(&[gain, seconds]) || seconds < 0.0 {
            self.refused.count(Refusal::Invalid);
            return;
        }
        self.orders.push(SoundOrder::Master { gain, seconds });
    }

    pub fn live(&self, voice: VoiceId) -> bool {
        self.live.contains(&voice)
    }

    pub fn output(&self) -> &Output {
        &self.output
    }

    pub fn take_orders(&mut self, files: &mut Files) -> Vec<SoundOrder> {
        for (sound, path) in std::mem::take(&mut self.unasked) {
            let reading = files.read(&path, |bytes| Ok(Arc::<[u8]>::from(bytes)));
            self.reading.push((sound, reading));
        }
        for (sound, reading) in std::mem::take(&mut self.reading) {
            match reading.take(files) {
                Taken::Waiting(reading) => self.reading.push((sound, reading)),
                Taken::Ready(bytes) => {
                    self.bytes_read += bytes.len() as u64;
                    self.set(sound, Loaded::Decoding);
                    self.orders.push(SoundOrder::Decode { sound, bytes });
                }
                Taken::Failed(error) => self.set(sound, Loaded::Failed(error.to_string())),
            }
        }
        std::mem::take(&mut self.orders)
    }

    fn set(&mut self, sound: SoundId, loaded: Loaded) {
        if let Some(slot) = self.sounds.get_mut(sound.0 as usize) {
            *slot = loaded;
        }
    }

    pub fn hear(&mut self, reply: SoundReply) {
        match reply {
            SoundReply::Opened {
                running,
                rate,
                channels,
                latency_seconds,
            } => {
                self.output = Output::Open {
                    running,
                    rate,
                    channels,
                    latency_seconds,
                };
            }
            SoundReply::OpenFailed { error } => self.output = Output::Failed(error),
            SoundReply::Absent { reason } => self.output = Output::Absent(reason),
            SoundReply::Decoded { sound, length } => {
                if length.rate == 0 || length.channels == 0 {
                    self.set(sound, Loaded::Failed("empty_format".to_owned()));
                    return;
                }
                self.bytes_decoded += length.frames * u64::from(length.channels) * 4;
                self.set(sound, Loaded::Ready(length));
            }
            SoundReply::DecodeFailed { sound, error } => self.set(sound, Loaded::Failed(error)),
            SoundReply::Ended { voice } => {
                if self.live.remove(&voice) {
                    self.ended += 1;
                }
            }
            SoundReply::Dropped { voice, error } => {
                if self.live.remove(&voice) {
                    self.dropped += 1;
                }
                self.last_error = Some(error);
            }
            SoundReply::Underruns { total } => self.underruns = Some(total),
        }
    }

    pub fn report(&self) -> SoundReport {
        let mut sounds = SoundCounts {
            asked: u32::try_from(self.sounds.len()).unwrap_or(u32::MAX),
            ..SoundCounts::default()
        };
        for loaded in &self.sounds {
            match loaded {
                Loaded::Reading => sounds.reading += 1,
                Loaded::Decoding => sounds.decoding += 1,
                Loaded::Ready(_) => sounds.ready += 1,
                Loaded::Failed(_) => sounds.failed += 1,
            }
        }
        let (output, rate, channels, latency_seconds, error) = match &self.output {
            Output::Closed => ("closed", None, None, None, None),
            Output::Asked => ("asked", None, None, None, None),
            Output::Open {
                running,
                rate,
                channels,
                latency_seconds,
            } => (
                if *running { "running" } else { "suspended" },
                Some(*rate),
                Some(*channels),
                Some(*latency_seconds),
                None,
            ),
            Output::Failed(error) => ("failed", None, None, None, Some(error.clone())),
            Output::Absent(reason) => ("absent", None, None, None, Some(reason.clone())),
        };
        SoundReport {
            output,
            rate,
            channels,
            latency_seconds,
            error: error.or_else(|| self.last_error.clone()),
            sounds,
            bytes_read: self.bytes_read,
            bytes_decoded: self.bytes_decoded,
            voices: VoiceCounts {
                started: self.started,
                live: u32::try_from(self.live.len()).unwrap_or(u32::MAX),
                ended: self.ended,
                dropped: self.dropped,
            },
            refused: self.refused,
            orders_waiting: u32::try_from(self.orders.len()).unwrap_or(u32::MAX),
            underruns: self.underruns,
        }
    }
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(crate) fn names(world: &mut bevy_ecs::world::World) {
    let entry = crate::NameEntry::read("sound.state", crate::Kind::State, |reach, _| {
        Ok(reach.world.get_resource::<Sounds>().map_or_else(
            || crate::unavailable("no_sounds"),
            |sounds| serde_json::to_value(sounds.report()).unwrap_or_default(),
        ))
    });
    crate::register(world, entry);
}
