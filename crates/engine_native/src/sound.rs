use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, PoisonError};

use engine::{Game, GameHost};
use engine_core::{Length, SoundId, SoundOrder, SoundReply, VoiceId};

use crate::mixer::{Command, Pcm};
use crate::output::{Device, decode, open};

type Decoded = (SoundId, Result<Pcm, String>);

struct Decoder {
    asks: Sender<(SoundId, Arc<[u8]>)>,
    results: Receiver<Decoded>,
}

impl Decoder {
    fn start() -> Result<Self, String> {
        let (asks, asked) = channel::<(SoundId, Arc<[u8]>)>();
        let (done, results) = channel::<Decoded>();
        std::thread::Builder::new()
            .name("engine_sound_decode".to_owned())
            .spawn(move || {
                for (sound, bytes) in asked {
                    if done.send((sound, decode(&bytes))).is_err() {
                        return;
                    }
                }
            })
            .map_err(|error| error.to_string())?;
        Ok(Self { asks, results })
    }
}

#[derive(Default)]
pub(crate) struct Speaker {
    device: Option<Device>,
    decoder: Option<Decoder>,
    pcm: BTreeMap<SoundId, Arc<Pcm>>,
    xruns: u64,
    silent: bool,
}

impl Speaker {
    pub(crate) fn silent() -> Self {
        Self {
            silent: true,
            ..Self::default()
        }
    }

    pub(crate) fn exchange<G: Game>(&mut self, host: &mut GameHost<G>) {
        for order in host.take_sound() {
            self.order(host, order);
        }
        self.decoded(host);
        self.heard(host);
    }

    fn order<G: Game>(&mut self, host: &mut GameHost<G>, order: SoundOrder) {
        match order {
            SoundOrder::Open { .. } if self.silent => host.hear_sound(SoundReply::Absent {
                reason: "offscreen_no_output".to_owned(),
            }),
            SoundOrder::Open { latency_seconds } => match open(latency_seconds) {
                Ok(device) => self.device = Some(device),
                Err(error) => host.hear_sound(SoundReply::OpenFailed { error }),
            },
            SoundOrder::Decode { sound, bytes } => {
                if self.decoder.is_none() {
                    match Decoder::start() {
                        Ok(decoder) => self.decoder = Some(decoder),
                        Err(error) => {
                            host.hear_sound(SoundReply::DecodeFailed { sound, error });
                            return;
                        }
                    }
                }
                let asked = self
                    .decoder
                    .as_ref()
                    .is_some_and(|decoder| decoder.asks.send((sound, bytes)).is_ok());
                if !asked {
                    host.hear_sound(SoundReply::DecodeFailed {
                        sound,
                        error: "decoder_stopped".to_owned(),
                    });
                }
            }
            SoundOrder::Start { voice, play } => {
                let sent = self.pcm.get(&play.sound).and_then(|pcm| {
                    let device = self.device.as_ref()?;
                    let command = Command::Start {
                        voice: voice.0,
                        pcm: Arc::clone(pcm),
                        gain: play.gain,
                        pan: play.pan,
                        rate: play.rate,
                        looping: play.looping,
                        offset_seconds: play.offset_seconds,
                    };
                    device.commands.send(command).ok()
                });
                if sent.is_none() {
                    host.hear_sound(SoundReply::Dropped {
                        voice,
                        error: "no_output_or_sound".to_owned(),
                    });
                }
            }
            SoundOrder::Tune { voice, tune } => self.send(Command::Tune {
                voice: voice.0,
                gain: tune.gain,
                pan: tune.pan,
                rate: tune.rate,
                seconds: tune.seconds,
            }),
            SoundOrder::Stop {
                voice,
                fade_seconds,
            } => self.send(Command::Stop {
                voice: voice.0,
                seconds: fade_seconds,
            }),
            SoundOrder::Master { gain, seconds } => self.send(Command::Master { gain, seconds }),
        }
    }

    fn send(&self, command: Command) {
        if let Some(device) = &self.device {
            let _ = device.commands.send(command);
        }
    }

    fn decoded<G: Game>(&mut self, host: &mut GameHost<G>) {
        let Some(decoder) = &self.decoder else {
            return;
        };
        while let Ok((sound, decoded)) = decoder.results.try_recv() {
            match decoded {
                Ok(pcm) => {
                    let length = Length {
                        frames: pcm.frames() as u64,
                        rate: pcm.rate,
                        channels: u32::try_from(pcm.channels).unwrap_or(u32::MAX),
                    };
                    self.pcm.insert(sound, Arc::new(pcm));
                    host.hear_sound(SoundReply::Decoded { sound, length });
                }
                Err(error) => host.hear_sound(SoundReply::DecodeFailed { sound, error }),
            }
        }
    }

    fn heard<G: Game>(&mut self, host: &mut GameHost<G>) {
        let Some(device) = &mut self.device else {
            return;
        };
        let ahead = device.ahead_nanos.load(Ordering::Relaxed);
        if !device.told && ahead > 0 {
            device.told = true;
            host.hear_sound(SoundReply::Opened {
                running: true,
                rate: device.rate,
                channels: device.channels,
                latency_seconds: ((ahead - 1) as f64 / 1e9) as f32,
            });
        }
        while let Ok(voice) = device.ended.try_recv() {
            host.hear_sound(SoundReply::Ended {
                voice: VoiceId(voice),
            });
        }
        let xruns = device.xruns.load(Ordering::Relaxed);
        if xruns != self.xruns {
            self.xruns = xruns;
            host.hear_sound(SoundReply::Underruns { total: xruns });
        }
        let failure = device
            .failure
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(error) = failure {
            host.hear_sound(SoundReply::OpenFailed { error });
        }
    }
}
