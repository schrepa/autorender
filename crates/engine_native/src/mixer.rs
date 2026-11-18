use std::f32::consts::FRAC_PI_2;
use std::sync::Arc;

use engine_core::math::{clamp, sin_cos};

pub(crate) struct Pcm {
    pub(crate) samples: Vec<f32>,
    pub(crate) channels: usize,
    pub(crate) rate: u32,
}

impl Pcm {
    pub(crate) fn frames(&self) -> usize {
        self.samples.len() / self.channels
    }

    fn at(&self, frame: usize, channel: usize) -> f32 {
        self.samples
            .get(frame * self.channels + channel)
            .copied()
            .unwrap_or(0.0)
    }
}

pub(crate) enum Command {
    Start {
        voice: u32,
        pcm: Arc<Pcm>,
        gain: f32,
        pan: f32,
        rate: f32,
        looping: bool,
        offset_seconds: f32,
    },
    Tune {
        voice: u32,
        gain: f32,
        pan: f32,
        rate: f32,
        seconds: f32,
    },
    Stop {
        voice: u32,
        seconds: f32,
    },
    Master {
        gain: f32,
        seconds: f32,
    },
}

#[derive(Clone, Copy)]
struct Ramp {
    value: f32,
    target: f32,
    step: f32,
    left: u32,
}

impl Ramp {
    fn at(value: f32) -> Self {
        Self {
            value,
            target: value,
            step: 0.0,
            left: 0,
        }
    }

    fn toward(&mut self, target: f32, frames: u32) {
        self.target = target;
        self.left = frames;
        if frames == 0 {
            self.value = target;
            self.step = 0.0;
        } else {
            self.step = (target - self.value) / frames as f32;
        }
    }

    fn next(&mut self) -> f32 {
        if self.left > 0 {
            self.left -= 1;
            self.value = if self.left == 0 {
                self.target
            } else {
                self.value + self.step
            };
        }
        self.value
    }

    fn moving(&self) -> bool {
        self.left > 0
    }
}

struct Voice {
    id: u32,
    pcm: Arc<Pcm>,
    position: f64,
    gain: Ramp,
    pan: Ramp,
    rate: Ramp,
    looping: bool,
    stopping: bool,
    sides: (f32, f32),
}

fn sides(pan: f32, stereo: bool) -> (f32, f32) {
    let pan = clamp(pan, -1.0, 1.0);
    let x = if stereo {
        if pan <= 0.0 { pan + 1.0 } else { pan }
    } else {
        (pan + 1.0) * 0.5
    };
    let (sin, cos) = sin_cos(x * FRAC_PI_2);
    (cos, sin)
}

impl Voice {
    fn sample(&self, channel: usize) -> f32 {
        let frames = self.pcm.frames();
        let index = self.position.floor();
        let fraction = (self.position - index) as f32;
        let first = index as usize;
        let second = if first + 1 < frames {
            Some(first + 1)
        } else if self.looping {
            Some(0)
        } else {
            None
        };
        let a = self.pcm.at(first, channel);
        let b = second.map_or(0.0, |frame| self.pcm.at(frame, channel));
        a + (b - a) * fraction
    }

    fn render(&mut self, device_rate: u32, left: &mut [f32], right: &mut [f32]) -> bool {
        let stereo = self.pcm.channels > 1;
        let frames = self.pcm.frames() as f64;
        let ratio = f64::from(self.pcm.rate) / f64::from(device_rate);
        for (left, right) in left.iter_mut().zip(right.iter_mut()) {
            if self.stopping && !self.gain.moving() {
                return false;
            }
            if self.position >= frames {
                if !self.looping || frames <= 0.0 {
                    return false;
                }
                self.position %= frames;
            }
            let gain = self.gain.next();
            if self.pan.moving() {
                self.sides = sides(self.pan.next(), stereo);
            }
            let (to_left, to_right) = self.sides;
            if stereo {
                let (in_left, in_right) = (self.sample(0), self.sample(1));
                if self.pan.value <= 0.0 {
                    *left += (in_left + in_right * to_left) * gain;
                    *right += in_right * to_right * gain;
                } else {
                    *left += in_left * to_left * gain;
                    *right += (in_right + in_left * to_right) * gain;
                }
            } else {
                let sample = self.sample(0) * gain;
                *left += sample * to_left;
                *right += sample * to_right;
            }
            self.position += f64::from(self.rate.next()) * ratio;
        }
        !(self.stopping && !self.gain.moving())
    }
}

pub(crate) struct Mixer {
    rate: u32,
    voices: Vec<Voice>,
    master: Ramp,
    ended: Vec<u32>,
}

impl Mixer {
    pub(crate) fn new(rate: u32) -> Self {
        Self {
            rate,
            voices: Vec::new(),
            master: Ramp::at(1.0),
            ended: Vec::new(),
        }
    }

    fn frames(&self, seconds: f32) -> u32 {
        (f64::from(seconds) * f64::from(self.rate)).round() as u32
    }

    pub(crate) fn apply(&mut self, command: Command) {
        match command {
            Command::Start {
                voice,
                pcm,
                gain,
                pan,
                rate,
                looping,
                offset_seconds,
            } => {
                let stereo = pcm.channels > 1;
                let position = f64::from(offset_seconds) * f64::from(pcm.rate);
                self.voices.push(Voice {
                    id: voice,
                    pcm,
                    position,
                    gain: Ramp::at(gain),
                    pan: Ramp::at(pan),
                    rate: Ramp::at(rate),
                    looping,
                    stopping: false,
                    sides: sides(pan, stereo),
                });
            }
            Command::Tune {
                voice,
                gain,
                pan,
                rate,
                seconds,
            } => {
                let frames = self.frames(seconds);
                if let Some(held) = self.voices.iter_mut().find(|held| held.id == voice) {
                    held.gain.toward(gain, frames);
                    held.pan.toward(pan, frames);
                    held.rate.toward(rate, frames);
                    if frames == 0 {
                        held.sides = sides(pan, held.pcm.channels > 1);
                    }
                }
            }
            Command::Stop { voice, seconds } => {
                let frames = self.frames(seconds);
                if let Some(held) = self.voices.iter_mut().find(|held| held.id == voice) {
                    held.gain.toward(0.0, frames);
                    held.stopping = true;
                }
            }
            Command::Master { gain, seconds } => {
                let frames = self.frames(seconds);
                self.master.toward(gain, frames);
            }
        }
    }

    pub(crate) fn render(&mut self, left: &mut [f32], right: &mut [f32]) {
        left.fill(0.0);
        right.fill(0.0);
        let rate = self.rate;
        let ended = &mut self.ended;
        self.voices.retain_mut(|voice| {
            let playing = voice.render(rate, left, right);
            if !playing {
                ended.push(voice.id);
            }
            playing
        });
        for (left, right) in left.iter_mut().zip(right.iter_mut()) {
            let master = self.master.next();
            *left *= master;
            *right *= master;
        }
    }

    pub(crate) fn take_ended(&mut self) -> std::vec::Drain<'_, u32> {
        self.ended.drain(..)
    }
}
