use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::Error;
use engine_core::math::{clamp, max, min};

use super::files::Rig;
use crate::play::figures::{BaseLayer, Figure, FigurePlay};

pub(super) struct Readout {
    pub(super) base_seconds: f32,
    pub(super) base_done: bool,
    pub(super) base_cue: u32,
    pub(super) upper_seconds: f32,
    pub(super) upper_done: bool,
    pub(super) upper_cue: Option<u32>,
}

fn rate(rig: &Rig, base: &BaseLayer) -> f32 {
    match base.fit_seconds {
        Some(fit) if fit > 0.0 => rig.duration(&base.clip) / fit,
        Some(_) | None => base.speed,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct LayerClock {
    pub(super) clip: String,
    cue: u32,
    speed: f32,
    looping: bool,
    seconds: f32,
    step: f32,
    end: Option<f32>,
    synced: bool,
}

impl LayerClock {
    fn started(clip: &str, cue: u32, speed: f32, looping: bool) -> Self {
        Self {
            clip: clip.to_owned(),
            cue,
            speed,
            looping,
            seconds: 0.0,
            step: 0.0,
            end: None,
            synced: false,
        }
    }

    fn windowed(mut self, window: Option<[f32; 2]>) -> Self {
        if let Some([start, end]) = window {
            self.seconds = max(start, 0.0);
            self.end = Some(max(end, start));
        }
        self
    }

    fn limit(&self, duration: f32) -> f32 {
        self.end.map_or(duration, |end| min(end, duration))
    }

    fn restarts(&self, clip: &str, cue: u32) -> bool {
        self.clip != clip || self.cue != cue
    }

    fn advance(&mut self, duration: f32, delta: f32) {
        let before = self.seconds;
        let reached = before + delta * self.speed;
        if self.looping && duration > 0.0 {
            self.seconds = reached.rem_euclid(duration);
            self.step = reached - before;
        } else {
            self.seconds = min(reached, self.limit(duration));
            self.step = self.seconds - before;
        }
    }

    pub(super) fn at(&self, duration: f32, back: f32) -> f32 {
        let seconds = self.seconds - back * self.step;
        if self.looping && duration > 0.0 {
            seconds.rem_euclid(duration)
        } else {
            clamp(seconds, 0.0, duration)
        }
    }

    fn done(&self, duration: f32) -> bool {
        !self.looping && self.seconds >= self.limit(duration)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Fade {
    pub(super) from: LayerClock,
    elapsed: f32,
    step: f32,
    seconds: f32,
}

impl Fade {
    pub(super) fn weight(&self, back: f32) -> f32 {
        clamp((self.elapsed - back * self.step) / self.seconds, 0.0, 1.0)
    }
}

pub(super) struct Clocks {
    pub(super) figure: Figure,
    pub(super) rig: Result<Arc<Rig>, Error>,
    pub(super) base: Option<LayerClock>,
    pub(super) fade: Option<Fade>,
    pub(super) upper: Option<LayerClock>,
}

impl Clocks {
    pub(super) fn new(figure: Figure, rig: Result<Arc<Rig>, Error>) -> Self {
        Self {
            figure,
            rig,
            base: None,
            fade: None,
            upper: None,
        }
    }

    pub(super) fn tick(&mut self, play: Option<&FigurePlay>, delta: f32) {
        let Ok(rig) = &self.rig else {
            return;
        };
        let Some(play) = play else {
            self.base = None;
            self.fade = None;
            self.upper = None;
            return;
        };
        if let Some(fade) = &mut self.fade {
            if fade.elapsed >= fade.seconds {
                self.fade = None;
            } else {
                fade.from.advance(rig.duration(&fade.from.clip), delta);
                let before = fade.elapsed;
                fade.elapsed = min(before + delta, fade.seconds);
                fade.step = fade.elapsed - before;
            }
        }
        let base = &play.base;
        if base.synced
            && let Some(clock) = &mut self.base
            && clock.synced
            && clock.cue == base.cue
            && clock.clip != base.clip
        {
            let before = rig.duration(&clock.clip);
            let share = if before > 0.0 {
                clock.seconds / before
            } else {
                0.0
            };
            clock.clip.clone_from(&base.clip);
            clock.seconds = share * rig.duration(&base.clip);
            clock.step = 0.0;
        }
        match &mut self.base {
            Some(clock) if !clock.restarts(&base.clip, base.cue) => {
                clock.speed = rate(rig, base);
                clock.looping = base.looping;
                clock.advance(rig.duration(&base.clip), delta);
            }
            Some(_) | None => {
                let started =
                    LayerClock::started(&base.clip, base.cue, rate(rig, base), base.looping);
                let old = self.base.replace(started);
                self.fade = old.filter(|_| base.fade_seconds > 0.0).map(|from| Fade {
                    from,
                    elapsed: 0.0,
                    step: 0.0,
                    seconds: base.fade_seconds,
                });
            }
        }
        if let Some(clock) = &mut self.base {
            clock.synced = base.synced;
        }
        match (&play.upper, &mut self.upper) {
            (None, _) => self.upper = None,
            (Some(upper), Some(clock)) if !clock.restarts(&upper.clip, upper.cue) => {
                clock.speed = upper.speed;
                clock.looping = upper.looping;
                clock.advance(rig.duration(&upper.clip), delta);
            }
            (Some(upper), Some(_) | None) => {
                self.upper = Some(
                    LayerClock::started(&upper.clip, upper.cue, upper.speed, upper.looping)
                        .windowed(upper.window),
                );
            }
        }
    }

    pub(super) fn readout(&self) -> Readout {
        let duration = |clock: &LayerClock| {
            self.rig
                .as_ref()
                .map_or(0.0, |rig| rig.duration(&clock.clip))
        };
        let of = |clock: Option<&LayerClock>| {
            clock.map_or((0.0, false), |clock| {
                (clock.seconds, clock.done(duration(clock)))
            })
        };
        let (base_seconds, base_done) = of(self.base.as_ref());
        let (upper_seconds, upper_done) = of(self.upper.as_ref());
        Readout {
            base_seconds,
            base_done,
            base_cue: self.base.as_ref().map_or(0, |clock| clock.cue),
            upper_seconds,
            upper_done,
            upper_cue: self.upper.as_ref().map(|clock| clock.cue),
        }
    }
}

#[derive(Resource, Default)]
pub struct FigureClocks {
    pub(super) held: BTreeMap<Entity, Clocks>,
}
