use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::glam::Mat4;
use engine_core::math::{clamp, max, min};
use engine_core::{Error, ErrorCode};

use super::rig::{FigureFiles, Rig};
use crate::play::skinned::{ClipNo, FigureClock, FigurePlay, UpperLayer};

pub(super) fn clip_in(rig: &Rig, clip: ClipNo, path: &str) -> Result<usize, Error> {
    (clip.0 < rig.clips.len())
        .then_some(clip.0)
        .ok_or_else(|| Error::new(ErrorCode::UnknownReference, path.to_owned()).got(clip.0))
}

pub(super) fn clips_in(rig: &Rig, play: &FigurePlay) -> Result<(usize, Option<usize>), Error> {
    let base = clip_in(rig, play.base.clip, "$.figure_play.base.clip")?;
    for blended in &play.base.blend {
        clip_in(rig, blended.clip, "$.figure_play.base.blend")?;
    }
    let upper = (play.upper.as_ref())
        .map(|upper| clip_in(rig, upper.clip, "$.figure_play.upper.clip"))
        .transpose()?;
    Ok((base, upper))
}

fn fitted(rig: &Rig, clip: usize, fit: Option<f32>, speed: f32) -> f32 {
    match fit {
        Some(fit) if fit > 0.0 => rig.duration(clip) / fit,
        Some(_) | None => speed,
    }
}

#[derive(Clone, Debug)]
pub(super) struct LayerClock {
    pub(super) clip: usize,
    cue: u32,
    speed: f32,
    looping: bool,
    seconds: f32,
    step: f32,
    end: Option<f32>,
    synced: bool,
}

impl LayerClock {
    fn started(clip: usize, cue: u32, speed: f32, looping: bool) -> Self {
        Self {
            clip,
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

    fn restarts(&self, clip: usize, cue: u32) -> bool {
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

#[derive(Clone, Debug)]
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

pub(super) type Mask = Result<Arc<[f32]>, Error>;

#[derive(Clone, Debug)]
pub(super) struct Upper {
    pub(super) clock: LayerClock,
    pub(super) layer: UpperLayer,
    pub(super) mask: Mask,
    pub(super) played: Option<Mask>,
}

fn masked(files: &mut FigureFiles, rig: &Rig, layer: &UpperLayer) -> (Mask, Option<Mask>) {
    let mask = files.mask(rig, layer.from.0, layer.feather, "$.figure_play.upper.from");
    let played = (layer.played_from)
        .map(|from| files.mask(rig, from.0, 0, "$.figure_play.upper.played_from"));
    (mask, played)
}

#[derive(Clone, Debug)]
pub(super) struct UpperFade {
    pub(super) from: Option<Upper>,
    elapsed: f32,
    step: f32,
    seconds: f32,
}

impl UpperFade {
    pub(super) fn weight(&self, back: f32) -> f32 {
        clamp((self.elapsed - back * self.step) / self.seconds, 0.0, 1.0)
    }
}

fn faded(old: Option<Upper>, new: Option<&Upper>) -> Option<UpperFade> {
    let additive = |upper: &&Upper| upper.layer.additive;
    let eased = new
        .filter(|upper| !upper.layer.additive)
        .map_or(0.0, |upper| upper.layer.fade_seconds);
    let seconds = if eased > 0.0 {
        eased
    } else {
        new.filter(additive)
            .or(old.as_ref().filter(additive))
            .map_or(0.0, |upper| upper.layer.fade_seconds)
    };
    (seconds > 0.0).then(|| UpperFade {
        from: old.filter(|upper| upper.layer.additive),
        elapsed: 0.0,
        step: 0.0,
        seconds,
    })
}

#[derive(Clone)]
pub(super) struct Clocks {
    pub(super) limbed: bool,
    pub(super) rig: Result<Arc<Rig>, Error>,
    pub(super) base: Option<LayerClock>,
    pub(super) fade: Option<Fade>,
    pub(super) upper: Option<Upper>,
    pub(super) upper_fade: Option<UpperFade>,
    pub(super) fit: f32,
    pub(super) fit_rest: Option<f32>,
    pub(super) fitted: bool,
    pub(super) top: Option<f32>,
    pub(super) world: Option<Mat4>,
}

impl Clocks {
    pub(super) fn new(limbed: bool, rig: Result<Arc<Rig>, Error>) -> Self {
        Self {
            limbed,
            rig,
            base: None,
            fade: None,
            upper: None,
            upper_fade: None,
            fit: 1.0,
            fit_rest: None,
            fitted: false,
            top: None,
            world: None,
        }
    }

    pub(super) fn tick(
        &mut self,
        (rig, play): (&Rig, Option<&FigurePlay>),
        files: &mut FigureFiles,
        delta: f32,
    ) -> Result<(), Error> {
        let Some(play) = play else {
            self.base = None;
            self.fade = None;
            self.upper = None;
            self.upper_fade = None;
            return Ok(());
        };
        let (base_clip, upper_clip) = clips_in(rig, play)?;
        if let Some(fade) = &mut self.fade {
            if fade.elapsed >= fade.seconds {
                self.fade = None;
            } else {
                fade.from.advance(rig.duration(fade.from.clip), delta);
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
            && clock.clip != base_clip
        {
            let before = rig.duration(clock.clip);
            let share = if before > 0.0 {
                clock.seconds / before
            } else {
                0.0
            };
            clock.clip = base_clip;
            clock.seconds = share * rig.duration(base_clip);
            clock.step = 0.0;
        }
        let rate = fitted(rig, base_clip, base.fit_seconds, base.speed);
        match &mut self.base {
            Some(clock) if !clock.restarts(base_clip, base.cue) => {
                clock.speed = rate;
                clock.looping = base.looping;
                clock.advance(rig.duration(base_clip), delta);
            }
            Some(_) | None => {
                let started = LayerClock::started(base_clip, base.cue, rate, base.looping);
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
        let wanted = play.upper.as_ref().zip(upper_clip);
        self.upper_tick(rig, wanted, files, delta);
        Ok(())
    }

    fn upper_tick(
        &mut self,
        rig: &Rig,
        wanted: Option<(&UpperLayer, usize)>,
        files: &mut FigureFiles,
        delta: f32,
    ) {
        if let Some(fade) = &mut self.upper_fade {
            if fade.elapsed >= fade.seconds {
                self.upper_fade = None;
            } else {
                if let Some(from) = &mut fade.from {
                    from.clock.advance(rig.duration(from.clock.clip), delta);
                }
                let before = fade.elapsed;
                fade.elapsed = min(before + delta, fade.seconds);
                fade.step = fade.elapsed - before;
            }
        }
        match (wanted, &mut self.upper) {
            (None, None) => {}
            (Some((upper, clip)), Some(held)) if !held.clock.restarts(clip, upper.cue) => {
                held.clock.speed = fitted(rig, clip, upper.fit_seconds, upper.speed);
                held.clock.looping = upper.looping;
                held.clock.advance(rig.duration(clip), delta);
                if held.layer != *upper {
                    let remask = held.layer.from != upper.from
                        || held.layer.feather != upper.feather
                        || held.layer.played_from != upper.played_from;
                    held.layer = *upper;
                    if remask {
                        (held.mask, held.played) = masked(files, rig, upper);
                    }
                }
            }
            (Some(_) | None, Some(_) | None) => {
                let started = wanted.map(|(upper, clip)| {
                    let (mask, played) = masked(files, rig, upper);
                    let speed = fitted(rig, clip, upper.fit_seconds, upper.speed);
                    Upper {
                        clock: LayerClock::started(clip, upper.cue, speed, upper.looping)
                            .windowed(upper.window),
                        layer: *upper,
                        mask,
                        played,
                    }
                });
                let old = std::mem::replace(&mut self.upper, started);
                self.upper_fade = faded(old, self.upper.as_ref());
            }
        }
    }

    pub(super) fn read(&self, rig: &Rig, facts: &mut FigureClock) {
        let of = |clock: Option<&LayerClock>| {
            clock.map_or((0.0, false), |clock| {
                (clock.seconds, clock.done(rig.duration(clock.clip)))
            })
        };
        let upper = self.upper.as_ref().map(|upper| &upper.clock);
        (facts.base_seconds, facts.base_done) = of(self.base.as_ref());
        (facts.upper_seconds, facts.upper_done) = of(upper);
        facts.base_cue = self.base.as_ref().map_or(0, |clock| clock.cue);
        facts.upper_cue = upper.map(|clock| clock.cue);
        facts.upper_rate = upper.map_or(0.0, |clock| clock.speed);
        facts.fit = self.fit;
    }
}

#[derive(Resource, Default)]
pub struct FigureClocks {
    pub(super) held: Arc<BTreeMap<Entity, Clocks>>,
}
