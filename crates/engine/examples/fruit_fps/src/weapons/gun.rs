use engine_math::{Random, min, ticks};

use super::numbers::{GunKind, GunNumbers, WeaponNumbers};
use super::recoil::{Aim, Kick, Recoil, Shot, Stance, bloomed_deg, fire, recover};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Holstered,
    Equipping { left: u32 },
    Ready,
    Reloading { left: u32, total: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pull {
    Nothing,
    Dry,
    Shoot,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gun {
    kind: GunKind,
    magazine: u32,
    reserve: u32,
    phase: Phase,
    cooldown_ticks: f32,
    armed: bool,
    scope_level: usize,
    scoped_ticks: u32,
    settled: usize,
    zoom_from: f32,
    zoom_ticks: u32,
    rescope: Option<usize>,
    recoil: Recoil,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pace {
    pub fire_rate: f32,
    pub reload: f32,
    pub spread: f32,
}

impl Pace {
    pub const SAME: Pace = Pace {
        fire_rate: 1.0,
        reload: 1.0,
        spread: 1.0,
    };
}

impl Default for Pace {
    fn default() -> Self {
        Pace::SAME
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct GunInput {
    pub fire_held: bool,
    pub reload_pressed: bool,
    pub scope_pressed: bool,
    pub pace: Pace,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct GunTick {
    pub shot: Option<Shot>,
    pub dry_fire: bool,
    pub reload_started: bool,
    pub reload_done: bool,
    pub kick: Kick,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GunView<'a> {
    pub name: &'a str,
    pub magazine: u32,
    pub reserve: u32,
    pub reload_progress: Option<f32>,
    pub spread_deg: f32,
    pub scope_level: usize,
    pub ready: bool,
}

impl Gun {
    pub fn new(kind: GunKind, numbers: &WeaponNumbers) -> Self {
        let gun = numbers.gun(kind);
        Self::with_ammo(kind, gun.magazine, gun.reserve)
    }

    pub fn with_ammo(kind: GunKind, magazine: u32, reserve: u32) -> Self {
        Self {
            kind,
            magazine,
            reserve,
            phase: Phase::Holstered,
            cooldown_ticks: 0.0,
            armed: false,
            scope_level: 0,
            scoped_ticks: 0,
            settled: 0,
            zoom_from: 1.0,
            zoom_ticks: 0,
            rescope: None,
            recoil: Recoil::default(),
        }
    }

    fn settled_scope(&self, gun: &GunNumbers, ticks_per_second: u32) -> usize {
        match &gun.scope {
            Some(scope) if self.scoped_ticks > ticks(scope.scope_in_seconds, ticks_per_second) => {
                self.scope_level
            }
            Some(_) | None => 0,
        }
    }

    fn rate_factor(&self, gun: &GunNumbers) -> f32 {
        match &gun.scope {
            Some(scope) if self.scope_level > 0 => scope.fire_rate_factor,
            Some(_) | None => 1.0,
        }
    }

    pub fn kind(&self) -> GunKind {
        self.kind
    }

    pub fn magazine(&self) -> u32 {
        self.magazine
    }

    pub fn reserve(&self) -> u32 {
        self.reserve
    }

    pub fn scope_level(&self) -> usize {
        self.scope_level
    }

    pub fn equip_left(&self) -> Option<u32> {
        match self.phase {
            Phase::Equipping { left } => Some(left),
            Phase::Holstered | Phase::Ready | Phase::Reloading { .. } => None,
        }
    }

    pub fn idle(&self) -> bool {
        self.phase == Phase::Ready
            && self.scope_level == 0
            && self.rescope.is_none()
            && self.cooldown_ticks <= 0.0
    }

    pub fn recoil(&self) -> &Recoil {
        &self.recoil
    }

    pub fn take_out(&mut self, numbers: &WeaponNumbers, ticks_per_second: u32) {
        let left = ticks(numbers.gun(self.kind).equip_seconds, ticks_per_second);
        self.phase = if left == 0 {
            Phase::Ready
        } else {
            Phase::Equipping { left }
        };
        self.unscope();
        self.armed = false;
    }

    pub fn put_away(&mut self) {
        self.phase = Phase::Holstered;
        self.unscope();
    }

    fn unscope(&mut self) {
        self.scope_level = 0;
        self.scoped_ticks = 0;
        self.settled = 0;
        self.zoom_from = 1.0;
        self.zoom_ticks = 0;
        self.rescope = None;
    }

    fn scope_to(&mut self, level: usize, gun: &GunNumbers, ticks_per_second: u32) {
        self.zoom_from = self.zoom_of(gun, ticks_per_second);
        self.zoom_ticks = 0;
        self.scope_level = level;
    }

    pub fn scope_raised(&self) -> bool {
        self.settled > 0
    }

    pub fn speed_factor(&self, numbers: &WeaponNumbers) -> f32 {
        let gun = numbers.gun(self.kind);
        match &gun.scope {
            Some(scope) if self.scope_level > 0 => gun.run_speed_factor * scope.aim_speed_factor,
            Some(_) | None => gun.run_speed_factor,
        }
    }

    pub fn view_zoom(&self, numbers: &WeaponNumbers, ticks_per_second: u32) -> f32 {
        self.zoom_of(numbers.gun(self.kind), ticks_per_second)
    }

    fn zoom_of(&self, gun: &GunNumbers, ticks_per_second: u32) -> f32 {
        let Some(scope) = &gun.scope else {
            return 1.0;
        };
        let level = self.scope_level.checked_sub(1);
        let Some(&target) = level.and_then(|index| scope.zoom.get(index)) else {
            return 1.0;
        };
        let raise = ticks(scope.scope_in_seconds, ticks_per_second);
        if raise == 0 {
            return target;
        }
        let share = min(self.zoom_ticks as f32 / raise as f32, 1.0);
        self.zoom_from + (target - self.zoom_from) * share
    }

    pub fn view<'a>(&self, numbers: &'a WeaponNumbers, stance: Stance) -> GunView<'a> {
        let gun = numbers.gun(self.kind);
        let reload_progress = match self.phase {
            Phase::Reloading { left, total } => Some(1.0 - left as f32 / total.max(1) as f32),
            Phase::Holstered | Phase::Equipping { .. } | Phase::Ready => None,
        };
        GunView {
            name: &gun.name,
            magazine: self.magazine,
            reserve: self.reserve,
            reload_progress,
            spread_deg: bloomed_deg(gun, stance, self.settled, self.recoil.shots),
            scope_level: self.scope_level,
            ready: self.phase == Phase::Ready,
        }
    }

    pub fn tick(
        &mut self,
        numbers: &WeaponNumbers,
        ticks_per_second: u32,
        input: GunInput,
        aim: Aim,
        stance: Stance,
        rng: &mut Random,
    ) -> GunTick {
        let gun = numbers.gun(self.kind);
        let cycled = self.cooldown_ticks <= 0.0;
        let mut out = GunTick::default();
        if !input.fire_held {
            self.armed = true;
        }
        let pull = match self.phase {
            Phase::Holstered => Pull::Nothing,
            Phase::Equipping { left } => {
                self.equip_step(left);
                Pull::Nothing
            }
            Phase::Reloading { left, total } => {
                self.reload_step(gun, left, total, &mut out);
                Pull::Nothing
            }
            Phase::Ready => self.ready_step(gun, ticks_per_second, input, cycled, &mut out),
        };
        self.scoped_ticks = if self.scope_level > 0 {
            self.scoped_ticks.saturating_add(1)
        } else {
            0
        };
        self.zoom_ticks = self.zoom_ticks.saturating_add(1);
        self.settled = self.settled_scope(gun, ticks_per_second);
        match pull {
            Pull::Shoot => {
                self.magazine -= 1;
                self.armed = false;
                let rate = gun.shots_per_second * input.pace.fire_rate * self.rate_factor(gun);
                self.cooldown_ticks += ticks_per_second as f32 / rate - 1.0;
                let spread =
                    bloomed_deg(gun, stance, self.settled, self.recoil.shots) * input.pace.spread;
                let recoil = match &gun.scope {
                    Some(scope) if self.settled > 0 => gun.recoil.scaled(scope.recoil_factor),
                    Some(_) | None => gun.recoil,
                };
                let (shot, kick) = fire(&recoil, &mut self.recoil, aim, spread, rng);
                if gun.scope.as_ref().is_some_and(|scope| scope.breaks_on_shot) {
                    let level = self.scope_level;
                    self.unscope();
                    self.rescope = (level > 0).then_some(level);
                }
                out.shot = Some(shot);
                out.kick = kick;
            }
            Pull::Dry | Pull::Nothing => {
                out.dry_fire = pull == Pull::Dry;
                if cycled {
                    self.cooldown_ticks = 0.0;
                    out.kick = recover(&gun.recoil, &mut self.recoil, ticks_per_second);
                } else {
                    self.cooldown_ticks -= 1.0;
                }
            }
        }
        out
    }

    fn equip_step(&mut self, left: u32) {
        if left > 1 {
            self.phase = Phase::Equipping { left: left - 1 };
        } else {
            self.phase = Phase::Ready;
        }
    }

    fn reload_step(&mut self, gun: &GunNumbers, left: u32, total: u32, out: &mut GunTick) {
        if left > 1 {
            self.phase = Phase::Reloading {
                left: left - 1,
                total,
            };
            return;
        }
        let moved = (gun.magazine - self.magazine).min(self.reserve);
        self.magazine += moved;
        self.reserve -= moved;
        self.phase = Phase::Ready;
        out.reload_done = true;
    }

    fn ready_step(
        &mut self,
        gun: &GunNumbers,
        ticks_per_second: u32,
        input: GunInput,
        cycled: bool,
        out: &mut GunTick,
    ) -> Pull {
        if input.scope_pressed
            && let Some(scope) = &gun.scope
        {
            let level = if self.rescope.take().is_some() {
                0
            } else {
                self.scope_level
            };
            self.scope_to((level + 1) % (scope.zoom.len() + 1), gun, ticks_per_second);
        } else if cycled && let Some(level) = self.rescope.take() {
            self.scope_to(level, gun, ticks_per_second);
        }
        let wanted = input.reload_pressed || (self.magazine == 0 && cycled);
        if wanted && self.magazine < gun.magazine && self.reserve > 0 {
            let total = ticks(gun.reload_seconds / input.pace.reload, ticks_per_second);
            self.phase = Phase::Reloading { left: total, total };
            self.unscope();
            out.reload_started = true;
            return Pull::Nothing;
        }
        if !input.fire_held || !cycled {
            return Pull::Nothing;
        }
        if self.magazine == 0 {
            if self.armed {
                self.armed = false;
                return Pull::Dry;
            }
            return Pull::Nothing;
        }
        if gun.auto || self.armed {
            Pull::Shoot
        } else {
            Pull::Nothing
        }
    }
}
