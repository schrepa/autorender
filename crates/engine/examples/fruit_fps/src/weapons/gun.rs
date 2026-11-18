use super::numbers::{GunKind, GunNumbers, WeaponNumbers, ticks};
use super::recoil::{Aim, Kick, Recoil, Shot, ShotRng, Stance, fire, recover, spread_deg};

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
    pub became_ready: bool,
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

    pub fn is_ready(&self) -> bool {
        self.phase == Phase::Ready
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
        self.scope_level = 0;
        self.scoped_ticks = 0;
        self.settled = 0;
        self.armed = false;
    }

    pub fn put_away(&mut self) {
        self.phase = Phase::Holstered;
        self.scope_level = 0;
        self.scoped_ticks = 0;
        self.settled = 0;
    }

    pub fn speed_factor(&self, numbers: &WeaponNumbers) -> f32 {
        let gun = numbers.gun(self.kind);
        match &gun.scope {
            Some(scope) if self.scope_level > 0 => gun.run_speed_factor * scope.aim_speed_factor,
            Some(_) | None => gun.run_speed_factor,
        }
    }

    pub fn view_zoom(&self, numbers: &WeaponNumbers) -> f32 {
        let gun = numbers.gun(self.kind);
        gun.scope
            .as_ref()
            .and_then(|scope| scope.zoom.get(self.scope_level.checked_sub(1)?))
            .copied()
            .unwrap_or(1.0)
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
            spread_deg: spread_deg(gun, stance, self.settled),
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
        rng: &mut ShotRng,
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
                self.equip_step(left, &mut out);
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
        self.settled = self.settled_scope(gun, ticks_per_second);
        match pull {
            Pull::Shoot => {
                self.magazine -= 1;
                self.armed = false;
                let rate = gun.shots_per_second * input.pace.fire_rate * self.rate_factor(gun);
                self.cooldown_ticks += ticks_per_second as f32 / rate - 1.0;
                let spread = spread_deg(gun, stance, self.settled) * input.pace.spread;
                let (shot, kick) = fire(&gun.recoil, &mut self.recoil, aim, spread, rng);
                if gun.scope.as_ref().is_some_and(|scope| scope.breaks_on_shot) {
                    self.scope_level = 0;
                    self.scoped_ticks = 0;
                    self.settled = 0;
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

    fn equip_step(&mut self, left: u32, out: &mut GunTick) {
        if left > 1 {
            self.phase = Phase::Equipping { left: left - 1 };
        } else {
            self.phase = Phase::Ready;
            out.became_ready = true;
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
            self.scope_level = (self.scope_level + 1) % (scope.zoom.len() + 1);
        }
        if input.reload_pressed && self.magazine < gun.magazine && self.reserve > 0 {
            let total = ticks(gun.reload_seconds / input.pace.reload, ticks_per_second);
            self.phase = Phase::Reloading { left: total, total };
            self.scope_level = 0;
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

pub fn scoped_fov_radians(base_fov_radians: f32, zoom: f32) -> f32 {
    2.0 * ((base_fov_radians * 0.5).tan() / zoom).atan()
}
