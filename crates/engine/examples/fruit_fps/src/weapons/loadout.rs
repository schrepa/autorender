use super::gun::{Gun, GunInput, GunTick, GunView, Pace};
use super::numbers::{GunKind, WeaponNumbers, ticks};
use super::recoil::{Aim, Kick, Recoil, Shot, ShotRng, Stance};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Slot {
    Primary,
    Sidearm,
    Melee,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct LoadoutInput {
    pub fire_held: bool,
    pub reload_pressed: bool,
    pub scope_pressed: bool,
    pub switch_to: Option<Slot>,
    pub pace: Pace,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Fired {
    Shot { gun: GunKind, shot: Shot },
    Swing { aim: Aim, heavy: bool },
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct LoadoutTick {
    pub fired: Option<Fired>,
    pub active: GunTick,
    pub switched: bool,
    pub kick: Kick,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LoadoutView<'a> {
    pub slot: Slot,
    pub name: &'a str,
    pub magazine: Option<u32>,
    pub reserve: Option<u32>,
    pub reload_progress: Option<f32>,
    pub spread_deg: f32,
    pub scope_level: usize,
    pub ready: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Loadout {
    pub primary: Option<Gun>,
    pub sidearm: Gun,
    pub active: Slot,
    melee_wait: u32,
    melee_armed: bool,
    rng: ShotRng,
    pace: Pace,
}

impl Loadout {
    pub fn new(
        numbers: &WeaponNumbers,
        primary: Option<GunKind>,
        rng: ShotRng,
        ticks_per_second: u32,
    ) -> Self {
        let mut loadout = Self {
            primary: primary.map(|kind| Gun::new(kind, numbers)),
            sidearm: Gun::new(GunKind::Sidearm, numbers),
            active: Slot::Sidearm,
            melee_wait: 0,
            melee_armed: false,
            rng,
            pace: Pace::SAME,
        };
        let first = if loadout.primary.is_some() {
            Slot::Primary
        } else {
            Slot::Sidearm
        };
        loadout.switch(numbers, first, ticks_per_second);
        loadout
    }

    pub fn rng(&self) -> ShotRng {
        self.rng
    }

    pub fn active_gun(&self) -> Option<&Gun> {
        match self.active {
            Slot::Primary => self.primary.as_ref(),
            Slot::Sidearm => Some(&self.sidearm),
            Slot::Melee => None,
        }
    }

    pub fn recoil(&self) -> Recoil {
        let mut total = Recoil::default();
        for gun in self.guns() {
            let recoil = gun.recoil();
            total.shots += recoil.shots;
            total.pitch_radians += recoil.pitch_radians;
            total.yaw_radians += recoil.yaw_radians;
        }
        total
    }

    pub fn speed_factor(&self, numbers: &WeaponNumbers) -> f32 {
        self.active_gun()
            .map_or(numbers.melee.speed_factor, |gun| gun.speed_factor(numbers))
    }

    pub fn view_zoom(&self, numbers: &WeaponNumbers) -> f32 {
        self.active_gun().map_or(1.0, |gun| gun.view_zoom(numbers))
    }

    pub fn view<'a>(&self, numbers: &'a WeaponNumbers, stance: Stance) -> LoadoutView<'a> {
        match self.active_gun() {
            Some(gun) => {
                let GunView {
                    name,
                    magazine,
                    reserve,
                    reload_progress,
                    spread_deg,
                    scope_level,
                    ready,
                } = gun.view(numbers, stance);
                LoadoutView {
                    slot: self.active,
                    name,
                    magazine: Some(magazine),
                    reserve: Some(reserve),
                    reload_progress,
                    spread_deg: spread_deg * self.pace.spread,
                    scope_level,
                    ready,
                }
            }
            None => LoadoutView {
                slot: Slot::Melee,
                name: &numbers.melee.name,
                magazine: None,
                reserve: None,
                reload_progress: None,
                spread_deg: 0.0,
                scope_level: 0,
                ready: self.melee_wait == 0,
            },
        }
    }

    pub fn give_primary(&mut self, numbers: &WeaponNumbers, kind: GunKind) {
        self.primary = Some(Gun::new(kind, numbers));
    }

    pub fn give(&mut self, numbers: &WeaponNumbers, kind: GunKind, ticks_per_second: u32) {
        let slot = if kind == GunKind::Sidearm {
            self.sidearm = Gun::new(kind, numbers);
            Slot::Sidearm
        } else {
            self.primary = Some(Gun::new(kind, numbers));
            Slot::Primary
        };
        self.switch(numbers, slot, ticks_per_second);
    }

    pub fn pace(&self) -> Pace {
        self.pace
    }

    pub fn tick(
        &mut self,
        numbers: &WeaponNumbers,
        ticks_per_second: u32,
        input: LoadoutInput,
        aim: Aim,
        stance: Stance,
    ) -> LoadoutTick {
        let mut out = LoadoutTick::default();
        if let Some(slot) = input.switch_to
            && slot != self.active
            && (slot != Slot::Primary || self.primary.is_some())
        {
            self.switch(numbers, slot, ticks_per_second);
            out.switched = true;
        }
        self.pace = input.pace;
        let gun_input = GunInput {
            fire_held: input.fire_held,
            reload_pressed: input.reload_pressed,
            scope_pressed: input.scope_pressed,
            pace: input.pace,
        };
        let active = self.active;
        let rng = &mut self.rng;
        let mut kick = Kick::default();
        for (slot, gun) in [
            (Slot::Primary, self.primary.as_mut()),
            (Slot::Sidearm, Some(&mut self.sidearm)),
        ] {
            let Some(gun) = gun else {
                continue;
            };
            let given = if slot == active {
                gun_input
            } else {
                GunInput {
                    pace: input.pace,
                    ..GunInput::default()
                }
            };
            let ticked = gun.tick(numbers, ticks_per_second, given, aim, stance, rng);
            kick.pitch_radians += ticked.kick.pitch_radians;
            kick.yaw_radians += ticked.kick.yaw_radians;
            if slot == active {
                out.fired = ticked.shot.map(|shot| Fired::Shot {
                    gun: gun.kind(),
                    shot,
                });
                out.active = ticked;
            }
        }
        if active == Slot::Melee {
            out.fired = self.swing(numbers, ticks_per_second, input, aim);
        }
        out.kick = kick;
        out
    }

    fn swing(
        &mut self,
        numbers: &WeaponNumbers,
        ticks_per_second: u32,
        input: LoadoutInput,
        aim: Aim,
    ) -> Option<Fired> {
        if !input.fire_held {
            self.melee_armed = true;
        }
        if self.melee_wait > 0 {
            self.melee_wait -= 1;
            return None;
        }
        let melee = &numbers.melee;
        let (heavy, seconds) = if input.scope_pressed {
            (true, melee.heavy_swing_seconds)
        } else if input.fire_held && self.melee_armed {
            (false, melee.swing_seconds)
        } else {
            return None;
        };
        self.melee_armed = false;
        self.melee_wait = ticks(seconds, ticks_per_second);
        Some(Fired::Swing { aim, heavy })
    }

    pub fn drop_primary(&mut self, numbers: &WeaponNumbers, ticks_per_second: u32) -> Option<Gun> {
        let dropped = self.primary.take()?;
        if self.active == Slot::Primary {
            self.switch(numbers, Slot::Sidearm, ticks_per_second);
        }
        Some(dropped)
    }

    pub fn pick_up(
        &mut self,
        numbers: &WeaponNumbers,
        gun: Gun,
        ticks_per_second: u32,
    ) -> Option<Gun> {
        let (slot, held) = if gun.kind() == GunKind::Sidearm {
            (
                Slot::Sidearm,
                Some(std::mem::replace(&mut self.sidearm, gun)),
            )
        } else {
            (Slot::Primary, self.primary.replace(gun))
        };
        self.switch(numbers, slot, ticks_per_second);
        held
    }

    fn switch(&mut self, numbers: &WeaponNumbers, slot: Slot, ticks_per_second: u32) {
        if let Some(gun) = self.primary.as_mut() {
            gun.put_away();
        }
        self.sidearm.put_away();
        self.active = slot;
        self.melee_armed = false;
        match slot {
            Slot::Primary => {
                if let Some(gun) = self.primary.as_mut() {
                    gun.take_out(numbers, ticks_per_second);
                }
            }
            Slot::Sidearm => self.sidearm.take_out(numbers, ticks_per_second),
            Slot::Melee => {
                self.melee_wait = ticks(numbers.melee.equip_seconds, ticks_per_second);
            }
        }
    }

    fn guns(&self) -> impl Iterator<Item = &Gun> {
        self.primary.iter().chain(std::iter::once(&self.sidearm))
    }
}

pub fn actor_rng(match_seed: u64, slot: u32) -> ShotRng {
    let mut mixed =
        match_seed ^ (u64::from(slot).wrapping_add(1)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    mixed = (mixed ^ (mixed >> 33)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    mixed ^= mixed >> 33;
    ShotRng::new(mixed)
}
