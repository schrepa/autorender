use engine::prelude::{Component, Resource};
use glam::Vec3;

use crate::play::events::arms::Weapon;
use crate::weapons::{Gun, GunKind, Loadout, Stance};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Held {
    Gun(GunKind),
    Melee,
    Device,
}

impl Held {
    pub fn key(self) -> &'static str {
        match self {
            Held::Gun(kind) => Weapon::Gun(kind).key(),
            Held::Melee => Weapon::Melee.key(),
            Held::Device => "device",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandAction {
    Equip,
    Swing { heavy: bool, step: u8 },
    Plant,
    Defuse,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hand {
    pub held: Held,
    pub action: Option<HandAction>,
    pub progress: f32,
    pub seconds: f32,
}

#[derive(Component, Clone, Debug, PartialEq)]
pub struct Arms {
    pub loadout: Loadout,
    pub kick_pitch_radians: f32,
    pub kick_yaw_radians: f32,
    pub stance: Stance,
    pub landing_ticks: u32,
    pub tag_ticks: u32,
    pub tag_total: u32,
    pub tag_slow: f32,
    pub use_held: bool,
    pub pickable: Option<u32>,
    pub run_factor: f32,
    pub hand: Hand,
    pub device_out: bool,
    pub device_equip_left: u32,
    pub busy: bool,
    pub weapon_down: bool,
    pub zoom: f32,
    pub scoped: bool,
}

impl Arms {
    pub fn new(loadout: Loadout, run_factor: f32) -> Self {
        let held = loadout
            .active_gun()
            .map_or(Held::Melee, |gun| Held::Gun(gun.kind()));
        Self {
            loadout,
            kick_pitch_radians: 0.0,
            kick_yaw_radians: 0.0,
            stance: Stance::Stand,
            landing_ticks: 0,
            tag_ticks: 0,
            tag_total: 0,
            tag_slow: 0.0,
            use_held: false,
            pickable: None,
            run_factor,
            hand: Hand {
                held,
                action: None,
                progress: 0.0,
                seconds: 0.0,
            },
            device_out: false,
            device_equip_left: 0,
            busy: false,
            weapon_down: false,
            zoom: 1.0,
            scoped: false,
        }
    }

    pub fn tag_factor(&self) -> f32 {
        if self.tag_total == 0 {
            return 1.0;
        }
        1.0 - self.tag_slow * self.tag_ticks as f32 / self.tag_total as f32
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dropped {
    pub id: u32,
    pub position: Vec3,
    pub thrown_from: Option<Vec3>,
    pub by: Option<u32>,
    pub weapon: GunKind,
    pub magazine: u32,
    pub reserve: u32,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct Drops {
    pub items: Vec<Dropped>,
    next_id: u32,
}

impl Drops {
    pub fn push(&mut self, position: Vec3, gun: &Gun) -> u32 {
        self.put(position, None, None, gun)
    }

    pub fn throw(&mut self, (from, by): (Vec3, u32), to: Vec3, gun: &Gun) -> u32 {
        self.put(to, Some(from), Some(by), gun)
    }

    fn put(
        &mut self,
        position: Vec3,
        thrown_from: Option<Vec3>,
        by: Option<u32>,
        gun: &Gun,
    ) -> u32 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.items.push(Dropped {
            id,
            position,
            thrown_from,
            by,
            weapon: gun.kind(),
            magazine: gun.magazine(),
            reserve: gun.reserve(),
        });
        id
    }

    pub fn take(&mut self, id: u32) -> Option<Dropped> {
        let at = self.items.iter().position(|item| item.id == id)?;
        Some(self.items.remove(at))
    }

    pub fn get(&self, id: u32) -> Option<&Dropped> {
        self.items.iter().find(|item| item.id == id)
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }
}
