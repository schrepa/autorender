use engine::prelude::{Component, Resource};
use glam::Vec3;

use crate::weapons::{Gun, GunKind, Loadout, Stance};

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
}

impl Arms {
    pub fn new(loadout: Loadout) -> Self {
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
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.items.push(Dropped {
            id,
            position,
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
