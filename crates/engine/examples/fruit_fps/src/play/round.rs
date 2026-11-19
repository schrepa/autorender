use engine::prelude::*;

use crate::rules::{Match, Phase, Side, Slot, View};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seat {
    pub entity: Option<Entity>,
    pub slot: Slot,
    pub level: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum DeviceSpot {
    #[default]
    Nowhere,
    Carried {
        slot: usize,
        entity: Option<Entity>,
    },
    Dropped {
        at: Vec3,
    },
    Planted {
        site: usize,
        at: Vec3,
        yaw: f32,
        fuse_ticks_left: u32,
    },
    Exploded {
        site: usize,
        at: Vec3,
        yaw: f32,
    },
    Defused {
        site: usize,
        at: Vec3,
        yaw: f32,
    },
}

impl DeviceSpot {
    pub fn at(&self) -> Option<Vec3> {
        match *self {
            DeviceSpot::Nowhere | DeviceSpot::Carried { .. } => None,
            DeviceSpot::Dropped { at }
            | DeviceSpot::Planted { at, .. }
            | DeviceSpot::Exploded { at, .. }
            | DeviceSpot::Defused { at, .. } => Some(at),
        }
    }
}

#[derive(Resource, Default)]
pub struct Round {
    pub game: Option<Match>,
    pub seed: u64,
    pub matches: u32,
    pub seats: Vec<Seat>,
    pub buy_lock: bool,
    pub device: DeviceSpot,
    pub restart_ticks: u32,
}

impl Round {
    pub fn view(&self) -> Option<View<'_>> {
        self.game.as_ref().map(Match::view)
    }

    pub fn phase(&self) -> Option<Phase> {
        self.game.as_ref().map(|game| game.view().phase)
    }

    pub fn side_of_slot(&self, slot: usize) -> Option<Side> {
        let game = self.game.as_ref()?;
        let team = game.view().players.get(slot)?.team;
        Some(game.side_of(team))
    }
}
