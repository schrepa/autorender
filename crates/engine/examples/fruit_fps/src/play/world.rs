use engine::prelude::Resource;
use glam::{Quat, Vec3};

use crate::rules::Side;
use crate::solids::{Dropped, Hit, Solids};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spawn {
    pub position: Vec3,
    pub yaw_radians: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Zone {
    SiteA,
    SiteB,
    BuyAttack,
    BuyDefend,
}

impl Zone {
    pub const ALL: [Zone; 4] = [Zone::SiteA, Zone::SiteB, Zone::BuyAttack, Zone::BuyDefend];
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoneBox {
    pub zone: Zone,
    pub centre: Vec3,
    pub rotation: Quat,
    pub half: Vec3,
}

impl ZoneBox {
    pub fn contains(&self, point: Vec3) -> bool {
        let local = self.rotation.inverse() * (point - self.centre);
        local.abs().cmple(self.half).all()
    }
}

#[derive(Clone, Debug, Default)]
pub struct Markers {
    pub attack: Vec<Spawn>,
    pub defend: Vec<Spawn>,
    pub zones: Vec<ZoneBox>,
}

#[derive(Resource, Clone, Debug)]
pub struct Arena {
    solids: Solids,
    markers: Markers,
    ready: bool,
    revision: u64,
}

impl Default for Arena {
    fn default() -> Self {
        Arena {
            solids: Solids::build(&[], &[], &[]),
            markers: Markers::default(),
            ready: false,
            revision: 0,
        }
    }
}

impl Arena {
    pub fn adopt(&mut self, solids: Solids, markers: Markers) {
        self.solids = solids;
        self.markers = markers;
        self.ready = true;
        self.revision += 1;
    }

    pub fn ready(&self) -> bool {
        self.ready
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn solids(&self) -> &Solids {
        &self.solids
    }

    pub fn solids_mut(&mut self) -> &mut Solids {
        &mut self.solids
    }

    pub fn dropped(&self) -> Dropped {
        self.solids.dropped()
    }

    pub fn ray(&self, origin: Vec3, direction: Vec3, max: f32) -> Option<Hit> {
        self.solids.ray(origin, direction, max)
    }

    pub fn sees(&self, from: Vec3, to: Vec3) -> bool {
        !self.solids.blocked(from, to)
    }

    pub fn ground_under(&self, point: Vec3, max_drop: f32) -> Option<Hit> {
        self.solids.ground_under(point, max_drop)
    }

    pub fn spawns(&self, side: Side) -> &[Spawn] {
        match side {
            Side::Attack => &self.markers.attack,
            Side::Defend => &self.markers.defend,
        }
    }

    pub fn zone(&self, zone: Zone, point: Vec3) -> bool {
        self.markers
            .zones
            .iter()
            .any(|held| held.zone == zone && held.contains(point))
    }

    pub fn zones_at(&self, point: Vec3) -> Vec<Zone> {
        Zone::ALL
            .into_iter()
            .filter(|&zone| self.zone(zone, point))
            .collect()
    }

    pub fn site(&self, point: Vec3) -> Option<usize> {
        [Zone::SiteA, Zone::SiteB]
            .into_iter()
            .position(|zone| self.zone(zone, point))
    }
}
