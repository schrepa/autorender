use glam::Vec3;

use crate::abilities::{self, AbilitySpec, Aim, Delivery};
use crate::bots::{Places, World};
use crate::play::paths::Paths;
use crate::play::powers::Powers;
use crate::play::world::Arena;
use crate::weapons::forward;

pub(super) struct Thrower<'a> {
    pub actor: abilities::Actor,
    pub specs: &'a [AbilitySpec; 4],
    pub powers: &'a Powers,
}

pub(super) struct BotWorld<'a> {
    pub arena: &'a Arena,
    pub paths: &'a Paths,
    pub team: usize,
    pub places: &'a Places,
    pub thrower: Option<Thrower<'a>>,
}

impl World for BotWorld<'_> {
    fn sees(&self, from: Vec3, to: Vec3) -> bool {
        self.arena.sees(from, to)
    }

    fn path(&self, from: Vec3, to: Vec3, corners: &mut Vec<Vec3>) -> bool {
        self.paths.path_into(from, to, Some(self.team), corners)
    }

    fn cover_from(&self, threat: Vec3, near: Vec3, max_distance: f32) -> Option<Vec3> {
        self.paths
            .best_cover(self.arena, threat, near, max_distance)
    }

    fn open(&self, at: Vec3) -> bool {
        self.paths.nearest(at, 0.0).is_some()
    }

    fn places(&self) -> &Places {
        self.places
    }

    fn throw(&self, slot: usize, origin: Vec3, yaw: f32, pitch: f32) -> Option<Vec3> {
        let thrower = self.thrower.as_ref()?;
        let spec = thrower.specs.get(slot)?;
        if !matches!(spec.delivery, Delivery::Throw(_)) {
            return None;
        }
        let direction = forward(yaw, pitch);
        let caster = abilities::Actor {
            eye: origin,
            view: direction,
            ..thrower.actor
        };
        thrower
            .powers
            .landing(self.arena, &caster, spec, Aim { origin, direction })
            .filter(|preview| preview.valid)
            .map(|preview| preview.at)
    }
}
