use glam::{Mat3, Vec3};

use crate::abilities::{self, AbilityId, AbilitySpec, Aim, Delivery, ThrowSurface, WorldHit};
use crate::bots::{Places, World};
use crate::play::paths::Paths;
use crate::play::world::Arena;
use crate::solids::{Hit, Kind};
use crate::weapons::forward;

pub(super) struct Thrower<'a> {
    pub actor: abilities::Actor,
    pub specs: &'a [AbilitySpec; 4],
    pub surfaces: &'a [ThrowSurface],
}

pub(super) struct BotWorld<'a> {
    pub arena: &'a Arena,
    pub paths: &'a Paths,
    pub team: usize,
    pub places: &'a Places,
    pub thrower: Option<Thrower<'a>>,
}

struct Flight<'a> {
    arena: &'a Arena,
    surfaces: &'a [ThrowSurface],
}

impl Flight<'_> {
    fn hit(&self, hit: Hit) -> WorldHit {
        let bounce = self
            .surfaces
            .iter()
            .find(|surface| Kind::named(&surface.kind) == hit.kind)
            .map_or(1.0, |surface| surface.bounce);
        WorldHit {
            distance: hit.distance,
            normal: hit.normal,
            bounce,
        }
    }
}

impl abilities::World for Flight<'_> {
    fn ray(&self, origin: Vec3, direction: Vec3, max: f32) -> Option<WorldHit> {
        self.arena
            .ray(origin, direction, max)
            .map(|hit| self.hit(hit))
    }

    fn sphere_sweep(&self, centre: Vec3, radius: f32, motion: Vec3) -> Option<WorldHit> {
        self.arena
            .solids()
            .sphere_sweep(centre, radius, motion)
            .map(|hit| self.hit(hit))
    }

    fn add_box(
        &mut self,
        _id: u32,
        _ability: AbilityId,
        _centre: Vec3,
        _half_extents: Vec3,
        _axes: Mat3,
    ) {
    }

    fn remove_box(&mut self, _id: u32) {}
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
        let aim = Aim { origin, direction };
        let flight = Flight {
            arena: self.arena,
            surfaces: thrower.surfaces,
        };
        abilities::preview(&flight, &caster, spec, aim)
            .filter(|preview| preview.valid)
            .map(|preview| preview.at)
    }
}
