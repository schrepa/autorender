use glam::Vec3;

use crate::bots::{Places, World};
use crate::play::paths::Paths;
use crate::play::world::Arena;

pub(super) struct BotWorld<'a> {
    pub arena: &'a Arena,
    pub paths: &'a Paths,
    pub team: usize,
    pub places: &'a Places,
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
}
