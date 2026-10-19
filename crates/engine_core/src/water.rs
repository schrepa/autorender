//! The world's water as everything that meets it reads it: one sample
//! over any point, whichever source holds water there. Terrains' rivers,
//! lakes and falls and the sea each register a source; what floats, wades,
//! drifts or sprays asks [`WaterSources::sample_in`] and gets the water whose
//! surface stands highest over the point, with how deep it is, how fast and
//! which way it runs and how much foam rides it. Drawing reads the same
//! sources' data, so what water does matches what it shows. The other way,
//! the solids standing in the world ([`WaterObstacles`]) hold water back
//! where they rest on a river's bed.

use bevy_ecs::prelude::*;
use glam::{Mat3, Vec3};
use serde::Serialize;

/// What kind of water a sample is: a river's, a lake's, a waterfall's jet or
/// the sea's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum WaterKind {
    River,
    Lake,
    Fall,
    Sea,
}

/// The water over a world point: its kind, its surface's world height, the
/// meters it stands over the ground or bed below the point, its velocity
/// (world meters a second; falls' carry their drop) and the foam riding it
/// (0 to 1).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct WaterSample {
    pub kind: WaterKind,
    pub surface: f32,
    pub depth: f32,
    pub velocity: [f32; 3],
    pub foam: f32,
}

/// A source of water: its name and how it answers for a world point,
/// `None` where it holds none there (or does not know yet).
pub type WaterSource = fn(&World, Vec3) -> Option<WaterSample>;

/// Every source of the world's water: features holding water register one
/// as they install. Without sources the world is dry.
#[derive(Resource, Default)]
pub struct WaterSources {
    sources: Vec<(&'static str, WaterSource)>,
}

impl WaterSources {
    /// Adds `source` under `name`, replacing one of that name.
    pub fn register(&mut self, name: &'static str, source: WaterSource) {
        match self.sources.iter_mut().find(|(held, _)| *held == name) {
            Some(entry) => entry.1 = source,
            None => self.sources.push((name, source)),
        }
    }

    /// The names of the sources, in the order they registered.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.sources.iter().map(|(name, _)| *name)
    }

    /// The water over world `point` in `world`: of the sources holding water
    /// there, the one whose surface stands highest.
    pub fn sample_in(world: &World, point: Vec3) -> Option<WaterSample> {
        let sources = world.get_resource::<Self>()?;
        sources
            .sources
            .iter()
            .filter_map(|(_, source)| source(world, point))
            .max_by(|a, b| a.surface.total_cmp(&b.surface))
    }
}

/// A solid standing in the world, as water meets it: its world center, the
/// axes mapping the unit cube (or ball, when `round`) to it, columns scaled
/// by its extents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterObstacle {
    pub center: Vec3,
    pub axes: Mat3,
    pub round: bool,
}

/// The solids that may stand in the world's water this fixed tick: physics
/// lists its bodies and the colliders near rivers, and terrains' rivers are
/// held back where one rests on a bed. Without it nothing holds them back.
#[derive(Resource, Default, Debug)]
pub struct WaterObstacles {
    pub obstacles: Vec<WaterObstacle>,
}
