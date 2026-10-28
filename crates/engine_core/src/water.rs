use bevy_ecs::prelude::*;
use glam::{Mat3, Vec3};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum WaterKind {
    River,
    Lake,
    Fall,
    Sea,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct WaterSample {
    pub kind: WaterKind,
    pub surface: f32,
    pub depth: f32,
    pub velocity: [f32; 3],
    pub foam: f32,
}

pub type WaterSource = fn(&World, Vec3) -> Option<WaterSample>;

#[derive(Resource, Default)]
pub struct WaterSources {
    sources: Vec<(&'static str, WaterSource)>,
}

impl WaterSources {
    pub fn register(&mut self, name: &'static str, source: WaterSource) {
        match self.sources.iter_mut().find(|(held, _)| *held == name) {
            Some(entry) => entry.1 = source,
            None => self.sources.push((name, source)),
        }
    }

    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.sources.iter().map(|(name, _)| *name)
    }

    pub fn sample_in(world: &World, point: Vec3) -> Option<WaterSample> {
        let sources = world.get_resource::<Self>()?;
        sources
            .sources
            .iter()
            .filter_map(|(_, source)| source(world, point))
            .max_by(|a, b| a.surface.total_cmp(&b.surface))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterObstacle {
    pub center: Vec3,
    pub axes: Mat3,
    pub round: bool,
    pub velocity: Vec3,
}

impl WaterObstacle {
    pub fn reach(&self) -> Vec3 {
        let rows = self.axes.transpose();
        let along = |row: Vec3| {
            if self.round {
                row.length()
            } else {
                row.abs().element_sum()
            }
        };
        Vec3::new(along(rows.x_axis), along(rows.y_axis), along(rows.z_axis))
    }
}

#[derive(Resource, Default, Debug)]
pub struct WaterObstacles {
    pub obstacles: Vec<WaterObstacle>,
}
