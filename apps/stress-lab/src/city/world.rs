//! The wild city as a Stress Lab world: its traffic and machines move while
//! anything moves, under golden hour, seen across the river.

use engine::collections::HashMap;

use engine::prelude::*;

use super::{PER_BLOCK, populate};
use crate::kit::motion::add_systems;
use crate::kit::plan::{PERIOD, budget, river_row};
use crate::kit::{StressConfig, StressFile, StressWorld, Workload};

pub(crate) struct CityWorld;

impl StressWorld for CityWorld {
    fn name(&self) -> &'static str {
        "city"
    }

    fn label(&self) -> &'static str {
        "Wild city"
    }

    fn look(&self) -> &'static str {
        "golden"
    }

    fn populate(
        &self,
        world: &mut World,
        config: &StressConfig,
        _files: &HashMap<String, StressFile>,
    ) -> engine::Result<(u32, (Vec3, Vec3))> {
        let city = populate(world, config)?;
        world.insert_resource(Workload {
            transform_nodes: city.transform_nodes,
            moving_nodes: city.moving_nodes,
            sensor_nodes: city.sensor_nodes,
            probes: city.probes,
            overlap_hits: Vec::with_capacity(if config.queries_per_tick > 0 { 128 } else { 0 }),
        });
        Ok((city.moving_nodes, city.bounds))
    }

    fn add_systems(&self, app: &mut App, moving_nodes: u32) {
        if moving_nodes > 0 {
            add_systems(app);
        }
    }

    fn camera(&self, _world: &World, config: &StressConfig, view: &str) -> Option<Camera> {
        // Across the river toward the city's core.
        (view == "close").then(|| {
            let (position, target) = close_view(config);
            Camera {
                position,
                target,
                far: 5000.0,
                ..Camera::default()
            }
        })
    }
}

/// A street-level view across the river toward the core, or from an
/// avenue when the city has no river.
fn close_view(config: &StressConfig) -> (Vec3, Vec3) {
    let (_, _, radius) = budget(config, PER_BLOCK);
    match river_row(radius) {
        Some(row) => (
            Vec3::new(-PERIOD * 0.5, 14.0, row as f32 * PERIOD + PERIOD * 0.5),
            Vec3::new(PERIOD * 0.5, 55.0, 0.0),
        ),
        None => (Vec3::new(0.0, 16.0, 110.0), Vec3::new(0.0, 30.0, 0.0)),
    }
}
