//! The metro as a Stress Lab world: its billboards change pictures whether
//! or not anything moves, its glass refracts, and it is seen from the air
//! and across the river.

use engine::collections::HashMap;

use engine::prelude::*;

use super::{PER_BLOCK, add_systems, populate};
use crate::kit::plan::{PERIOD, budget, river_row};
use crate::kit::{StressConfig, StressFile, StressWorld, Workload};

pub(crate) struct MetroWorld;

impl StressWorld for MetroWorld {
    fn name(&self) -> &'static str {
        "metro"
    }

    fn label(&self) -> &'static str {
        "Metro"
    }

    fn look(&self) -> &'static str {
        "golden"
    }

    fn populate(
        &self,
        world: &mut World,
        config: &StressConfig,
        _files: &HashMap<String, StressFile>,
    ) -> Result<(u32, (Vec3, Vec3)), String> {
        let city = populate(world, config)?;
        world.insert_resource(Workload {
            transform_nodes: city.transform_nodes,
            moving_nodes: city.moving_nodes,
            sensor_nodes: city.sensor_nodes,
            probes: city.probes,
            overlap_hits: Vec::with_capacity(if config.queries_per_tick > 0 { 128 } else { 0 }),
        });
        // Glass refracts what is behind it.
        world.insert_resource(Refraction::default());
        Ok((city.moving_nodes, city.bounds))
    }

    fn refracts(&self) -> bool {
        true
    }

    fn add_systems(&self, app: &mut App, _moving_nodes: u32) {
        add_systems(app);
    }

    fn camera(&self, _world: &World, config: &StressConfig, view: &str) -> Option<Camera> {
        match view {
            "overview" => {
                let (position, target) = overview(config);
                Some(Camera {
                    position,
                    target,
                    far: 5000.0,
                    vertical_fov_radians: 55.0_f32.to_radians(),
                    ..Camera::default()
                })
            }
            "close" => {
                let (position, target) = close_view(config);
                Some(Camera {
                    position,
                    target,
                    far: 5000.0,
                    ..Camera::default()
                })
            }
            _ => None,
        }
    }
}

/// A street-level view across the river toward the core.
fn close_view(config: &StressConfig) -> (Vec3, Vec3) {
    let (_, _, radius) = budget(config, PER_BLOCK);
    match river_row(radius) {
        Some(row) => (
            Vec3::new(-PERIOD * 0.3, 9.0, row as f32 * PERIOD + PERIOD * 0.62),
            Vec3::new(PERIOD * 0.8, 40.0, 0.0),
        ),
        None => (Vec3::new(0.0, 12.0, 80.0), Vec3::new(0.0, 25.0, 0.0)),
    }
}

/// An aerial view of the whole metro.
fn overview(config: &StressConfig) -> (Vec3, Vec3) {
    let (_, _, radius) = budget(config, PER_BLOCK);
    let reach = (radius + 2.0) * PERIOD;
    (
        Vec3::new(reach * 0.55, reach * 0.62, reach * 1.05),
        Vec3::new(0.0, 10.0, 0.0),
    )
}
