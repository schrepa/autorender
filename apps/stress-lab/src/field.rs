//! The tower field: a grid of towers bobbing and turning in place, the
//! original workload.

use engine::collections::HashMap;

use engine::prelude::*;

use crate::kit::field::{animate, field};
use crate::kit::motion::Animation;
use crate::kit::{StressConfig, StressFile, StressWorld};

pub(crate) struct FieldWorld;

impl StressWorld for FieldWorld {
    fn name(&self) -> &'static str {
        "field"
    }

    fn label(&self) -> &'static str {
        "Tower field"
    }

    fn look(&self) -> &'static str {
        "night"
    }

    fn populate(
        &self,
        world: &mut World,
        config: &StressConfig,
        _files: &HashMap<String, StressFile>,
    ) -> engine::Result<u32> {
        field(world, config, 0.0)
    }

    fn add_systems(&self, app: &mut App, moving_nodes: u32) {
        if moving_nodes > 0 {
            app.add_systems(FixedSimulation, animate.in_set(Animation));
        }
    }
}
