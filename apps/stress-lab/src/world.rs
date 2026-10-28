#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use engine::prelude::*;

use crate::{StressConfig, StressFile, StressScene, city, dense, field, highlands, metro};

pub(crate) trait StressWorld: Sync {
    fn name(&self) -> &'static str;

    fn label(&self) -> &'static str;

    fn look(&self) -> &'static str;

    fn populate(
        &self,
        world: &mut World,
        config: &StressConfig,
        files: &HashMap<String, StressFile>,
    ) -> engine::Result<u32>;

    fn environment(&self, _world: &mut World) {}

    fn reflections(&self) -> Reflections {
        Reflections::default()
    }

    fn refracts(&self) -> bool {
        false
    }

    fn add_systems(&self, app: &mut App, moving_nodes: u32);

    fn camera(&self, _world: &World, _config: &StressConfig, _view: &str) -> Option<Camera> {
        None
    }

    fn camera_speed(&self, span: f32) -> f32 {
        math::clamp(span * 0.3, 22.0, 220.0)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn files(&self, _assets: &Path, _rocks: bool) -> engine::Result<HashMap<String, StressFile>> {
        Ok(HashMap::default())
    }
}

impl StressScene {
    pub(crate) fn world(self) -> &'static dyn StressWorld {
        match self {
            Self::Field => &field::FieldWorld,
            Self::City => &city::CityWorld,
            Self::Metro => &metro::MetroWorld,
            Self::Dense => &dense::DenseWorld,
            Self::Highlands => &highlands::HighlandsWorld,
        }
    }
}
