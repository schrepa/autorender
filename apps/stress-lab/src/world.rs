//! The Stress Lab's worlds. Each builds its content, adds the systems that
//! move it, and names its look, cameras and files; [`StressScene::world`]
//! is the one place a configuration's scene picks its world.

// The world registry names every world.
#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use engine::prelude::*;

use crate::{StressConfig, StressFile, StressScene, city, dense, field, highlands, metro};

/// One of the Stress Lab's worlds.
pub(crate) trait StressWorld: Sync {
    /// Its name in configurations.
    fn name(&self) -> &'static str;

    /// Its name in menus.
    fn label(&self) -> &'static str;

    /// The lighting preset its own look (`scene`) is.
    fn look(&self) -> &'static str;

    /// Builds its content into `world` from `config` and the files it reads:
    /// how many nodes move, and the bounds its shadows keep in one map.
    fn populate(
        &self,
        world: &mut World,
        config: &StressConfig,
        files: &HashMap<String, StressFile>,
    ) -> engine::Result<(u32, (Vec3, Vec3))>;

    /// Changes what every world's environment holds, once populated.
    fn environment(&self, _world: &mut World) {}

    /// Its shadows: the whole of `bounds` in one map, seen from every view.
    fn shadows(&self, bounds: (Vec3, Vec3)) -> Shadows {
        Shadows {
            bounds: Some(bounds),
            distance: 1600.0,
            near_distance: 90.0,
            resolution: 4096,
            ..Shadows::default()
        }
    }

    /// Its screen-space reflections.
    fn reflections(&self) -> Reflections {
        Reflections::default()
    }

    /// Whether glass in it refracts what is behind.
    fn refracts(&self) -> bool {
        false
    }

    /// Adds the systems its content runs, `moving_nodes` of its nodes
    /// moving.
    fn add_systems(&self, app: &mut App, moving_nodes: u32);

    /// Its camera at view `view` of `world`, where it frames that view its
    /// own way; every world's framing otherwise.
    fn camera(&self, _world: &World, _config: &StressConfig, _view: &str) -> Option<Camera> {
        None
    }

    /// How fast the fly camera crosses it, `span` across.
    fn camera_speed(&self, span: f32) -> f32 {
        math::clamp(span * 0.3, 22.0, 220.0)
    }

    /// The files it reads from the folder `assets` natively, as the page
    /// fetches them; `rocks` adds optional scanned rock maps where present.
    #[cfg(not(target_arch = "wasm32"))]
    fn files(&self, _assets: &Path, _rocks: bool) -> engine::Result<HashMap<String, StressFile>> {
        Ok(HashMap::default())
    }
}

impl StressScene {
    /// The world it builds.
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
