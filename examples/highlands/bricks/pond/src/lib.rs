#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod part;
mod smokes;
mod water;

use engine_core::{Brick, Bricks, ComponentKind, Runs, SceneComponents, Schedule, Smoke, Source};
use engine_render::RenderBricks;
use glam::Vec2;
use serde::{Deserialize, Serialize};

pub struct PondBrick;

impl Brick for PondBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[Smoke {
        name: "highlands_pond",
        runs: Runs::Gpu,
        run: smokes::run,
    }];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(ComponentKind::of::<Pond>("pond"), None)
            .systems(Schedule::ViewSync, water::publish_ponds)
            .part::<part::PondGpu>();
    }
}

#[derive(bevy_ecs::prelude::Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "PondFile")]
pub struct Pond {
    pub center: [f32; 2],
    pub size: [f32; 2],
    pub surface: f32,
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PondFile {
    center: [f32; 2],
    size: [f32; 2],
    surface: f32,
}

impl Default for PondFile {
    fn default() -> Self {
        Self {
            center: [0.0; 2],
            size: [1.0; 2],
            surface: 0.0,
        }
    }
}

impl TryFrom<PondFile> for Pond {
    type Error = String;

    fn try_from(file: PondFile) -> Result<Self, String> {
        if !file.center.iter().all(|value| value.is_finite()) {
            return Err(format!("$.center: finite, got {:?}", file.center));
        }
        if !file.size.iter().all(|value| *value > 0.0) {
            return Err(format!("$.size: above 0, got {:?}", file.size));
        }
        if !file.surface.is_finite() {
            return Err(format!("$.surface: finite, got {}", file.surface));
        }
        Ok(Self {
            center: file.center,
            size: file.size,
            surface: file.surface,
        })
    }
}

impl Pond {
    pub fn low(&self) -> Vec2 {
        Vec2::from(self.center) - 0.5 * Vec2::from(self.size)
    }

    pub fn high(&self) -> Vec2 {
        Vec2::from(self.center) + 0.5 * Vec2::from(self.size)
    }

    pub fn holds(&self, x: f32, z: f32) -> bool {
        let (low, high) = (self.low(), self.high());
        x >= low.x && x <= high.x && z >= low.y && z <= high.y
    }
}
