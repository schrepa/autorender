use bevy_ecs::prelude::{Resource, World};
use engine_settings::{InvalidSetting, settings};
use glam::Vec2;
use serde::{Deserialize, Serialize};

use engine_core::math::sin;
use engine_core::{Brick, Bricks, EnvironmentSection, RenderSnapshot};

pub const GUST_SLOW_SHARE: f32 = 0.6;
pub const GUST_SLOW_RATE: f32 = 0.9;
pub const GUST_SLOW_TRAVEL: f32 = 0.05;
pub const GUST_QUICK_SHARE: f32 = 0.4;
pub const GUST_QUICK_RATE: f32 = 2.3;
pub const GUST_QUICK_TRAVEL: f32 = 0.17;
pub const GUST_QUICK_ACROSS: f32 = 0.31;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneWind")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Wind {
        pub velocity: [f32; 2] = [2.5, 1.0] => range(-40.0, 40.0),
        pub gusts: f32 = 0.45 => range(0.0, 1.0),
    }
}

impl Wind {
    pub const STILL: Self = Self {
        velocity: [0.0; 2],
        gusts: 0.0,
    };

    pub fn off() -> Self {
        Self::STILL
    }

    pub fn at(&self, x: f32, z: f32, seconds: f32) -> Vec2 {
        let velocity = Vec2::from_array(self.velocity);
        let speed = velocity.length();
        if speed < 1.0e-4 {
            return Vec2::ZERO;
        }
        let toward = velocity / speed;
        let travel = x * toward.x + z * toward.y;
        let gust = 1.0
            + self.gusts
                * (GUST_SLOW_SHARE * sin(seconds * GUST_SLOW_RATE - travel * GUST_SLOW_TRAVEL)
                    + GUST_QUICK_SHARE
                        * sin(seconds * GUST_QUICK_RATE - travel * GUST_QUICK_TRAVEL
                            + x * GUST_QUICK_ACROSS));
        velocity * gust
    }
}

impl EnvironmentSection for Wind {
    const KEY: &'static str = "wind";
    const NAME: &'static str = "Wind";
    const OPT_IN: bool = false;

    fn section_off() -> Self {
        Self::off()
    }

    fn section_check(&self) -> Result<(), InvalidSetting> {
        self.validate()
    }

    fn section_sanitized(self) -> Self {
        self.sanitized()
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(Wind)
    }
}

impl Brick for Wind {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.settings::<Wind>().extract(extract_wind);
    }
}

fn extract_wind(world: &World, snapshot: &mut RenderSnapshot) {
    let wind = world
        .get_resource::<Wind>()
        .copied()
        .unwrap_or_else(Wind::off)
        .sanitized();
    snapshot.wind = [wind.velocity[0], wind.velocity[1], wind.gusts, 0.0];
}
