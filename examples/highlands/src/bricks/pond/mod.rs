#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/bricks/pond.rs"));

mod part;
mod smokes;
mod water;

use engine_core::{
    Brick, Bricks, ComponentKind, Error, ErrorCode, GpuStruct, Runs, SceneComponents, Schedule,
    Smoke, Source, fields,
};
use engine_render::RenderBricks;
use glam::Vec2;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const GENERATED: &[&str] = &[part::WGSL, <part::PondShape as GpuStruct>::WGSL];

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
            .join::<SceneComponents>(
                ComponentKind::of::<Pond>("pond")
                    .check(check)
                    .schema(schema)
                    .example(example),
                None,
            )
            .systems(Schedule::ViewSync, water::publish_ponds)
            .generated(GENERATED)
            .part::<part::PondGpu>();
    }
}

fields! {
    #[derive(
        bevy_ecs::prelude::Component,
        Clone,
        Copy,
        Debug,
        PartialEq,
        Serialize,
        Deserialize,
        schemars::JsonSchema,
    )]
    #[serde(deny_unknown_fields)]
    pub struct Pond {
        pub center: [f32; 2] = [0.0; 2] => finite,
        pub size: [f32; 2] = [1.0; 2] => above(0.0),
        pub surface: f32 = 0.0 => finite,
    }
}

fn check(pond: &Pond, path: &str) -> Result<(), Error> {
    pond.validate()
        .map_err(|invalid| invalid.error(ErrorCode::InvalidValue, path))
}

fn schema() -> Value {
    serde_json::to_value(schemars::schema_for!(Pond)).unwrap_or(Value::Null)
}

fn example() -> Value {
    json!({"center": [0.0, 0.0], "size": [16.0, 16.0], "surface": 0.0})
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
