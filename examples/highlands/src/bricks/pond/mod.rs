#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/bricks/pond.rs"));

mod part;
mod water;

use engine_core::{
    Attributes, Brick, Bricks, ComponentKind, Error, ErrorCode, GpuStruct, KindDecl, KindSkip,
    SceneComponents, Schedule, Source, fields,
};
use engine_places_rendering::lighting::inputs;
use engine_places_rendering::reflections::TraceDepths;
use engine_render::{PointInput, RenderBricks, SurfaceKind};
use glam::Vec2;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const GENERATED: &[&str] = &[part::WGSL, <part::PondShape as GpuStruct>::WGSL];

const SETS: [PointInput; 4] = [
    inputs::ROUGHNESS,
    inputs::METALNESS,
    inputs::OCCLUSION,
    inputs::TRANSMISSION,
];

pub struct PondBrick;

impl Brick for PondBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<Pond>("pond")
                    .check(check)
                    .schema(schema),
                None,
            )
            .systems(Schedule::ViewSync, water::publish_ponds)
            .generated(GENERATED)
            .surface_kind(SurfaceKind { name: "pond" })
            .join::<TraceDepths>((), Some(shaders::pond::trace_depth::MODULE))
            .draw_kind(KindDecl {
                kind: part::POND_KIND,
                needs: Attributes::NONE,
                streams: &[],
                skips: &[
                    KindSkip {
                        pass: "shadow",
                        reason: "the pond surface casts no shadows",
                    },
                    KindSkip {
                        pass: "probe",
                        reason: "the pond surface stays out of probe captures",
                    },
                    KindSkip {
                        pass: "color",
                        reason: "the pond surface draws in its own pass, over the scene it covers",
                    },
                    KindSkip {
                        pass: "prepass",
                        reason: "the pond surface covers the scene behind it, so it fills no depth before it",
                    },
                    KindSkip {
                        pass: "blended",
                        reason: "the pond surface covers the scene behind it in its own pass",
                    },
                ],
            })
            .pass::<part::PondPass>()
            .draws::<part::PondGpu>(part::POND_KIND, part::PondGpu::draw);
        for input in SETS {
            bricks.point_set(input.set());
        }
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
