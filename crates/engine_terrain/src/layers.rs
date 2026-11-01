use std::sync::Arc;

use engine_codec::{Reader, record, tagged};
use engine_core::{Error, ErrorCode};
use engine_heightfield::Layers;
use engine_heightfield::layer::{TerrainLayer, layer_kind};
use engine_heightfield::layers::MAX_HEIGHT;
use engine_heightfield::noise::smooth;
use engine_math::max;
use engine_settings::{InvalidSetting, Rule, Setting, settings};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum StampOp {
    Raise(#[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = MAX_HEIGHT)))] f32),
    Lower(#[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = MAX_HEIGHT)))] f32),
    Flatten(
        #[cfg_attr(feature = "schema", schemars(range(min = -MAX_HEIGHT, max = MAX_HEIGHT)))] f32,
    ),
    Hole,
}

impl Setting for StampOp {
    fn check(&self, field: &'static str, _: Rule) -> Result<(), InvalidSetting> {
        let (variant, height, rule) = match *self {
            Self::Raise(height) => ("raise", height, Rule::Range(0.0, MAX_HEIGHT as f64)),
            Self::Lower(height) => ("lower", height, Rule::Range(0.0, MAX_HEIGHT as f64)),
            Self::Flatten(height) => (
                "flatten",
                height,
                Rule::Range(-MAX_HEIGHT as f64, MAX_HEIGHT as f64),
            ),
            Self::Hole => return Ok(()),
        };
        height
            .check(variant, rule)
            .map_err(|invalid| invalid.within(field))
    }
}

settings! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct Stamp {
        pub center: [f32; 2] = [0.0; 2] => range(-Layers::MAX_COORDINATE, Layers::MAX_COORDINATE),
        pub radius: f32 = 1.0 => range(0.0, Layers::MAX_COORDINATE),
        #[serde(default)]
        pub falloff: f32 = 0.0 => range(0.0, Layers::MAX_COORDINATE),
        pub op: StampOp = StampOp::Hole => any,
    }
}

impl Stamp {
    #[inline]
    pub fn weight(&self, x: f32, z: f32) -> f32 {
        let dx = x - self.center[0];
        let dz = z - self.center[1];
        let distance = (dx * dx + dz * dz).sqrt();
        1.0 - smooth((distance - self.radius) / max(self.falloff, 1.0e-6))
    }

    pub fn reach(&self) -> f32 {
        self.radius + self.falloff
    }
}

const STAMP: &str = "stamp";

pub fn with_stamps(below: &Layers, stamps: &[Stamp]) -> Result<Layers, Error> {
    let mut added: Vec<Arc<dyn TerrainLayer>> = Vec::new();
    if !stamps.is_empty() {
        let kind = layer_kind(STAMP)
            .ok_or_else(|| Error::new(ErrorCode::UnknownValue, "$.stamps").variants([STAMP]))?;
        for (offset, stamp) in stamps.iter().enumerate() {
            let at = format!("$.layers[{}]", below.layers().len() + offset);
            stamp.validate().map_err(|invalid| {
                Error::from(invalid)
                    .coded(ErrorCode::InvalidValue)
                    .within(&at)
            })?;
            let bytes = engine_codec::encode(stamp);
            let made = (kind.read)(&mut Reader::new(&bytes))
                .map_err(|error| Error::from(error).within(&at))?;
            added.push(made);
        }
    }
    Ok(below.with_added(added))
}

tagged!(StampOp {
    0 => Raise(meters),
    1 => Lower(meters),
    2 => Flatten(height),
    3 => Hole,
});

record!(Stamp {
    center,
    radius,
    falloff,
    op,
});
