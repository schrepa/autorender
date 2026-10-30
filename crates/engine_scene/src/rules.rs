use engine_core::glam::Vec2;
use engine_core::{Error, Grounded};
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};

pub const MAX_ALONG: usize = 10_000;

const MAX_DISTANCE: f32 = 1.0e7;

const MIN_EVERY: f32 = 0.01;

const MAX_OFFSET: f32 = 1.0e4;

const MAX_REACH: f32 = 1.0e6;

fn standing() -> Option<Grounded> {
    Some(Grounded::default())
}

engine_settings::settings! {
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct SceneAlong {
        pub id: String = String::new() => id,
        pub prefab: String = String::new() => id,
        pub terrain: String = String::new() => id,
        pub layer: String = String::new() => id,
        #[serde(default)]
        pub first: f32 = 0.0 => range(0.0, MAX_DISTANCE),
        pub every: f32 = 1.0 => range(MIN_EVERY, MAX_DISTANCE),
        #[serde(default)]
        pub offset: f32 = 0.0 => range(-MAX_OFFSET, MAX_OFFSET),
        #[serde(default = "standing")]
        pub grounded: Option<Grounded> = standing() => any,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub streamed: Option<SceneAlongStreamed> = None => any,
    }
}

engine_settings::settings! {
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct SceneAlongStreamed {
        pub reach: f32 = 0.0 => range(0.0, MAX_REACH),
    }
}

impl SceneAlong {
    pub fn spots(&self, points: &[[f32; 2]], path: &str) -> Result<Vec<(Vec2, f32)>, Error> {
        let mut spots = Vec::new();
        let mut next = self.first;
        let mut walked = 0.0;
        for pair in points.windows(2) {
            let (from, to) = (Vec2::from(pair[0]), Vec2::from(pair[1]));
            let length = from.distance(to);
            if length <= 0.0 {
                continue;
            }
            let along = (to - from) / length;
            while next <= walked + length {
                if spots.len() == MAX_ALONG {
                    return Err(Error::new(ErrorCode::OutOfRange, format!("{path}.every"))
                        .at_most(MAX_ALONG as u64)
                        .got(self.every));
                }
                let point = from + along * (next - walked) + along.perp() * self.offset;
                spots.push((point, -along.y.atan2(along.x)));
                next += self.every;
            }
            walked += length;
        }
        Ok(spots)
    }
}
