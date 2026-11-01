use engine_core::bevy_ecs::prelude::Component;
use engine_settings::fields;
use serde::{Deserialize, Serialize};

fields! {
    #[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct Grounded {
        pub offset: f32 = 0.0 => range(-10_000.0, 10_000.0),
    }
}
