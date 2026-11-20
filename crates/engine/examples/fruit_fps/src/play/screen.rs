use engine::prelude::{Resource, settings};

use crate::rules::BuyRequest;

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScreenState {
    pub holds_pointer: bool,
}

#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct ScreenBuys {
    pub requests: Vec<BuyRequest>,
}

settings! {
    #[section("options")]
    #[serde(deny_unknown_fields)]
    pub struct Options {
        pub crosshair: u32 = 2 => any,
        pub crosshair_gap: f32 = 4.0 => range(0.0, 20.0),
        pub crosshair_length: f32 = 6.0 => range(1.0, 20.0),
        pub crosshair_thickness: f32 = 2.0 => range(1.0, 6.0),
        pub crosshair_outline: f32 = 2.0 => range(0.0, 3.0),
        pub crosshair_dot: bool = true => any,
    }
}
