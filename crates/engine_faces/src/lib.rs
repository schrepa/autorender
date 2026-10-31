#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

pub mod defaults;
pub mod display;
pub mod environment;
pub mod heightfield;
pub mod lights;
pub mod occupancy;
pub mod placed;
pub mod places;
pub mod shadows;
pub mod slots;
pub mod smokes;
pub mod sun;
