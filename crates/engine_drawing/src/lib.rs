#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod binds;
mod stand_in;
mod view;

pub use binds::MaterialBinds;
pub use stand_in::StandIn;
pub use view::{DetailLimit, SizeSkip, nearest_squared, outside, pixel_scale};
