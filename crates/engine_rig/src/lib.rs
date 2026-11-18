#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod bound;
mod bytes;
mod clip;
mod error;
mod pose;
mod skeleton;

pub use bound::{Bound, PaletteRow, bind, model_matrices, palette};
pub use clip::{Clip, Keys, RootTrack, RotationTrack, TranslationTrack};
pub use pose::{Reach, blend, reach, shift, turn};
pub use skeleton::{Joint, Posed, Skeleton};
