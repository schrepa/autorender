#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod codes;
mod error;
pub mod fields;
mod hash;

pub use codes::ErrorCode;
pub use error::{Allowed, Detail, Error, Expected, MAX_GOT_BYTES, closest};
pub use fields::{
    ExtraRules, ID_PATTERN, InvalidSetting, MAX_ID_BYTES, Rule, Setting, is_id, state, stated,
    valid_uri,
};
pub use hash::content_hash;

pub mod rule {
    pub use crate::fields::{Bound, above, above_to, any, at_least, count, finite, id, range};
}

#[doc(hidden)]
pub mod __private {
    pub use ::schemars;
}
