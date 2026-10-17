//! Authored per-material alpha-mask mip filtering.

use engine_core::AlphaCoverage;
use serde::{Deserialize, Serialize};

/// Requested cutout filtering; inspect AlphaCoverage for the effective
/// state and fallback reason after rendering. Omission means auto.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SceneAlphaCoverage {
    /// Use a coverage curve when eligible, otherwise the legacy multiplier.
    #[default]
    Auto,
    /// Always use the legacy multiplier, including shadows and source bakes.
    Legacy,
}

impl From<SceneAlphaCoverage> for AlphaCoverage {
    fn from(mode: SceneAlphaCoverage) -> Self {
        match mode {
            SceneAlphaCoverage::Auto => Self::Auto,
            SceneAlphaCoverage::Legacy => Self::Legacy,
        }
    }
}

impl From<AlphaCoverage> for SceneAlphaCoverage {
    fn from(mode: AlphaCoverage) -> Self {
        match mode {
            AlphaCoverage::Auto => Self::Auto,
            AlphaCoverage::Legacy => Self::Legacy,
        }
    }
}
