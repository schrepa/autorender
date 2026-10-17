//! Prepared material coverage facts, shared by renderers and inspection.

use serde::Serialize;

/// The coverage state actually prepared for a material binding. Corrected
/// tests still use legacy at non-unit combined opacity and in uniform mip tails.
#[derive(Debug, Clone, Copy, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "effective", rename_all = "snake_case")]
pub enum AlphaCoverageState {
    /// A texture/cutoff-specific curve is present in the GPU material.
    Corrected {
        /// The number of mip levels in its source texture.
        mip_levels: u32,
        /// Mip estimates at or beyond this level use legacy filtering.
        legacy_from_mip: u32,
    },
    /// The original alpha-test multiplier is used for this binding.
    Legacy {
        /// Why: requested_legacy, not_masked, missing_base_texture,
        /// cutoff, material_opacity, world_mapping, sampler, texture_format,
        /// texture_dimensions, no_mips, empty_or_full_coverage or invalid_factor.
        reason: &'static str,
    },
}
