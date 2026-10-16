//! Bounded, requested pages of the renderer's production light-depth state.

use serde::Serialize;

/// One selected light, with its final GPU placement as exact binary32 bits.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DepthLightEntry {
    /// Index in the original selected light buffer.
    pub id: u32,
    /// World position after applying its displayed pose.
    pub position_bits: [u32; 3],
    /// Original range, also for spot lights.
    pub range_bits: u32,
}

/// A page of one opaque phase's production masks and selected light lists.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DepthOccupancyPage {
    /// Retained world identity.
    pub view: u64,
    /// Opaque phase, zero or one.
    pub phase: u32,
    /// Nonzero tag copied from that submitted phase's arena.
    pub generation: u32,
    /// Frames since the submitted phase.
    pub readback_age: u64,
    /// Original three-dimensional cluster.
    pub cluster: u32,
    /// Occupied fine-depth bits.
    pub mask: u32,
    /// The slice's 31 binary32 thresholds, as their exact bits.
    pub threshold_bits: Vec<u32>,
    /// Lengths before and after stable filtering.
    pub original_length: u32,
    /// Length of the filtered selected list.
    pub filtered_length: u32,
    /// Page offset in each list; pages hold at most 32 entries.
    pub offset: u32,
    /// The original selected entries at this offset.
    pub original: Vec<DepthLightEntry>,
    /// The filtered indices at this offset, from the same submitted phase.
    pub filtered: Vec<u32>,
    /// The full-list uniform's tag. Zero prevents reading a filtered tail.
    pub full_generation: u32,
    /// Unchanged capacity of the original index allocation, in entries.
    pub index_capacity: u32,
    /// Bytes required by the ranges arena at the unchanged capacity.
    pub arena_bytes: u64,
    /// Device binding/buffer limit, or the diagnostic budget supplied.
    pub allowed_bytes: u64,
    /// The normal allocation helper's answer for that budget.
    pub allocation_fits: bool,
}
