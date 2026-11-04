#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

pub mod bed;
pub mod errors;
pub mod gather;
pub mod given;
pub mod grid;
pub mod hydraulics;
pub mod hydrology;
pub mod layer;
pub mod layers;
pub mod noise;
pub mod path;
pub mod rivers;
pub mod sights;
pub mod tile;

pub use bed::{Bed, NARROWEST, Reach};
pub use errors::ErrorPyramid;
pub use gather::{Gather, TileHeights};
pub use given::GivenRiver;
pub use grid::{
    MAX_LEVEL, Pyramid, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey, covers, reaches,
    reaches_span,
};
pub use hydrology::{Fall, HydrologyBake, HydrologySample, HydrologyStats, RIVER_FAR, WaterGrid};
pub use layer::{LayerTile, TerrainLayer};
pub use layers::{Blend, HeightImage, Layers, Stamp, StampOp};
pub use path::{Bridge, PathSample};
pub use rivers::{Given, RIVER_STEP, River, RiverLine, Rivers};
pub use sights::{FlowSummary, LakeSight, RiverCourse, RiverListing, RiverSight};

pub fn bound(value: f32) -> f64 {
    engine_codec::rule::Bound::bound(value)
}

pub fn schema<T: schemars::JsonSchema>() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(T)).unwrap_or(serde_json::Value::Null)
}
