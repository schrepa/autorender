#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

pub mod bed;
pub mod given;
pub mod grid;
pub mod hydraulics;
pub mod hydrology;
pub mod layer;
pub mod layers;
pub mod noise;
pub mod path;
pub mod rivers;

pub use bed::{Bed, NARROWEST, Reach};
pub use given::GivenRiver;
pub use grid::{Pyramid, TILE_QUADS, TILE_SAMPLES, TileKey};
pub use hydrology::{Fall, HydrologyBake, HydrologySample, HydrologyStats, RIVER_FAR, WaterGrid};
pub use layer::{LayerBake, LayerTile, TerrainLayer};
pub use layers::{Blend, HeightImage, Layers};
pub use path::{Bridge, PathSample};
pub use rivers::{Given, RIVER_STEP, River, RiverLine, Rivers};

pub fn schema<T: schemars::JsonSchema>() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(T)).unwrap_or(serde_json::Value::Null)
}
