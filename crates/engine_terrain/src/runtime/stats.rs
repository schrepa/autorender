use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TerrainStats {
    pub terrains: u32,
    pub resident_tiles: u32,
    pub loading_tiles: u32,
    pub missing_tiles: u32,
    pub waiting_tiles: u32,
    pub cached_tiles: u32,
    pub cached_bytes: u64,
    pub restored_tiles: u64,
    pub prefetching_tiles: u32,
    pub rewatered_tiles: u32,
    pub stale_water_tiles: u32,
    pub settled_records: u32,
    pub settling_records: u32,
    pub shaded_tiles: u32,
    pub shading_records: u32,
    pub dropped_tiles: u32,
    pub fallback_tiles: u32,
    pub baking: u32,
    pub lit: u32,
    pub lighting: u32,
    pub bounced: u32,
    pub bouncing: u32,
    pub colored: u32,
    pub coloring: u32,
    pub bridges: u32,
    pub tunnels: u32,
    pub falls: u32,
    pub white_water: u32,
    pub surroundings_tiles: u32,
    pub loading: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TerrainBakes {
    pub baking: bool,
    pub hydrology: Option<HydrologyStats>,
    pub layers: Vec<LayerStats>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LayerStats {
    pub index: usize,
    pub layer: String,
    pub stats: serde_json::Value,
}

pub(super) fn layer_stats(layers: &Layers) -> Vec<LayerStats> {
    layers
        .layers()
        .iter()
        .enumerate()
        .filter_map(|(index, layer)| {
            layer.stats().map(|stats| LayerStats {
                index,
                layer: layer.kind().name.to_owned(),
                stats,
            })
        })
        .collect()
}
