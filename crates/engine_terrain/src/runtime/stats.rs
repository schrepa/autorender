use super::*;

stats! {
    #[section("terrain_stats")]
    pub struct TerrainStats {
        #[pending] pub pending: u32,
        pub terrains: u32,
        pub resident_tiles: u32,
        pub loading_tiles: u32,
        pub missing_tiles: u32,
        pub waiting_tiles: u32,
        pub cached_tiles: u32,
        pub cached_bytes: u64,
        pub restored_tiles: u64,
        pub prefetching_tiles: u32,
        pub tile_cut: Cut,
        pub dropped_tiles: u32,
        pub fallback_tiles: u32,
        pub baking: u32,
        pub bridges: u32,
        pub tunnels: u32,
        pub surroundings_tiles: u32,
        pub loading: bool,
        pub failed_jobs: u64,
        pub latest_job_failure: Option<String>,
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TerrainBakes {
    pub baking: bool,
    pub hydrology: Option<HydrologyStats>,
    pub layers: Vec<LayerStats>,
}

#[derive(Clone, Debug, PartialEq, Serialize, schemars::JsonSchema)]
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
