use bevy_ecs::prelude::*;
use engine_core::{Readback, stats};
use engine_render::DebugShown;

use crate::settings::{Haze, Sky, SkyLighting, haze_runs};

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct StaticTableRow {
    pub kind: String,
    pub parts: u64,
    pub arrived: u64,
    pub showing: bool,
}

stats! {
    #[section("static_tables")]
    pub struct StaticTablesView {
        #[pending] pub loading: u32,
        pub tables: Vec<StaticTableRow>,
        pub failed_jobs: u64,
        pub latest_job_failure: Option<String>,
    }
}

stats! {
    #[section("cloud_stats")]
    pub struct CloudStats {
        pub rows: u32,
        pub view_pixels: u32,
    }
}

stats! {
    #[section("sky_stats")]
    pub struct SkyStats {
        pub tables: u32,
        pub haze: bool,
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SkyCounts {
    rows: u32,
    view_pixels: u32,
    tables: u32,
}

impl Readback for SkyCounts {
    fn arrived(&mut self, _: u64, _: u32, bytes: &[u8]) {
        let word = |at: usize| {
            bytes
                .get(at * 4..at * 4 + 4)
                .map_or(0, |word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        };
        *self = Self {
            rows: word(0),
            view_pixels: word(1),
            tables: word(2),
        };
    }
}

pub(crate) fn sky_stats(
    (counts, sky, haze, light, shown): (Res<SkyCounts>, Res<Sky>, Res<Haze>, Res<SkyLighting>, Res<DebugShown>),
    mut clouds: ResMut<CloudStats>,
    mut stats: ResMut<SkyStats>,
) {
    clouds.set_if_neq(CloudStats {
        rows: counts.rows,
        view_pixels: counts.view_pixels,
    });
    stats.set_if_neq(SkyStats {
        tables: counts.tables,
        haze: shown.owner.is_none() && haze_runs(&haze, &sky, &light.values()),
    });
}
