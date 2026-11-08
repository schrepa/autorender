use bevy_ecs::prelude::*;
use engine_core::{Readable, Readback, stats};

use crate::sky::SkyPlan;

#[derive(Resource, Default)]
pub(crate) struct SkyLightLevel {
    read: f32,
}

impl Readable for SkyLightLevel {
    fn arrived(&mut self, _: u32, read: Readback<&[u8]>) {
        if let Some(word) = read
            .value
            .get(..4)
            .and_then(|word| <[u8; 4]>::try_from(word).ok())
        {
            self.read = f32::from_le_bytes(word);
        }
    }
}

#[derive(
    Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct StaticTableRow {
    pub kind: String,
    pub parts: u64,
    pub arrived: u64,
    pub showing: bool,
}

stats! {
    #[section("static_tables")]
    pub struct StaticTablesView {
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
        pub light: f32,
    }
}

stats! {
    #[section("haze_stats")]
    pub struct HazeStats {
        pub active: bool,
    }
}

pub(crate) fn sky_stats(
    (plan, light): (Res<SkyPlan>, Res<SkyLightLevel>),
    (mut clouds, mut sky, mut haze): (ResMut<CloudStats>, ResMut<SkyStats>, ResMut<HazeStats>),
) {
    let (rows, view_pixels) = plan.cloud_rows();
    clouds.set_if_neq(CloudStats { rows, view_pixels });
    let tables = plan.tables();
    sky.set_if_neq(SkyStats {
        tables,
        light: if tables > 0 { light.read } else { 0.0 },
    });
    haze.set_if_neq(HazeStats {
        active: plan.hazes(),
    });
}
