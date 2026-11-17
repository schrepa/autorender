use bevy_ecs::prelude::*;
use engine_core::{Readable, Readback, stats};

use crate::sky::{
    DERIVED_CLEAR, DERIVED_HORIZON, DERIVED_ROWS, DERIVED_SKY, DERIVED_SKY_UP, DERIVED_SUN,
    SkyPlan,
};

#[derive(Resource, Default)]
pub(crate) struct SkyLightLevel {
    derived: [[f32; 4]; DERIVED_ROWS],
}

impl Readable for SkyLightLevel {
    fn arrived(&mut self, _: u32, read: Readback<&[u8]>) {
        let words = read
            .value
            .chunks_exact(4)
            .filter_map(|word| <[u8; 4]>::try_from(word).ok())
            .map(f32::from_le_bytes);
        for (kept, word) in self.derived.iter_mut().flatten().zip(words) {
            *kept = word;
        }
    }
}

impl SkyLightLevel {
    fn row(&self, row: u64) -> [f32; 4] {
        self.derived
            .get(row as usize)
            .copied()
            .unwrap_or_default()
    }

    fn level(&self) -> f32 {
        self.row(DERIVED_HORIZON)[3]
    }
}

fn luminance(light: [f32; 4]) -> f32 {
    0.2126 * light[0] + 0.7152 * light[1] + 0.0722 * light[2]
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
    #[section("lighting_stats")]
    pub struct LightingStats {
        pub sun_facing: f32,
        pub sun_level: f32,
        pub sky_level_clear: f32,
        pub sky_level: f32,
        pub sun_to_sky: f32,
        pub light_level: f32,
    }
}

pub(crate) fn lighting_stats(light: Res<SkyLightLevel>, mut stats: ResMut<LightingStats>) {
    let sun = light.row(DERIVED_SUN);
    let sun_facing = luminance(sun);
    let sun_level = sun_facing * sun[3].max(0.0);
    let (flat, up) = (light.row(DERIVED_SKY), light.row(DERIVED_SKY_UP));
    let sky_level = luminance(std::array::from_fn(|channel| flat[channel] + up[channel]));
    stats.set_if_neq(LightingStats {
        sun_facing,
        sun_level,
        sky_level_clear: luminance(light.row(DERIVED_CLEAR)),
        sky_level,
        sun_to_sky: if sky_level > 0.0 {
            sun_level / sky_level
        } else {
            0.0
        },
        light_level: light.level(),
    });
}

pub(crate) fn sky_stats(
    (plan, light): (Res<SkyPlan>, Res<SkyLightLevel>),
    (mut clouds, mut sky): (ResMut<CloudStats>, ResMut<SkyStats>),
) {
    let (rows, view_pixels) = plan.cloud_rows();
    clouds.set_if_neq(CloudStats { rows, view_pixels });
    let tables = plan.tables();
    sky.set_if_neq(SkyStats {
        tables,
        light: if tables > 0 { light.level() } else { 0.0 },
    });
}
