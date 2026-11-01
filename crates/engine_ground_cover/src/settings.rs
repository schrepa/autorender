use std::collections::BTreeMap;

use engine_core::{Cut, settings, stats};

#[derive(
    Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(default, deny_unknown_fields)]
pub struct CoverLook {
    pub density: f32,
    pub height: [f32; 2],
    pub width: f32,
    pub color: [f32; 3],
    pub sway: f32,
}

impl Default for CoverLook {
    fn default() -> Self {
        Self {
            density: 90.0,
            height: [0.12, 0.4],
            width: 0.025,
            color: [0.13, 0.2, 0.05],
            sway: 0.6,
        }
    }
}

settings! {
    #[section("ground_cover")]
    #[serde(deny_unknown_fields)]
    pub struct GroundCover {
        pub enabled: bool = true => any,
        pub density: f32 = 1.0 => range(0.0, 4.0),
        pub max_blades: Option<u32> = None => at_least(1),
        pub looks: BTreeMap<String, CoverLook> = BTreeMap::new() => any,
    }
}

stats! {
    #[section("ground_cover_stats")]
    pub struct GroundCoverStats {
        pub blade_cut: Cut,
        pub reach: u32,
        pub tiers: u32,
    }
}
