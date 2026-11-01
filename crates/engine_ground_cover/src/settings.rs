use std::borrow::Cow;
use std::collections::BTreeMap;

use engine_core::{Cut, Expected, ExtraRules, InvalidSetting, fields, settings, stats};

fields! {
    extra_rules;
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "Cover")]
    pub struct CoverLook {
        pub density: f32 = 90.0 => range(0.0, 400.0),
        pub height: [f32; 2] = [0.12, 0.4] => range(0.01, 4.0),
        pub width: f32 = 0.025 => range(0.001, 0.25),
        pub color: [f32; 3] = [0.13, 0.2, 0.05] => range(0.0, 1.0),
        pub sway: f32 = 0.6 => range(0.0, 1.0),
    }
}

impl ExtraRules for CoverLook {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.height[0] <= self.height[1] {
            Ok(())
        } else {
            Err(InvalidSetting::new("height", Expected::Sorted))
        }
    }

    fn sanitize_extra(self) -> Self {
        let [low, high] = self.height;
        Self {
            height: [
                engine_core::math::min(low, high),
                engine_core::math::max(low, high),
            ],
            ..self
        }
    }
}

settings! {
    #[section("ground_cover")]
    #[check(looks_checked)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneGroundCover")]
    pub struct GroundCover {
        pub enabled: bool = true => any,
        pub density: f32 = 1.0 => range(0.0, 4.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_blades: Option<u32> = None => budget,
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        pub looks: BTreeMap<String, CoverLook> = BTreeMap::new() => any,
    }
}

fn looks_checked(cover: &GroundCover) -> Result<(), InvalidSetting> {
    for (name, look) in &cover.looks {
        look.validate().map_err(|invalid| InvalidSetting {
            field: Cow::Owned(format!("looks.{name}.{}", invalid.field)),
            detail: invalid.detail,
        })?;
    }
    Ok(())
}

stats! {
    #[section("ground_cover_stats")]
    pub struct GroundCoverStats {
        pub blade_cut: Cut,
        pub reach: u32,
        pub tiers: u32,
    }
}
