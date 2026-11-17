use engine_core::{Expected, ExtraRules, InvalidSetting, TextureHandle, fields, settings};

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "WaterOptics")]
    pub struct WaterOptics {
        pub forward: f32 = 0.4 => range(0.01, 0.95),
        pub bubbles: f32 = 30.0 => range(0.0, 1000.0),
        pub bubble_albedo: f32 = 0.45 => range(0.0, 1.0),
        pub refraction_reach: f32 = 4.0 => range(0.0, 100.0),
        pub roughness: [f32; 2] = [0.03, 0.6] => range(0.01, 1.0),
    }
}

fields! {
    extra_rules;
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "WaterWaves")]
    pub struct WaterWaves {
        pub count: u32 = 40 => range(1, crate::gpu::WATER_WAVES_MOST),
        pub longest: f32 = 6.0 => range(0.01, 1000.0),
        pub shortest: f32 = 0.05 => range(0.01, 1000.0),
        pub steepness: f32 = 0.2 => range(0.0, 1.0),
        pub tilt: f32 = 0.7 => range(0.0, 4.0),
        pub spread: [f32; 2] = [0.8, 1.1] => range(0.0, std::f32::consts::PI),
        pub chop: f32 = 0.6 => range(0.0, 1.0),
        pub seed: u32 = 2_654_435_769 => any,
        pub breaking: f32 = 0.78 => range(0.01, 4.0),
        pub still: [f32; 2] = [0.15, 0.7] => range(0.0, 100.0),
        pub shown: [f32; 2] = [2.0, 4.0] => range(1.0, 64.0),
        pub gust_strength: [f32; 2] = [0.8, 0.4] => range(0.0, 8.0),
        pub gust_size: [f32; 2] = [90.0, 33.3] => range(1.0, 100000.0),
        pub gust_drift: [f32; 2] = [1.0, 0.7] => range(0.0, 4.0),
        pub gust_long: f32 = 0.7 => range(0.0, 1.0),
        pub gust_calmest: f32 = 0.15 => range(0.0, 1.0),
        pub grouping: f32 = 0.9 => range(0.0, 8.0),
        pub push: [f32; 2] = [0.43, 0.15] => range(0.0, 1000.0),
        pub push_size: [f32; 2] = [17.5, 5.6] => range(0.01, 100000.0),
        pub push_ride: [f32; 2] = [1.0, 0.6] => range(0.0, 4.0),
    }
}

impl ExtraRules for WaterWaves {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.shortest > self.longest {
            return Err(InvalidSetting::new("shortest", Expected::Sorted));
        }
        Ok(())
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "WaterFlow")]
    pub struct WaterFlow {
        pub moving: [f32; 2] = [0.05, 0.35] => range(0.0, 100.0),
        pub frame: f32 = 1.5 => range(0.05, 100.0),
        pub carry_slowest: f32 = 0.1 => range(0.001, 10.0),
        pub carry_step: f32 = 1.4 => range(1.01, 16.0),
        pub rung_band: f32 = 0.15 => range(0.001, 0.5),
        pub relief_steepest: f32 = 0.9 => range(0.01, 100.0),
        pub shown: [f32; 2] = [8.0, 16.0] => range(0.0, 1024.0),
        pub drift_ride: f32 = 6.0 => range(0.0, 1000.0),
        pub drift_seconds: [f32; 2] = [0.25, 1.5] => range(0.001, 600.0),
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "WaterRipples")]
    pub struct WaterRipples {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub texture: Option<TextureHandle> = None => any,
        pub tile: f32 = 3.2 => range(0.01, 1000.0),
        pub fine_tile: f32 = 1.1 => range(0.01, 1000.0),
        pub stretch: f32 = 1.8 => range(0.05, 64.0),
        pub reach: f32 = 1.0 => range(0.0, 32.0),
        pub coarse: f32 = 0.6 => range(0.0, 1.0),
        pub skew: f32 = 0.2 => range(-4.0, 4.0),
        pub least: f32 = 0.045 => range(0.0, 1.0),
        pub share: f32 = 0.6 => range(0.0, 4.0),
        pub fine_share: f32 = 0.4 => range(0.0, 4.0),
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "WaterBoils")]
    pub struct WaterBoils {
        pub across: f32 = 0.8 => range(0.01, 1000.0),
        pub steep: f32 = 0.14 => range(0.0, 4.0),
        pub churning: [f32; 2] = [0.25, 0.7] => range(0.0, 1.0),
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "WaterBumps")]
    pub struct WaterBumps {
        pub froude: [f32; 2] = [0.35, 1.0] => range(0.0, 100.0),
        pub steep: f32 = 0.3 => range(0.0, 4.0),
        pub depths: f32 = 5.0 => range(0.0, 1000.0),
        pub deep: f32 = 0.3 => range(0.0, 4.0),
        pub shortest: f32 = 0.08 => range(0.001, 100.0),
        pub crests: f32 = 3.0 => range(0.05, 64.0),
        pub flicker: f32 = 0.6 => range(0.0, 4.0),
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "WaterFoam")]
    pub struct WaterFoam {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub texture: Option<TextureHandle> = None => any,
        pub tile: f32 = 3.0 => range(0.01, 1000.0),
        pub steep: f32 = 4.0 => range(0.0, 64.0),
        pub fades: [f32; 2] = [0.004, 0.08] => range(0.0, 100.0),
        pub soft: f32 = 0.5 => range(0.001, 1.0),
        pub halo: f32 = 0.5 => range(0.001, 1.0),
        pub body: f32 = 1.0 => range(0.001, 1.0),
        pub open: f32 = 0.0 => range(0.0, 16.0),
        pub filmy: f32 = 0.3 => range(0.0, 1.0),
        pub covers: [f32; 2] = [0.12, 0.95] => range(0.0, 1.0),
        pub tearing: [f32; 2] = [0.7, 1.3] => range(0.0, 100.0),
        pub froude: [f32; 2] = [1.0, 1.7] => range(0.0, 100.0),
        pub breaking: [f32; 2] = [0.6, 1.2] => range(0.0, 2.0),
        pub drop: [f32; 2] = [0.08, 0.35] => range(0.0, 100.0),
        pub blocking: [f32; 2] = [0.5, 0.9] => range(0.0, 1.0),
        pub crested: [f32; 2] = [1.1, 0.6] => range(-8.0, 8.0),
        pub optical: f32 = 4.0 => range(0.01, 1000.0),
        pub hollow: f32 = 2.2 => range(0.0, 16.0),
        pub albedo: f32 = 0.66 => range(0.0, 1.0),
        pub sky: f32 = 0.6 => range(0.0, 1.0),
        pub around: [f32; 2] = [0.35, 0.5] => range(0.0, 4.0),
        pub rough: [f32; 2] = [0.3, 0.7] => range(0.01, 1.0),
        pub mirrors: f32 = 0.5 => range(0.001, 1.0),
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "WaterLace")]
    pub struct WaterLace {
        pub amount: f32 = 0.35 => range(0.0, 1.0),
        pub depth: [f32; 2] = [0.04, 0.12] => range(0.0, 100.0),
        pub speed: [f32; 2] = [0.08, 0.3] => range(0.0, 100.0),
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "WaterDriftLines")]
    pub struct WaterDriftLines {
        pub amount: [f32; 2] = [0.0, 0.28] => range(0.0, 1.0),
        pub gathers: [f32; 2] = [0.5, 2.0] => range(0.0, 8.0),
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "WaterShore")]
    pub struct WaterShore {
        pub fade: f32 = 0.1 => range(0.0001, 100.0),
        pub edge: f32 = 0.05 => range(0.0001, 100.0),
        pub edge_pixels: f32 = 1.5 => range(0.0, 64.0),
        pub racing: [f32; 2] = [0.8, 1.6] => range(0.0, 100.0),
        pub wet: f32 = 0.35 => range(0.0, 1.0),
        pub soaked: f32 = 0.03 => range(0.0001, 100.0),
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "WaterCaustics")]
    pub struct WaterCaustics {
        pub contrast: f32 = 0.3 => range(0.0, 4.0),
        pub most: f32 = 1.5 => range(1.0, 100.0),
        pub depth: f32 = 1.5 => range(0.001, 1000.0),
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "WaterFalls")]
    pub struct WaterFalls {
        pub wander_scale: [f32; 2] = [1.6, 3.5] => range(0.001, 1000.0),
        pub wander_reach: f32 = 0.12 => range(0.0, 100.0),
        pub curtain_scale: [f32; 2] = [0.55, 0.9] => range(0.001, 1000.0),
        pub rope_scale: [f32; 2] = [0.16, 0.33] => range(0.001, 1000.0),
        pub strand_scale: [f32; 2] = [0.045, 0.11] => range(0.001, 1000.0),
        pub scale_shares: [f32; 3] = [0.5, 0.35, 0.35] => range(0.0, 4.0),
        pub gathering: f32 = 2.5 => range(0.0, 64.0),
        pub glassy_gathering: f32 = 0.15 => range(0.0, 1.0),
        pub tear_thickness: f32 = 0.004 => range(0.00001, 10.0),
        pub rock_roughness: f32 = 0.06 => range(0.00001, 10.0),
        pub broken_stands: f32 = 0.5 => range(-16.0, 16.0),
        pub tear_softness: f32 = 0.4 => range(0.001, 16.0),
        pub edge_thinning: f32 = 0.3 => range(0.0, 16.0),
        pub plunge_drowns: [f32; 2] = [1.0, 2.0] => range(0.0, 1000.0),
        pub relief_steepest: f32 = 1.2 => range(0.01, 100.0),
        pub furrow_shade: f32 = 0.55 => range(0.0, 1.0),
        pub matte_roughness: f32 = 0.7 => range(0.01, 1.0),
        pub albedo: f32 = 0.8 => range(0.0, 1.0),
        pub thin_tint: f32 = 0.6 => range(0.0, 1.0),
        pub scattering: f32 = 500.0 => range(0.0, 100000.0),
        pub forward: f32 = 0.85 => range(0.0, 1.0),
        pub faintest: f32 = 0.01 => range(0.0, 1.0),
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "WaterFallSheets")]
    pub struct WaterFallSheets {
        pub row: f32 = 0.75 => range(0.05, 64.0),
        pub columns: u32 = 8 => range(1, 64),
        pub overhang: f32 = 1.15 => range(0.0, 8.0),
        pub edge_spread: f32 = 0.035 => range(0.0, 4.0),
        pub bow: f32 = 0.12 => range(0.0, 8.0),
        pub thickness: [f32; 2] = [0.05, 1.2] => range(0.0001, 100.0),
        pub slowest: f32 = 0.3 => range(0.001, 100.0),
        pub springing: f32 = 4.0 => range(0.0, 64.0),
        pub breaking_from: f32 = 0.3 => range(0.0, 0.99),
        pub inception: [f32; 3] = [1.0, 2.0, 6.0] => range(0.001, 1000.0),
        pub skim_reach: f32 = 6.0 => range(0.0, 1000.0),
        pub air_share: f32 = 0.9 => range(0.0, 4.0),
        pub aired: [f32; 2] = [0.15, 0.55] => range(0.0, 1.0),
        pub airing: f32 = 0.4 => range(0.001, 600.0),
        pub air_release: f32 = 1.2 => range(0.001, 600.0),
        pub lingering: f32 = 0.3 => range(0.0, 1.0),
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "WaterFallSpray")]
    pub struct WaterFallSpray {
        pub color: [f32; 3] = [0.92, 0.95, 0.98] => range(0.0, 4.0),
        pub alpha: f32 = 0.45 => range(0.0, 1.0),
        pub foot_share: f32 = 0.45 => range(0.0, 4.0),
        pub watts: f32 = 400.0 => range(0.001, 1.0e9),
        pub rate: [f32; 2] = [8.0, 240.0] => range(0.0, 100000.0),
        pub throw: f32 = 0.35 => range(0.0, 16.0),
        pub speed: [f32; 2] = [1.5, 8.0] => range(0.0, 1000.0),
        pub forward: f32 = 0.4 => range(-16.0, 16.0),
        pub fan: f32 = 0.8 => range(0.0, 4.0),
        pub size: [f32; 2] = [0.025, 0.06] => range(0.0, 100.0),
        pub stretch: f32 = 0.03 => range(0.0, 16.0),
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "WaterMist")]
    pub struct WaterMist {
        pub color: [f32; 3] = [0.94, 0.96, 1.0] => range(0.0, 4.0),
        pub alpha: f32 = 0.12 => range(0.0, 1.0),
        pub full_watts: f32 = 5.0e4 => range(0.001, 1.0e9),
        pub least: f32 = 0.15 => range(0.0, 1.0),
        pub puff_per_head: f32 = 0.35 => range(0.0, 100.0),
        pub puff: [f32; 2] = [1.2, 9.0] => range(0.001, 1000.0),
        pub growth: [f32; 2] = [0.6, 1.8] => range(0.0, 100.0),
        pub spread: f32 = 0.25 => range(0.0, 4.0),
        pub overlap: f32 = 8.0 => range(0.0, 1000.0),
        pub most: f32 = 96.0 => range(0.0, 100000.0),
        pub seconds: f32 = 6.0 => range(0.001, 600.0),
        pub travel: f32 = 1.5 => range(0.0, 100.0),
        pub fan: f32 = 1.1 => range(0.0, 4.0),
        pub fade_in: f32 = 0.2 => range(0.0, 1.0),
        pub uneven: f32 = 0.3 => range(0.0, 1.0),
    }
}

fields! {
    #[derive(
        Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
    )]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "WaterDebugView")]
    pub struct WaterDebugView {
        pub deepest: f32 = 2.0 => range(0.001, 10000.0),
        pub fastest: f32 = 6.0 => range(0.001, 1000.0),
        pub froude_top: f32 = 3.0 => range(1.01, 100.0),
        pub critical: f32 = 0.04 => range(0.0, 1.0),
    }
}

settings! {
    #[section("water")]
    #[check(groups_checked)]
    #[switch(on)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneWater")]
    pub struct Water {
        pub color: [f32; 3] = [0.012, 0.045, 0.05] => range(0.0, 1.0),
        pub clarity: f32 = 2.5 => range(0.1, 100.0),
        pub sediment: f32 = 0.15 => range(0.0, 50.0),
        pub sediment_color: [f32; 3] = [0.05, 0.065, 0.03] => range(0.0, 1.0),
        pub subdivision_error: f32 = 1.0 => range(0.25, 64.0),
        pub tile_feed: u32 = 4096 => at_least(1),
        pub optics: WaterOptics = WaterOptics::default() => any,
        pub waves: WaterWaves = WaterWaves::default() => any,
        pub flow: WaterFlow = WaterFlow::default() => any,
        pub ripples: WaterRipples = WaterRipples::default() => any,
        pub boils: WaterBoils = WaterBoils::default() => any,
        pub bumps: WaterBumps = WaterBumps::default() => any,
        pub foam: WaterFoam = WaterFoam::default() => any,
        pub lace: WaterLace = WaterLace::default() => any,
        pub drift_lines: WaterDriftLines = WaterDriftLines::default() => any,
        pub shore: WaterShore = WaterShore::default() => any,
        pub caustics: WaterCaustics = WaterCaustics::default() => any,
        pub falls: WaterFalls = WaterFalls::default() => any,
        pub fall_sheets: WaterFallSheets = WaterFallSheets::default() => any,
        pub fall_spray: WaterFallSpray = WaterFallSpray::default() => any,
        pub mist: WaterMist = WaterMist::default() => any,
        pub debug_view: WaterDebugView = WaterDebugView::default() => any,
    }
}

fn groups_checked(water: &Water) -> Result<(), InvalidSetting> {
    let groups = [
        ("optics", water.optics.validate()),
        ("waves", water.waves.validate()),
        ("flow", water.flow.validate()),
        ("ripples", water.ripples.validate()),
        ("boils", water.boils.validate()),
        ("bumps", water.bumps.validate()),
        ("foam", water.foam.validate()),
        ("lace", water.lace.validate()),
        ("drift_lines", water.drift_lines.validate()),
        ("shore", water.shore.validate()),
        ("caustics", water.caustics.validate()),
        ("falls", water.falls.validate()),
        ("fall_sheets", water.fall_sheets.validate()),
        ("fall_spray", water.fall_spray.validate()),
        ("mist", water.mist.validate()),
        ("debug_view", water.debug_view.validate()),
    ];
    for (group, checked) in groups {
        checked.map_err(|invalid| invalid.within(group))?;
    }
    Ok(())
}
