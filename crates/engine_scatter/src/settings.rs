use std::borrow::Cow;

use engine_core::__private::{schemars, serde};
use engine_core::{
    Cut, Error, ErrorCode, GeometryHandle, InvalidSetting, MaterialHandle, Rule, Setting, settings,
    stats,
};

pub const MAX_PLACEMENT_LODS: usize = 7;

pub const MAX_PLACEMENT_PARTS: usize = 2;

const SWAY: Rule = Rule::range(0.0, 10.0);

settings! {
    #[section("foliage")]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneFoliage")]
    pub struct FoliageSettings {
        pub transmission: f32 = 1.0 => range(0.0, 4.0),
        pub variation: f32 = 1.0 => range(0.0, 1.0),
        pub hashed_cutoff: bool = false => any,
        pub ground_blend: f32 = 0.4 => range(0.0, 4.0),
    }
}

settings! {
    #[section("canopy")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneCanopy")]
    pub struct CanopySettings {
        pub handover_pixels: f32 = 1.0 => range(0.25, 256.0),
        pub sky_through: f32 = 0.25 => range(0.0, 1.0),
        pub sun_through: f32 = 0.1 => range(0.0, 1.0),
        pub crown_light: f32 = 0.6 => range(0.0, 1.0),
        pub roughness: f32 = 0.85 => range(0.03, 1.0),
        pub relief: f32 = 1.0 => range(0.0, 4.0),
    }
}

settings! {
    #[section("scatter")]
    #[check(check_looks)]
    #[serde(deny_unknown_fields)]
    pub struct ScatterSettings {
        pub looks: Vec<PlacedLook> = Vec::new() => any,
        pub parts: u32 = MAX_PLACEMENT_PARTS as u32 => any,
        pub lods: u32 = MAX_PLACEMENT_LODS as u32 => at_most(MAX_PLACEMENT_LODS),
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(crate = "engine_core::__private::serde", deny_unknown_fields)]
#[schemars(crate = "engine_core::__private::schemars")]
pub struct PlacedLook {
    pub mesh: GeometryHandle,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lods: Vec<GeometryHandle>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<PlacedPart>,
    #[serde(default, skip_serializing_if = "is_zero")]
    #[schemars(transform = engine_core::stated(SWAY))]
    pub sway: f32,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(crate = "engine_core::__private::serde", deny_unknown_fields)]
#[schemars(crate = "engine_core::__private::schemars")]
pub struct PlacedPart {
    pub material: MaterialHandle,
    pub meshes: Vec<GeometryHandle>,
}

stats! {
    #[section("scatter_stats")]
    pub struct ScatterStats {
        pub placed_objects: u32,
        pub placed_far_objects: u32,
        pub placed_impostors: u32,
        pub dropped_objects: u32,
        pub canopy_maps: u32,
        pub canopy_waiting: u32,
        pub canopy_failure: Option<String>,
        pub impostors: Cut,
        pub parts: Cut,
        pub lods: Cut,
        pub measured_levels: Vec<Vec<f32>>,
    }
}

fn is_zero(value: &f32) -> bool {
    *value == 0.0
}

fn refused(field: String, error: Error) -> InvalidSetting {
    InvalidSetting {
        field: Cow::Owned(field),
        detail: *error.detail,
    }
}

fn lods_fit(at: &str, lods: &[GeometryHandle]) -> Result<(), InvalidSetting> {
    if lods.len() > MAX_PLACEMENT_LODS {
        return Err(refused(
            format!("{at}.lods"),
            Error::new(ErrorCode::OutOfRange, "$")
                .at_most(MAX_PLACEMENT_LODS as u64)
                .got(lods.len()),
        ));
    }
    Ok(())
}

fn parts_fit(at: &str, parts: &[PlacedPart], levels: usize) -> Result<(), InvalidSetting> {
    if parts.len() > MAX_PLACEMENT_PARTS {
        return Err(refused(
            format!("{at}.parts"),
            Error::new(ErrorCode::OutOfRange, "$")
                .at_most(MAX_PLACEMENT_PARTS as u64)
                .got(parts.len()),
        ));
    }
    match parts
        .iter()
        .position(|part| part.meshes.is_empty() || part.meshes.len() > levels)
    {
        Some(index) => Err(refused(
            format!("{at}.parts[{index}].meshes"),
            Error::new(ErrorCode::OutOfRange, "$")
                .range(1.0, levels as f64)
                .got(parts[index].meshes.len()),
        )),
        None => Ok(()),
    }
}

fn check_looks(settings: &ScatterSettings) -> Result<(), InvalidSetting> {
    for (index, look) in settings.looks.iter().enumerate() {
        let at = format!("looks[{index}]");
        look.sway
            .check("sway", SWAY)
            .map_err(|invalid| invalid.within("looks").indexed("looks", index))?;
        lods_fit(&at, &look.lods)?;
        parts_fit(&at, &look.parts, 1 + look.lods.len())?;
    }
    Ok(())
}

impl ScatterSettings {
    pub(crate) fn look(&self, mesh: GeometryHandle) -> Option<&PlacedLook> {
        self.looks.iter().find(|look| look.mesh == mesh)
    }
}
