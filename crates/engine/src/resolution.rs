use engine_core::{Engine, Expected, InvalidSetting, settings};

settings! {
    #[section("resolution")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneResolution")]
    pub struct Resolution {
        pub scale: f32 = 1.0 => range(0.25, 2.0),
    }
}

settings! {
    #[section("dynamic_resolution")]
    #[check(sorted_scales)]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneDynamicResolution")]
    pub struct DynamicResolution {
        pub enabled: bool = false => any,
        pub min_scale: f32 = 0.5 => range(0.25, 1.0),
        pub max_scale: f32 = 1.0 => range(0.25, 1.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub budget_ms: Option<f32> = None => range(1.0, 1000.0),
    }
}

fn sorted_scales(resolution: &DynamicResolution) -> Result<(), InvalidSetting> {
    if resolution.min_scale > resolution.max_scale {
        return Err(InvalidSetting::new("min_scale", Expected::Sorted)
            .got(resolution.min_scale)
            .nearest(resolution.max_scale));
    }
    Ok(())
}

pub(crate) fn register(engine: &mut Engine) {
    engine
        .settings::<Resolution>()
        .settings::<DynamicResolution>();
}
