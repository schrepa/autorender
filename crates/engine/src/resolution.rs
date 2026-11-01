use bevy_ecs::resource::Resource;
use engine_core::{Engine, Error, ErrorCode, Section, settings};
use engine_settings::Expected;
use engine_settings::rule::range;

settings! {
    #[section("resolution")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneResolution")]
    pub struct Resolution {
        pub scale: f32 = 1.0 => range(0.25, 2.0),
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Resource,
    engine_core::__private::serde::Serialize,
    engine_core::__private::serde::Deserialize,
    engine_core::__private::schemars::JsonSchema,
)]
#[component(immutable)]
#[serde(crate = "engine_core::__private::serde", default, deny_unknown_fields)]
#[schemars(
    crate = "engine_core::__private::schemars",
    rename = "SceneDynamicResolution"
)]
pub struct DynamicResolution {
    pub enabled: bool,
    pub min_scale: f32,
    pub max_scale: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_ms: Option<f32>,
}

impl Default for DynamicResolution {
    fn default() -> Self {
        Self {
            enabled: false,
            min_scale: 0.5,
            max_scale: 1.0,
            budget_ms: None,
        }
    }
}

impl Section for DynamicResolution {
    const KEY: &'static str = "dynamic_resolution";

    fn check(&self) -> Result<(), Error> {
        range(0.25, 1.0).check(&self.min_scale, "dynamic_resolution.min_scale")?;
        range(0.25, 1.0).check(&self.max_scale, "dynamic_resolution.max_scale")?;
        range(1.0, 1000.0).check(&self.budget_ms, "dynamic_resolution.budget_ms")?;
        if self.min_scale > self.max_scale {
            return Err(
                Error::new(ErrorCode::InvalidValue, "dynamic_resolution.min_scale")
                    .expected(Expected::Sorted)
                    .got(self.min_scale)
                    .nearest(self.max_scale),
            );
        }
        Ok(())
    }
}

pub(crate) fn register(engine: &mut Engine) {
    engine
        .settings::<Resolution>()
        .settings::<DynamicResolution>();
}
