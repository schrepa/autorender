use engine_core::{Engine, settings};

settings! {
    #[section("resolution")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneResolution")]
    pub struct Resolution {
        pub scale: f32 = 1.0 => range(0.25, 2.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_pixel_ratio: Option<f32> = None => above(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_width: Option<u32> = None => at_least(1),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_height: Option<u32> = None => at_least(1),
    }
}

pub(crate) fn register(engine: &mut Engine) {
    engine.settings::<Resolution>();
}
