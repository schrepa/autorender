use engine_core::settings;

pub(crate) const OFF: &str = "off";

settings! {
    #[section("debug_view")]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneDebugView")]
    pub struct DebugView {
        pub view: String = String::from(OFF) => any,
        pub shade: bool = true => any,
    }
}

impl DebugView {
    pub fn showing(&self) -> bool {
        self.view != OFF
    }
}
