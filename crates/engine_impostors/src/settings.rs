use engine_core::{settings, stats};

settings! {
    #[section("impostors")]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneImpostors")]
    pub struct ImpostorSettings {
        pub enabled: bool = false => any,
        pub pixels: f32 = 128.0 => range(8.0, 4096.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max: Option<u32> = None => budget,
        pub work: u32 = 64 => at_least(1),
    }
}

stats! {
    #[section("impostors_stats")]
    pub struct ImpostorStats {
        pub failed: u32,
    }
}
