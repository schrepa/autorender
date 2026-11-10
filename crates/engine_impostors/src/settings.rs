use engine_core::{Cut, settings, stats};

settings! {
    #[section("impostors")]
    #[switch(off)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneImpostors")]
    pub struct ImpostorSettings {
        pub pixels: f32 = 128.0 => range(8.0, 4096.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max: Option<u32> = None => budget,
        pub work: u32 = 64 => at_least(1),
        pub min_triangles: u32 = 64 => any,
        pub cutoff: f32 = 0.5 => range(0.0, 1.0),
    }
}

stats! {
    #[section("impostors_stats")]
    pub struct ImpostorStats {
        pub failed: u32,
        pub mesh_impostors: u32,
        pub mesh_cut: Cut,
    }
}
