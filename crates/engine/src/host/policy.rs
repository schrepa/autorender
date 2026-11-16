macro_rules! policy {
    ($(pub const $name:ident: $ty:ty = $value:expr;)*) => {
        $(pub const $name: $ty = $value;)*

        #[cfg(all(feature = "schema", not(target_arch = "wasm32")))]
        pub(crate) fn constants() -> Vec<(&'static str, serde_json::Value)> {
            vec![$((stringify!($name), serde_json::json!($name))),*]
        }
    };
}

policy! {
    pub const FRAME_INTERVAL_MS: f64 = 1000.0 / 60.0;
    pub const FRAME_CORES: u32 = 2;
    pub const JOB_CACHE_MIB: u64 = 1024;
    pub const PIPELINE_CACHE_MIB: u64 = 256;
    pub const PACK_CACHE_MIB: u64 = 1024;
    pub const WHEEL_PIXELS: f32 = 100.0;
    pub const BENCHMARK_LIMIT_MS: u64 = 10_000;
    pub const READ_BACK_LIMIT_MS: u64 = 120_000;
}

pub fn default_workers(cores: u32) -> u32 {
    cores.saturating_sub(FRAME_CORES)
}
