use engine_core::ActionId;

macro_rules! policy {
    ($($(#[doc = $doc:literal])+ pub const $name:ident: $ty:ty = $value:expr;)*) => {
        $($(#[doc = $doc])+ pub const $name: $ty = $value;)*

        #[cfg(all(feature = "schema", not(target_arch = "wasm32")))]
        pub(crate) fn constants() -> Vec<(&'static str, String, serde_json::Value)> {
            vec![$((
                stringify!($name),
                [$($doc.trim()),+].join("\n"),
                serde_json::json!($name),
            )),*]
        }
    };
}

policy! {
    pub const QUIET_FRAMES: u32 = 3;
    pub const PREPARE_LIMIT_MS: u64 = 60_000;
    pub const FRAME_INTERVAL_MS: f64 = 1000.0 / 60.0;
    pub const PACE_SMOOTHING: f64 = 0.1;
    pub const PACE_INTERVAL_LIMIT_MS: f64 = 100.0;
    pub const BETWEEN_MARGIN_MS: f64 = 4.0;
    pub const FRAME_CORES: u32 = 2;
    pub const MAX_WORKERS: u32 = 32;
    pub const JOB_CACHE_MIB: u64 = 1024;
    pub const MAX_JOB_CACHE_MIB: u64 = 9_999_999;
    pub const MAX_ACTIONS: u32 = ActionId::COUNT;
    pub const LOOK_LIMIT: f32 = 1000.0;
    pub const BENCHMARK_LIMIT_MS: u64 = 10_000;
    pub const MAX_PIXEL_RATIO: f64 = 1.5;
    pub const MAX_VIEW_WIDTH: u32 = 2560;
    pub const MAX_VIEW_HEIGHT: u32 = 1440;
    pub const DYNAMIC_STEP: f64 = 1.0 / 32.0;
    pub const DYNAMIC_OVER: f64 = 1.05;
    pub const DYNAMIC_UNDER: f64 = 0.85;
    pub const DYNAMIC_AIM: f64 = 0.9;
    pub const DYNAMIC_LOWER_FRAMES: u32 = 3;
    pub const DYNAMIC_RAISE_FRAMES: u32 = 30;
    pub const DYNAMIC_PROBE_FRAMES: u32 = 120;
}

pub fn default_workers(cores: u32) -> u32 {
    cores.saturating_sub(FRAME_CORES).clamp(1, MAX_WORKERS)
}

pub fn cook_workers(cores: u32) -> u32 {
    cores.clamp(1, MAX_WORKERS)
}
