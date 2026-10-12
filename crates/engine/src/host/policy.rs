//! The values both hosts' policies take, declared once: the page reads them
//! from the generated `web/src/generated/host.ts`, the native host from
//! here.

use engine_core::ActionId;

/// Declares each policy value once: a constant, and an entry of
/// `constants`, which [`super::typescript`] writes for the web host.
macro_rules! policy {
    ($($(#[doc = $doc:literal])+ pub const $name:ident: $ty:ty = $value:expr;)*) => {
        $($(#[doc = $doc])+ pub const $name: $ty = $value;)*

        /// Each value's name, what it is and the value, as JSON.
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
    /// Frames in a row that create no pipeline before play starts, once
    /// the terrains loaded the view: the GPU has compiled what the first
    /// frames need.
    pub const QUIET_FRAMES: u32 = 3;
    /// Milliseconds after which play starts however preparation stands.
    pub const PREPARE_LIMIT_MS: u64 = 60_000;
    /// The interval between frames pacing assumes before it measured one.
    pub const FRAME_INTERVAL_MS: f64 = 1000.0 / 60.0;
    /// The share of each new interval pacing's running interval takes.
    pub const PACE_SMOOTHING: f64 = 0.1;
    /// Intervals this long or longer (a pause, a hidden page) leave pacing
    /// alone.
    pub const PACE_INTERVAL_LIMIT_MS: f64 = 100.0;
    /// Milliseconds loading between frames leaves before the next is due.
    pub const BETWEEN_MARGIN_MS: f64 = 4.0;
    /// Milliseconds a job worker runs a job before it looks at its messages
    /// and more urgent jobs again.
    pub const JOB_SLICE_MS: u64 = 8;
    /// Work units a job worker asks a job to do at a time, about one a
    /// sample.
    pub const JOB_STEP_UNITS: u64 = 4096;
    /// Logical processors per job worker a host runs by default.
    pub const CORES_PER_DEFAULT_WORKER: u32 = 2;
    /// The most job workers a host runs by default.
    pub const MAX_DEFAULT_WORKERS: u32 = 8;
    /// The most job workers a host runs.
    pub const MAX_WORKERS: u32 = 32;
    /// MiB of job outputs a host keeps by default.
    pub const JOB_CACHE_MIB: u64 = 1024;
    /// The most MiB of job outputs a host keeps.
    pub const MAX_JOB_CACHE_MIB: u64 = 9_999_999;
    /// Actions a game binds keys to, by index.
    pub const MAX_ACTIONS: u32 = ActionId::COUNT;
    /// The largest look delta a frame takes, either way, in pixels.
    pub const LOOK_LIMIT: f32 = 1000.0;
    /// Milliseconds a development benchmark may take.
    pub const BENCHMARK_LIMIT_MS: u64 = 10_000;
}

/// The job workers a host runs by default on `cores` logical processors:
/// one per [`CORES_PER_DEFAULT_WORKER`], at least one and at most
/// [`MAX_DEFAULT_WORKERS`].
pub fn default_workers(cores: u32) -> u32 {
    (cores / CORES_PER_DEFAULT_WORKER).clamp(1, MAX_DEFAULT_WORKERS)
}
