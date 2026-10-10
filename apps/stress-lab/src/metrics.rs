use engine::prelude::*;
use serde::Serialize;

/// The last frame's CPU simulation subsets in milliseconds and the bounded
/// spatial query counters, every frame's totals as decimal strings.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct StressMetrics {
    pub sync_pre_ms: f64,
    pub sync_post_ms: f64,
    pub sensor_ms: f64,
    pub queries_ms: f64,
    pub animation_ms: f64,
    pub query_calls: u32,
    pub query_hits: u32,
    pub query_errors: u32,
    pub total_calls: String,
    pub total_hits: String,
    pub total_errors: String,
    /// The last query's error code.
    pub last_error: Option<String>,
}

impl From<&FrameMetrics> for StressMetrics {
    fn from(metrics: &FrameMetrics) -> Self {
        Self {
            sync_pre_ms: metrics.sync_pre_ms,
            sync_post_ms: metrics.sync_post_ms,
            sensor_ms: metrics.sensor_ms,
            queries_ms: metrics.queries_ms,
            animation_ms: metrics.animation_ms,
            query_calls: metrics.query_calls,
            query_hits: metrics.query_hits,
            query_errors: metrics.query_errors,
            total_calls: metrics.total_calls.to_string(),
            total_hits: metrics.total_hits.to_string(),
            total_errors: metrics.total_errors.to_string(),
            last_error: metrics.last_error.map(str::to_owned),
        }
    }
}

/// Demo-only wall-clock brackets, reset at each browser submission boundary.
#[derive(Resource, Default, Clone, Copy)]
pub struct FrameMetrics {
    pub sync_pre_ms: f64,
    pub sync_post_ms: f64,
    pub sensor_ms: f64,
    pub queries_ms: f64,
    pub animation_ms: f64,
    pub query_calls: u32,
    pub query_hits: u32,
    pub query_errors: u32,
    pub total_calls: u64,
    pub total_hits: u64,
    pub total_errors: u64,
    pub last_error: Option<&'static str>,
    started: f64,
}

impl FrameMetrics {
    pub fn begin_frame(&mut self) {
        self.sync_pre_ms = 0.0;
        self.sync_post_ms = 0.0;
        self.sensor_ms = 0.0;
        self.queries_ms = 0.0;
        self.animation_ms = 0.0;
        self.query_calls = 0;
        self.query_hits = 0;
        self.query_errors = 0;
    }

    pub(crate) fn query_result(
        &mut self,
        result: Result<usize, engine::collision::CollisionError>,
    ) {
        self.query_calls += 1;
        self.total_calls = self.total_calls.saturating_add(1);
        match result {
            Ok(hits) => {
                self.query_hits += hits as u32;
                self.total_hits = self.total_hits.saturating_add(hits as u64);
            }
            Err(error) => {
                self.query_errors += 1;
                self.total_errors = self.total_errors.saturating_add(1);
                self.last_error = Some(error.code().as_str());
            }
        }
    }
}

pub(crate) fn now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        thread_local! {
            static CLOCK: web_sys::Performance = web_sys::window()
                .and_then(|window| window.performance()).expect("browser performance clock");
        }
        CLOCK.with(web_sys::Performance::now)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        thread_local! { static START: std::time::Instant = std::time::Instant::now(); }
        START.with(|start| start.elapsed().as_secs_f64() * 1000.0)
    }
}

pub(crate) fn add_timers(app: &mut App) {
    app.init_resource::<FrameMetrics>()
        .add_systems(
            FixedPreSimulation,
            (
                begin.before(CollisionSystems::Sync),
                end_pre.after(CollisionSystems::Sync),
            ),
        )
        .add_systems(
            FixedPostSimulation,
            (
                begin.before(CollisionSystems::Sync),
                end_post
                    .before(CollisionSystems::Detect)
                    .after(CollisionSystems::Sync),
                end_sensor.after(CollisionSystems::Detect),
            ),
        );
}

fn begin(mut metrics: ResMut<FrameMetrics>) {
    metrics.started = now_ms();
}
fn end_pre(mut metrics: ResMut<FrameMetrics>) {
    metrics.sync_pre_ms += now_ms() - metrics.started;
}
fn end_post(mut metrics: ResMut<FrameMetrics>) {
    let now = now_ms();
    metrics.sync_post_ms += now - metrics.started;
    metrics.started = now;
}
fn end_sensor(mut metrics: ResMut<FrameMetrics>) {
    metrics.sensor_ms += now_ms() - metrics.started;
}
