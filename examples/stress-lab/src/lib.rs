//! Deterministic stress workloads built exclusively through the public engine SDK.
#[cfg(target_arch = "wasm32")]
mod browser;
mod city;
mod config;
mod dense;
mod highlands;
mod metrics;
mod metro;
mod rocks;
mod scene;
mod shapes;

pub use config::{StressConfig, StressScene};
pub use dense::FILES as DENSE_FILES;
pub use highlands::FILES as HIGHLANDS_FILES;
pub use metrics::FrameMetrics;
pub use scene::{StressFile, create_workload, create_workload_with, look, set_view, stress_shafts};

use engine::prelude::*;

// Applications opt into allocator accounting explicitly; the engine never
// replaces a consumer's allocator. Normal builds retain Rust's default allocator.
#[cfg(feature = "profiling")]
#[global_allocator]
static ALLOCATOR: engine::profiling::TrackingAllocator<std::alloc::System> =
    engine::profiling::TrackingAllocator::new(std::alloc::System);

/// What a recording samples of [`stress_observation`] every frame, without
/// its JSON: collision sync, sensor, query and animation milliseconds;
/// active, invalid and overflowing colliders and collision completeness;
/// sensor pairs, candidates and completeness; query calls, hits and errors.
/// Flags are 1 or 0.
pub fn stress_sample(engine: &Engine) -> [f64; 14] {
    use engine::collision::SensorEvents;
    let world = engine.app.world();
    let collision = world.get_resource::<CollisionWorld>();
    let report = collision.map(CollisionWorld::report);
    let events = world.get_resource::<SensorEvents>();
    let metrics = world.resource::<FrameMetrics>();
    let flag = |value: bool| f64::from(u8::from(value));
    [
        metrics.sync_pre_ms + metrics.sync_post_ms,
        metrics.sensor_ms,
        metrics.queries_ms,
        metrics.animation_ms,
        report.map_or(0.0, |report| report.active_colliders as f64),
        report.map_or(0.0, |report| report.invalid_colliders as f64),
        report.map_or(0.0, |report| report.overflow_colliders as f64),
        flag(report.is_none_or(|report| report.complete)),
        events.map_or(0.0, |events| events.pairs().len() as f64),
        events.map_or(0.0, |events| events.candidate_pairs() as f64),
        flag(events.is_none_or(SensorEvents::complete)),
        metrics.query_calls as f64,
        metrics.query_hits as f64,
        metrics.query_errors as f64,
    ]
}

/// Small observations are serialized only when the host polls, never per entity.
pub fn stress_observation(engine: &Engine, generation: u32) -> serde_json::Value {
    use engine::collision::{CollisionReport, SensorEvents};
    let world = engine.app.world();
    let config = world.resource::<StressConfig>();
    let workload = world.resource::<scene::Workload>();
    let collision = world.get_resource::<CollisionWorld>();
    let report = collision.map_or(
        CollisionReport {
            complete: true,
            ..CollisionReport::default()
        },
        CollisionWorld::report,
    );
    let events = world.get_resource::<SensorEvents>();
    let assets = world.resource::<AssetLibrary>();
    let history = world.resource::<PresentationState>();
    let metrics = world.resource::<FrameMetrics>();
    let camera = world.resource::<Camera>();
    serde_json::json!({
        "version": 1, "generation": generation, "config": config,
        "frame": world.resource::<FrameTime>().frame.to_string(),
        "tick": world.resource::<FixedTime>().tick.to_string(),
        "renderables": config.renderables, "transform_nodes": workload.transform_nodes,
        "moving_nodes": workload.moving_nodes, "sensor_nodes": workload.sensor_nodes,
        "collision": {
            "enabled": collision.is_some(),
            "revision": report.revision.to_string(), "tick": report.tick.to_string(),
            "colliders": report.colliders, "active_colliders": report.active_colliders,
            "invalid_colliders": report.invalid_colliders, "overflow_colliders": report.overflow_colliders,
            "bvh_nodes": report.bvh_nodes, "cache_bytes": report.cache_bytes, "complete": report.complete,
            "error": collision.and_then(CollisionWorld::last_error).map(|error| error.code()),
            "sensor_pairs": events.map_or(0, |events| events.pairs().len()),
            "sensor_events": events.map_or(0, |events| events.events().len()),
            "sensor_complete": events.is_none_or(SensorEvents::complete),
            "sensor_candidates": events.map_or(0, SensorEvents::candidate_pairs),
            "sensor_error": events.and_then(SensorEvents::error).map(|error| error.code()),
            "sensor_attempted_tick": events.and_then(SensorEvents::tick).map(|tick| tick.to_string()),
            "sensor_committed_tick": events.and_then(SensorEvents::committed_tick).map(|tick| tick.to_string()),
        },
        "timings": {
            "sync_pre_ms": metrics.sync_pre_ms, "sync_post_ms": metrics.sync_post_ms,
            "sensor_ms": metrics.sensor_ms, "queries_ms": metrics.queries_ms,
            "animation_ms": metrics.animation_ms,
            "scope": "CPU subsets of simulation for the last submitted frame; not additive to simulation",
        },
        "queries": {
            "calls": metrics.query_calls, "hits": metrics.query_hits, "errors": metrics.query_errors,
            "total_calls": metrics.total_calls.to_string(), "total_hits": metrics.total_hits.to_string(),
            "total_errors": metrics.total_errors.to_string(), "last_error": metrics.last_error,
        },
        "assets": {
            "meshes": assets.mesh_count(), "materials": assets.material_count(),
            "textures": assets.texture_count(), "cpu_bytes": assets.resident_bytes(),
            "texture_bytes": assets.texture_bytes(),
        },
        "interpolation": {
            "history_nodes": history.history_nodes(), "history_bytes": history.history_bytes(),
            "presented_nodes": history.presented_nodes(),
            "presented_store_bytes": history.presented_store_bytes(),
            "scratch_bytes": history.presentation_scratch_bytes(),
        },
        "camera": { "position": camera.position.to_array(), "target": camera.target.to_array(),
            "vertical_fov_radians": camera.vertical_fov_radians, "near": camera.near, "far": camera.far },
    })
}
