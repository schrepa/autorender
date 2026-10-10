//! What the page reads of a workload: the snapshot it polls and the numbers
//! a recording samples every frame, declared once so the page's types,
//! menus and sample names are generated from them (`cargo xtask schema`).

use engine::collision::{CollisionReport, SensorEvents};
use engine::prelude::*;
use serde::Serialize;

use crate::scene::Workload;
use crate::{FrameMetrics, StressConfig};

/// A workload as the page polls it, serialized only when it polls.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct StressSnapshot {
    /// The snapshot format's version.
    pub version: u32,
    /// The workload's generation: one more each time a configuration
    /// rebuilds it.
    pub generation: u32,
    /// The configuration the workload was built from.
    pub config: StressConfig,
    /// The frame and fixed tick, as decimal strings.
    pub frame: String,
    pub tick: String,
    /// Renderable objects, transform nodes, and the moving and sensor ones.
    pub renderables: u32,
    pub transform_nodes: u32,
    pub moving_nodes: u32,
    pub sensor_nodes: u32,
    pub collision: CollisionSnapshot,
    pub timings: TimingSnapshot,
    pub queries: QuerySnapshot,
    pub assets: AssetSnapshot,
    pub interpolation: InterpolationSnapshot,
    pub camera: CameraSnapshot,
}

/// Collision and sensors: counts, completeness and errors.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CollisionSnapshot {
    /// Whether the workload has collision at all.
    pub enabled: bool,
    /// The collision world's revision and tick, as decimal strings.
    pub revision: String,
    pub tick: String,
    pub colliders: usize,
    pub active_colliders: usize,
    pub invalid_colliders: usize,
    pub overflow_colliders: usize,
    pub bvh_nodes: usize,
    pub cache_bytes: usize,
    /// Whether every collider entered the world.
    pub complete: bool,
    /// The collision world's last error code.
    pub error: Option<String>,
    pub sensor_pairs: usize,
    pub sensor_events: usize,
    /// Whether sensor detection covered every candidate pair.
    pub sensor_complete: bool,
    pub sensor_candidates: usize,
    /// Sensor detection's error code.
    pub sensor_error: Option<String>,
    /// The ticks sensor detection last ran and last committed, as decimal
    /// strings.
    pub sensor_attempted_tick: Option<String>,
    pub sensor_committed_tick: Option<String>,
}

/// CPU subsets of simulation for the last submitted frame, in
/// milliseconds; not additive to simulation.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TimingSnapshot {
    pub sync_pre_ms: f64,
    pub sync_post_ms: f64,
    pub sensor_ms: f64,
    pub queries_ms: f64,
    pub animation_ms: f64,
    /// What the timings cover.
    pub scope: String,
}

/// Spatial queries: the last frame's and every frame's, totals as decimal
/// strings.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct QuerySnapshot {
    pub calls: u32,
    pub hits: u32,
    pub errors: u32,
    pub total_calls: String,
    pub total_hits: String,
    pub total_errors: String,
    pub last_error: Option<String>,
}

/// The asset library: counts and resident CPU bytes.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AssetSnapshot {
    pub meshes: usize,
    pub materials: usize,
    pub textures: usize,
    pub cpu_bytes: usize,
    pub texture_bytes: usize,
}

/// Interpolation history and presented poses: nodes and bytes.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct InterpolationSnapshot {
    pub history_nodes: usize,
    pub history_bytes: usize,
    pub presented_nodes: usize,
    pub presented_store_bytes: usize,
    pub scratch_bytes: usize,
}

/// The camera: eye, target, vertical field of view and depth range.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CameraSnapshot {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub vertical_fov_radians: f32,
    pub near: f32,
    pub far: f32,
}

/// The workload's snapshot.
pub fn stress_observation(engine: &Engine, generation: u32) -> StressSnapshot {
    let world = engine.app.world();
    let config = world.resource::<StressConfig>();
    let workload = world.resource::<Workload>();
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
    StressSnapshot {
        version: 1,
        generation,
        // The culling development controls change too.
        config: StressConfig {
            culling: engine.frustum_culling(),
            ..config.clone()
        },
        frame: world.resource::<FrameTime>().frame.to_string(),
        tick: world.resource::<FixedTime>().tick.to_string(),
        renderables: config.renderables,
        transform_nodes: workload.transform_nodes,
        moving_nodes: workload.moving_nodes,
        sensor_nodes: workload.sensor_nodes,
        collision: CollisionSnapshot {
            enabled: collision.is_some(),
            revision: report.revision.to_string(),
            tick: report.tick.to_string(),
            colliders: report.colliders,
            active_colliders: report.active_colliders,
            invalid_colliders: report.invalid_colliders,
            overflow_colliders: report.overflow_colliders,
            bvh_nodes: report.bvh_nodes,
            cache_bytes: report.cache_bytes,
            complete: report.complete,
            error: collision
                .and_then(CollisionWorld::last_error)
                .map(|error| error.code().as_str().to_owned()),
            sensor_pairs: events.map_or(0, |events| events.pairs().len()),
            sensor_events: events.map_or(0, |events| events.events().len()),
            sensor_complete: events.is_none_or(SensorEvents::complete),
            sensor_candidates: events.map_or(0, SensorEvents::candidate_pairs),
            sensor_error: events
                .and_then(SensorEvents::error)
                .map(|error| error.code().as_str().to_owned()),
            sensor_attempted_tick: events
                .and_then(SensorEvents::tick)
                .map(|tick| tick.to_string()),
            sensor_committed_tick: events
                .and_then(SensorEvents::committed_tick)
                .map(|tick| tick.to_string()),
        },
        timings: TimingSnapshot {
            sync_pre_ms: metrics.sync_pre_ms,
            sync_post_ms: metrics.sync_post_ms,
            sensor_ms: metrics.sensor_ms,
            queries_ms: metrics.queries_ms,
            animation_ms: metrics.animation_ms,
            scope:
                "CPU subsets of simulation for the last submitted frame; not additive to simulation"
                    .into(),
        },
        queries: QuerySnapshot {
            calls: metrics.query_calls,
            hits: metrics.query_hits,
            errors: metrics.query_errors,
            total_calls: metrics.total_calls.to_string(),
            total_hits: metrics.total_hits.to_string(),
            total_errors: metrics.total_errors.to_string(),
            last_error: metrics.last_error.map(str::to_owned),
        },
        assets: AssetSnapshot {
            meshes: assets.mesh_count(),
            materials: assets.material_count(),
            textures: assets.texture_count(),
            cpu_bytes: assets.resident_bytes(),
            texture_bytes: assets.texture_bytes(),
        },
        interpolation: InterpolationSnapshot {
            history_nodes: history.history_nodes(),
            history_bytes: history.history_bytes(),
            presented_nodes: history.presented_nodes(),
            presented_store_bytes: history.presented_store_bytes(),
            scratch_bytes: history.presentation_scratch_bytes(),
        },
        camera: CameraSnapshot {
            position: camera.position.to_array(),
            target: camera.target.to_array(),
            vertical_fov_radians: camera.vertical_fov_radians,
            near: camera.near,
            far: camera.far,
        },
    }
}

/// Declares the per-frame sample and its field names from one list, so the
/// page reads each number by its name.
macro_rules! sample {
    ($(#[$meta:meta])* pub struct $name:ident { $($(#[$field_meta:meta])* $field:ident),* $(,)? }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name {
            $($(#[$field_meta])* pub $field: f64,)*
        }

        impl $name {
            /// The fields' names, in the order of [`Self::values`].
            pub const FIELDS: &'static [&'static str] = &[$(stringify!($field)),*];

            /// The fields' values, in the order of [`Self::FIELDS`].
            pub fn values(&self) -> Vec<f64> {
                vec![$(self.$field),*]
            }
        }
    };
}

sample! {
    /// What a recording samples of the snapshot every frame, without its
    /// JSON. Flags are 1 or 0.
    pub struct StressSample {
        /// Collision sync, sensor, query and animation milliseconds.
        sync_ms,
        sensor_ms,
        queries_ms,
        animation_ms,
        /// Active, invalid and overflowing colliders, and whether every
        /// collider entered the world.
        active_colliders,
        invalid_colliders,
        overflow_colliders,
        collision_complete,
        /// Sensor pairs and candidates, and whether detection covered them.
        sensor_pairs,
        sensor_candidates,
        sensor_complete,
        /// The last frame's query calls, hits and errors.
        query_calls,
        query_hits,
        query_errors,
    }
}

/// The workload's sample this frame.
pub fn stress_sample(engine: &Engine) -> StressSample {
    let world = engine.app.world();
    let report = world
        .get_resource::<CollisionWorld>()
        .map(CollisionWorld::report);
    let events = world.get_resource::<SensorEvents>();
    let metrics = world.resource::<FrameMetrics>();
    let flag = |value: bool| f64::from(u8::from(value));
    StressSample {
        sync_ms: metrics.sync_pre_ms + metrics.sync_post_ms,
        sensor_ms: metrics.sensor_ms,
        queries_ms: metrics.queries_ms,
        animation_ms: metrics.animation_ms,
        active_colliders: report.map_or(0.0, |report| report.active_colliders as f64),
        invalid_colliders: report.map_or(0.0, |report| report.invalid_colliders as f64),
        overflow_colliders: report.map_or(0.0, |report| report.overflow_colliders as f64),
        collision_complete: flag(report.is_none_or(|report| report.complete)),
        sensor_pairs: events.map_or(0.0, |events| events.pairs().len() as f64),
        sensor_candidates: events.map_or(0.0, |events| events.candidate_pairs() as f64),
        sensor_complete: flag(events.is_none_or(SensorEvents::complete)),
        query_calls: f64::from(metrics.query_calls),
        query_hits: f64::from(metrics.query_hits),
        query_errors: f64::from(metrics.query_errors),
    }
}
