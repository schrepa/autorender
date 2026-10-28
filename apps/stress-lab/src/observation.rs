use engine::collision::{CollisionReport, SensorEvents};
use engine::prelude::*;
use serde::Serialize;

use crate::scene::Workload;
use crate::{FrameMetrics, StressConfig};

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct StressSnapshot {
    pub version: u32,
    pub generation: u32,
    pub config: StressConfig,
    pub frame: String,
    pub tick: String,
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

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CollisionSnapshot {
    pub enabled: bool,
    pub revision: String,
    pub tick: String,
    pub colliders: usize,
    pub active_colliders: usize,
    pub invalid_colliders: usize,
    pub overflow_colliders: usize,
    pub bvh_nodes: usize,
    pub cache_bytes: usize,
    pub complete: bool,
    pub error: Option<String>,
    pub sensor_pairs: usize,
    pub sensor_events: usize,
    pub sensor_complete: bool,
    pub sensor_candidates: usize,
    pub sensor_error: Option<String>,
    pub sensor_attempted_tick: Option<String>,
    pub sensor_committed_tick: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TimingSnapshot {
    pub sync_pre_ms: f64,
    pub sync_post_ms: f64,
    pub sensor_ms: f64,
    pub queries_ms: f64,
    pub animation_ms: f64,
    pub scope: String,
}

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

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AssetSnapshot {
    pub meshes: usize,
    pub materials: usize,
    pub textures: usize,
    pub cpu_bytes: usize,
    pub texture_bytes: usize,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct InterpolationSnapshot {
    pub history_nodes: usize,
    pub history_bytes: usize,
    pub presented_nodes: usize,
    pub presented_store_bytes: usize,
    pub scratch_bytes: usize,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CameraSnapshot {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub vertical_fov_radians: f32,
    pub near: f32,
    pub far: f32,
}

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

macro_rules! sample {
    ($(#[$meta:meta])* pub struct $name:ident { $($(#[$field_meta:meta])* $field:ident),* $(,)? }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name {
            $($(#[$field_meta])* pub $field: f64,)*
        }

        impl $name {
            pub const FIELDS: &'static [&'static str] = &[$(stringify!($field)),*];

            pub fn values(&self) -> Vec<f64> {
                vec![$(self.$field),*]
            }
        }
    };
}

sample! {
    pub struct StressSample {
        sync_ms,
        sensor_ms,
        queries_ms,
        animation_ms,
        active_colliders,
        invalid_colliders,
        overflow_colliders,
        collision_complete,
        sensor_pairs,
        sensor_candidates,
        sensor_complete,
        query_calls,
        query_hits,
        query_errors,
    }
}

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
