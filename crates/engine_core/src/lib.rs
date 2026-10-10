//! Platform-independent gameplay, fixed scheduling, and owned render extraction.
//!
//! The SDK explicitly uses Bevy ECS and Bevy's [`Plugin`] contract. Register
//! systems in [`FixedSimulation`] or [`Presentation`], then drive [`Engine`]
//! from the platform event loop. Bevy's default `Main` schedules are not run.

mod animation;
mod antialiasing;
mod atmosphere;
mod bridges;
mod clock;
mod clouds;
mod commands;
mod contact_shadows;
mod culling;
mod decals;
mod deformation;
mod effects;
mod entity_indices;
mod environment;
mod extraction;
mod frame_stats;
mod gpu_culling;
mod ground_cover;
mod grounded;
mod hierarchy;
mod input;
mod interpolation;
mod jobs;
mod level_of_detail;
mod lights;
pub mod math;
mod navigation;
mod occupancy;
mod ocean;
mod packed;
mod particles;
mod probes;
mod retained_render;
mod scene;
mod skin;
mod sky;
mod spline_mesh;
mod streaming;
mod terrain;
mod transform_edits;
mod virtual_geometry;
mod water;
mod waterfalls;
mod wind;

pub use animation::{AnimationPlayer, AnimationSystems};
pub use antialiasing::TemporalAntialiasing;
pub use atmosphere::Atmosphere;
#[cfg(feature = "schema")]
pub use atmosphere::state_atmosphere_rules;
pub use bevy_app::{App, Plugin};
pub use bevy_ecs;
pub use bridges::{DECK_DEPTH, DECK_MARGIN, DECK_RISE};
pub use clock::{EngineConfig, FixedTime, FrameReport, FrameTime, PresentedTime};
pub use clouds::Clouds;
pub use commands::EngineEntityCommands;
pub use contact_shadows::ContactShadows;
pub use decals::{Decal, DecalBlend, DecalSettings, RenderDecal};
pub use deformation::{
    DEFORMATION_GRAIN, Deformation, DeformationMap, Deformer, MAX_TRACK_DEPTH, RIM_REACH,
    RIM_START, RenderDeformation, Track, TrackDepth, Tracks, carve_profile, rim_profile,
};
pub use effects::{
    AmbientOcclusion, Bloom, MAX_KEPT_LEVELS, MAX_SHADOW_CASCADES, Reflections, Refraction,
    Shadows, SunShafts,
};
pub use engine_assets::{
    Aabb, AlphaMode, AssetBudget, AssetError, AssetLibrary, Filter, Grounding, LodLevel,
    MAX_TEXTURE_DIMENSION, Material, MaterialBinding, MaterialHandle, Mesh, MeshHandle, MeshLods,
    MeshVertex, Sampler, Surface, Texture, TextureFormat, TextureHandle, TextureRef, TopLayer,
    Transmission, UvTransform, Wrap, mesh_lods,
};
pub use engine_assets::{
    AnimatedProperty, AnimationChannel, AnimationClip, AnimationSample, ClipHandle,
    KeyInterpolation, Keyframes,
};
pub use engine_assets::{
    CLUSTER_TRIANGLES, CLUSTER_VERTICES, CLUSTER_WORDS, DEFAULT_FALLBACK_TRIANGLES, GROUP_WORDS,
    NODE_WORDS, PAGE_WORDS, VIRTUAL_MESH_MAGIC, VirtualMesh,
};
pub use engine_settings::ErrorCode;
/// Heightfield terrain data: tile pyramids, layer stacks,
/// tiles, patches and the `ETR1` format.
pub use engine_terrain as terrain_data;
pub use engine_terrain::{
    AUTHORED_LAYER, AuthoredPlacement, Biome, BiomeError, BiomeLayer, BiomeRules, Biomes, Blend,
    Control, Erosion, ErosionLayer, ErosionStage, ErosionStats, FarPlacement, FileHeader,
    HeightImage, HeightmapLayer, Hydrology, HydrologyLayer, HydrologySample, HydrologyStage,
    HydrologyStats, Layer, LayerError, Layers, Mask, NoiseLayer, NoiseMask, Patch, PathLayer,
    PathProfile, PathStage, PathStats, Placed, PlacementChoice, PlacementLayer, Pyramid,
    PyramidError, Remap, Route, RouteOptions, Stamp, StampOp, Tile, TileKey, WaterGrid,
};
pub use environment::{Environment, EnvironmentSection, InvalidSetting};
pub use frame_stats::{
    Applied, CloudStats, CullingStats, DeformationStats, DetailStats, Drawn, FrameStats,
    GroundCoverStats, LightStats, OccupancyStats, OceanStats, ProbeStats, RenderPasses,
    RenderStats, ShadowStats, SkyStats, TerrainRenderStats, VirtualGeometryStats, WaterStats,
};
pub use glam;
pub use gpu_culling::{OcclusionCulling, SubpixelCulling};
pub use ground_cover::{Cover, GroundCover};
pub use grounded::{Grounded, GroundedAt};
pub use hierarchy::{
    GlobalTransform, GlobalTransforms, HierarchyError, HierarchyReport, MAX_HIERARCHY_DEPTH,
    MAX_NORMAL_INVERSE_NORM, MAX_TRANSFORM_NODES, Parent, TransformIndex, TransformNodeLimit,
    propagate_transforms, set_parent, set_parents, valid_global_matrix,
};
pub use input::{ActionId, ActionInput};
pub use interpolation::{
    GpuCurrent, GpuPresentation, InterpolateTransform, InterpolationError, NO_PARENT, NODE_ACTIVE,
    NODE_BLEND, NODE_PARENT_ACTIVE, NODE_VALID, NodeInfo, NodePose, PresentationMode,
    PresentationState, PresentedTransform, PresentedTransforms, reset_all_interpolation,
    reset_interpolation,
};
pub use jobs::{
    GAMEPLAY_JOBS, IDLE_JOBS, JobId, JobRequest, JobShare, JobStats, Jobs, URGENT_JOBS, VIEW_JOBS,
    content_hash, job_class, job_priority,
};
pub use level_of_detail::LevelOfDetail;
pub use lights::{
    LocalLight, LocalLightSettings, MAX_LIGHT_RANGE, MAX_SPOT_DEGREES, MIN_SPOT_DEGREES,
    RenderLight, Spot,
};
pub use navigation::{NavAgent, NavAgents, NavStats, NavStatus, NavSystems, NavWalk, Navigation};
pub use occupancy::Occupancy;
pub use ocean::{
    Floating, MAX_OCEAN_WAVES, Ocean, OceanPoint, OceanSurface, OceanSystems, OceanWave, OceanWaves,
};
pub use particles::{
    MAX_EMITTER_PARTICLES, MAX_SPREAD_DEGREES, ParticleBlend, ParticleEmitter, ParticleSettings,
    RenderEmitter,
};
pub use probes::{Interior, ReflectionProbe, ReflectionProbeSettings, RenderReflectionProbe};
pub use retained_render::{PoseRows, RenderFrame};
pub use scene::{
    Camera, Cube, InputState, Mesh3d, RenderInstance, RenderMaterial, RenderMesh, RenderSnapshot,
    RenderTexture, Tint, Transform,
};
pub use skin::{RenderJoint, Skin};
pub use sky::{Sky, TimeOfDay};
pub use spline_mesh::{CHUNK_LENGTH, MAX_SPLINE_POINTS, SplineChunks, SplineMesh};
pub use streaming::{
    CellStats, CellView, STREAMING_GRIDS, StreamSet, Streamed, StreamedId, StreamedItem,
    StreamingGrid, WorldCells, WorldChanges, WorldStreaming, remove_streamed,
};
pub use terrain::{
    MAX_COLLIDER_EXTENT, MAX_PLACEMENT_LODS, MIN_COLLIDER_EXTENT, MotionDetail, PlacedCollider,
    PlacedShape, PlacementAsset, PlacementLod, RenderPlacementAsset, RenderSurfaceMaps,
    RenderTerrain, RenderTerrains, SurfaceMaterial, SurfaceProjection, Surroundings,
    SurroundingsStatus, Terrain, TerrainBakes, TerrainDebug, TerrainEdit, TerrainFocus,
    TerrainPrefetch, TerrainSample, TerrainSettings, TerrainSource, TerrainStats, TerrainSurface,
    TerrainView, Terrains, TileRequest, Viewport,
};
pub use transform_edits::{TransformEditError, edit_transform};
pub use virtual_geometry::VirtualGeometry;
pub use water::Water;
pub use wind::Wind;

use bevy_app::PluginsState;
use bevy_ecs::{
    prelude::*,
    schedule::{ScheduleLabel, SingleThreadedExecutor},
};
use clock::FixedClock;
use interpolation::InterpolationState;

type RenderQuery = QueryState<
    (
        Entity,
        Has<Transform>,
        Option<Ref<'static, Cube>>,
        Option<Ref<'static, Mesh3d>>,
        Option<Ref<'static, Tint>>,
    ),
    Or<(With<Cube>, With<Mesh3d>)>,
>;

/// Simulation phase. Commands are applied at the end of every fixed tick.
#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixedSimulation;

/// Once per fixed tick before gameplay, after authoritative transform refresh.
#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixedPreSimulation;

/// Once per fixed tick after gameplay commands and authoritative propagation.
/// Commands here apply before final propagation and interpolation history capture.
#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixedPostSimulation;

/// Once-per-frame phase, after all fixed ticks and before render extraction.
#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Presentation;

/// Engine-owned derived-transform boundary after fixed and presentation work.
///
/// Gameplay registers in [`FixedSimulation`] and [`Presentation`]. This schedule
/// refreshes matrices only, including extraction while paused.
#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransformPropagation;

/// Once per rendered frame before its transforms refresh: the world follows
/// the camera. Streamed cells load and unload, grounded entities stand on
/// the ground, and bridges, waterfalls and spline meshes spawn and build, in
/// [`WorldSyncSystems`] order; game systems order against those sets. Paused
/// frames run it too; GPU frames presenting the interval before a tick in
/// flight do not.
#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorldSync;

/// Once per rendered frame after its transforms refresh, before extraction:
/// what follows the camera's view. Terrains stream toward it and the
/// deformation map follows it, in [`ViewSyncSystems`] order. Transforms
/// changed here propagate at the next boundary.
#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ViewSync;

/// Once per fixed tick after its final propagation, reading the tick's
/// settled transforms: deformers press tracks where the tick left them
/// ([`FixedTickEndSystems`]). Transforms changed here propagate at the next
/// boundary.
#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixedTickEnd;

/// The engine's work in [`WorldSync`], in this order.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorldSyncSystems {
    StreamCells,
    Ground,
    Bridges,
    Waterfalls,
    Splines,
}

/// The engine's work in [`ViewSync`], in this order.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewSyncSystems {
    StreamTerrains,
    FollowDeformation,
}

/// The engine's work in [`FixedTickEnd`].
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FixedTickEndSystems {
    PressTracks,
}

/// The most fixed ticks one development step request advances.
pub const MAX_DEV_TICKS: u32 = 120;

/// An invalid development step request: its ticks lie outside 1 to
/// [`MAX_DEV_TICKS`].
#[cfg(feature = "dev")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevStepError {
    /// Rejected tick count. Validation occurs before any engine mutation.
    pub ticks: u32,
}

#[cfg(feature = "dev")]
impl DevStepError {
    /// Maximum fixed ticks in one synchronous development request.
    pub const MAX_TICKS: u32 = MAX_DEV_TICKS;

    /// Check bounds before preparing inputs or changing a development session.
    pub fn validate(ticks: u32) -> Result<(), Self> {
        if (1..=Self::MAX_TICKS).contains(&ticks) {
            Ok(())
        } else {
            Err(Self { ticks })
        }
    }
}

#[cfg(feature = "dev")]
impl std::fmt::Display for DevStepError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Development steps require 1..={} ticks (received {}).",
            Self::MAX_TICKS,
            self.ticks
        )
    }
}

#[cfg(feature = "dev")]
impl std::error::Error for DevStepError {}

/// Owns gameplay state and scheduling, with no browser or GPU dependencies.
pub struct Engine {
    /// Supported extension surface: add plugins, resources, entities and systems.
    ///
    /// Register plugins before the first simulation frame. Keep this app's world and
    /// engine resources intact; replacing the world invalidates cached queries.
    pub app: App,
    clock: FixedClock,
    render_query: RenderQuery,
    retained: retained_render::RetainedRender,
    // Local lights and particle emitters of
    // each extraction.
    lights: lights::LightExtraction,
    particles: particles::ParticleExtraction,
    skins: skin::SkinExtraction,
    decals: decals::DecalExtraction,
    probes: probes::ProbeExtraction,
    plugins_finalized: bool,
    tick_ahead: TickAhead,
}

/// The fixed tick [`Engine::run_ahead`] ran ahead of its due frame, if any:
/// at most one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TickAhead {
    None,
    /// Its first stage ran; its second runs when it is due or completes early.
    InFlight,
    /// Completed before its due frame, which the clock counts it in: GPU
    /// frames keep presenting the previous interval meanwhile. `gathered`:
    /// its GPU inputs were gathered when it completed.
    Hidden {
        gathered: bool,
    },
    /// Completed before its due frame, and presented.
    Revealed,
}

impl TickAhead {
    /// While GPU frames present the previous interval (a tick in flight, or
    /// completed and hidden): whether a hidden tick's inputs were gathered.
    fn presents_previous(self) -> Option<bool> {
        match self {
            Self::InFlight => Some(false),
            Self::Hidden { gathered } => Some(gathered),
            Self::None | Self::Revealed => None,
        }
    }
}

impl Engine {
    /// Creates gameplay and transform schedules and default camera/input resources.
    pub fn new(config: EngineConfig) -> Self {
        let clock = FixedClock::new(config);
        // The engine's cube is admitted before the configured budget applies.
        let mut assets = AssetLibrary::default();
        let cube_mesh = assets
            .insert_mesh(Mesh::cube())
            .expect("an unlimited library holds a cube");
        assets.set_budget(config.asset_budget);
        let mut app = App::empty();
        // Bevy's message queues update in `First`, which the engine never
        // runs: without it, registering a queue is refused like any system
        // there.
        app.world_mut()
            .resource_mut::<Schedules>()
            .remove(bevy_app::First);
        app.add_observer(refuse_bevy_hierarchy)
            .init_schedule(FixedSimulation)
            .init_schedule(FixedPreSimulation)
            .init_schedule(FixedPostSimulation)
            .init_schedule(Presentation)
            .init_schedule(TransformPropagation)
            .add_systems(TransformPropagation, |world: &mut World| {
                propagate_transforms(world);
            })
            .add_systems(
                FixedSimulation,
                animation::animate.in_set(AnimationSystems::Animate),
            )
            .add_systems(FixedSimulation, ocean::float.in_set(OceanSystems::Float))
            .init_schedule(WorldSync)
            .init_schedule(ViewSync)
            .init_schedule(FixedTickEnd)
            // Before the frame's transforms refresh, the world follows the
            // camera; after, what follows its view.
            .add_systems(
                WorldSync,
                (
                    streaming::stream_cells.in_set(WorldSyncSystems::StreamCells),
                    grounded::settle.in_set(WorldSyncSystems::Ground),
                    bridges::sync.in_set(WorldSyncSystems::Bridges),
                    waterfalls::sync.in_set(WorldSyncSystems::Waterfalls),
                    spline_mesh::build.in_set(WorldSyncSystems::Splines),
                ),
            )
            .configure_sets(
                WorldSync,
                (
                    WorldSyncSystems::StreamCells,
                    WorldSyncSystems::Ground,
                    WorldSyncSystems::Bridges,
                    WorldSyncSystems::Waterfalls,
                    WorldSyncSystems::Splines,
                )
                    .chain(),
            )
            .add_systems(
                ViewSync,
                (
                    terrain::stream_terrains.in_set(ViewSyncSystems::StreamTerrains),
                    deformation::follow_camera.in_set(ViewSyncSystems::FollowDeformation),
                ),
            )
            .configure_sets(
                ViewSync,
                (
                    ViewSyncSystems::StreamTerrains,
                    ViewSyncSystems::FollowDeformation,
                )
                    .chain(),
            )
            // Deformers press tracks where each tick left them.
            .add_systems(
                FixedTickEnd,
                deformation::press.in_set(FixedTickEndSystems::PressTracks),
            )
            // The time of day moves the sun.
            .add_systems(Presentation, sky::advance_time_of_day)
            // Navigation agents walk.
            .add_systems(
                FixedSimulation,
                navigation::walk_agents.in_set(NavSystems::Walk),
            )
            // The engine's own work in a tick, in one order: agents walk,
            // floating objects ride the waves, then players pose their
            // targets. Gameplay orders itself against these sets.
            .configure_sets(
                FixedSimulation,
                (
                    NavSystems::Walk,
                    OceanSystems::Float,
                    AnimationSystems::Animate,
                )
                    .chain(),
            )
            .edit_schedule(FixedSimulation, |schedule| {
                schedule.set_executor(SingleThreadedExecutor::new());
            })
            .edit_schedule(FixedPreSimulation, |schedule| {
                schedule.set_executor(SingleThreadedExecutor::new());
            })
            .edit_schedule(FixedPostSimulation, |schedule| {
                schedule.set_executor(SingleThreadedExecutor::new());
            })
            .edit_schedule(Presentation, |schedule| {
                schedule.set_executor(SingleThreadedExecutor::new());
            })
            .edit_schedule(TransformPropagation, |schedule| {
                schedule.set_executor(SingleThreadedExecutor::new());
            })
            .edit_schedule(WorldSync, |schedule| {
                schedule.set_executor(SingleThreadedExecutor::new());
            })
            .edit_schedule(ViewSync, |schedule| {
                schedule.set_executor(SingleThreadedExecutor::new());
            })
            .edit_schedule(FixedTickEnd, |schedule| {
                schedule.set_executor(SingleThreadedExecutor::new());
            })
            .insert_resource(FixedTime {
                delta_seconds: clock.step_seconds() as f32,
                elapsed_seconds: 0.0,
                tick: 0,
            })
            .init_resource::<FrameTime>()
            .init_resource::<PresentedTime>()
            .init_resource::<Camera>()
            .init_resource::<Atmosphere>()
            .init_resource::<Tracks>()
            .init_resource::<deformation::DeformationFrame>()
            .init_resource::<WorldCells>()
            .init_resource::<WorldChanges>()
            .init_resource::<NavAgents>()
            .init_resource::<Jobs>()
            .init_resource::<Terrains>()
            .init_resource::<Viewport>()
            .init_resource::<InputState>()
            .init_resource::<ActionInput>()
            .init_resource::<PresentationState>()
            .init_resource::<InterpolationState>()
            .insert_resource(TransformNodeLimit(config.max_transform_nodes))
            .init_resource::<GlobalTransforms>()
            .insert_resource(assets);
        environment::init_sections(app.world_mut());
        let render_query = app.world_mut().query_filtered::<(
            Entity,
            Has<Transform>,
            Option<Ref<Cube>>,
            Option<Ref<Mesh3d>>,
            Option<Ref<Tint>>,
        ), Or<(With<Cube>, With<Mesh3d>)>>();
        let lights = lights::LightExtraction::new(app.world_mut());
        let particles = particles::ParticleExtraction::new(app.world_mut());
        let skins = skin::SkinExtraction::new(app.world_mut());
        let decals = decals::DecalExtraction::new(app.world_mut());
        let probes = probes::ProbeExtraction::new(app.world_mut());
        let mut retained = retained_render::RetainedRender::new(
            cube_mesh,
            config.frustum_culling,
            config.max_render_instances,
        );
        retained.ensure_source_scan(app.world_mut());
        Self {
            app,
            clock,
            render_query,
            retained,
            lights,
            particles,
            skins,
            decals,
            probes,
            plugins_finalized: false,
            tick_ahead: TickAhead::None,
        }
    }

    /// Runs bounded fixed simulation, then exactly one presentation phase.
    ///
    /// Nonfinite and negative deltas are treated as zero. Excess whole fixed
    /// ticks are discarded, preserving the fractional remainder. Presentation
    /// time is separately capped at 250 ms. Commands are visible on the next
    /// fixed tick, in presentation, and during subsequent extraction.
    ///
    /// Plugins must finish synchronous initialization before the first call.
    /// Bevy's `finish`/`cleanup` startup hooks run once; `cleanup` is not scene
    /// disposal. Use explicit ordering (`before`, `after`, `chain`) for systems
    /// that depend on each other within a phase.
    ///
    /// # Panics
    /// Panics if a plugin reports it is not ready on the first call, or a caller
    /// removes one of the required resources or engine schedules.
    pub fn advance(&mut self, delta_seconds: f64) -> FrameReport {
        self.finalize_plugins();
        let delta_seconds = clock::sanitize_delta(delta_seconds);
        let report = self.clock.advance(delta_seconds);
        self.run_frame(report.fixed_steps, delta_seconds.min(0.25));
        report
    }

    /// Advances exactly 1–120 fixed ticks as one development frame.
    ///
    /// Invalid counts return before any mutation, including plugin startup.
    /// Valid requests finalize plugins, discard the fractional clock remainder,
    /// and bypass the normal catch-up limit. Pending action edges survive and
    /// are consumed by the first tick only. No simulation time is dropped.
    ///
    /// Presentation runs once after all requested ticks. `FrameTime::frame`
    /// increments once; its delta is the requested simulation duration capped
    /// at 250 ms, as for ordinary frames. Call with one tick repeatedly when a
    /// game requires presentation after each tick. The report has zero dropped
    /// seconds and zero interpolation. Reset the host timestamp before resuming
    /// real-time advancement so development wall time does not become catch-up.
    ///
    /// # Panics
    /// Has the same plugin and required-resource contract as [`Self::advance`].
    #[cfg(feature = "dev")]
    pub fn step_fixed(&mut self, ticks: u32) -> Result<FrameReport, DevStepError> {
        DevStepError::validate(ticks)?;
        self.finalize_plugins();
        self.forget_ticks_ahead();
        self.clock.reset();
        let delta_seconds = (f64::from(ticks) * self.clock.step_seconds()).min(0.25);
        self.run_frame(ticks, delta_seconds);
        Ok(FrameReport {
            fixed_steps: ticks,
            dropped_seconds: 0.0,
            interpolation: 0.0,
        })
    }

    fn run_frame(&mut self, fixed_steps: u32, delta_seconds: f64) {
        let _profile = engine_profile::span("cpu", "simulation_frame");
        let mut fixed_steps = fixed_steps;
        if fixed_steps > 0 {
            match self.tick_ahead {
                TickAhead::None => {}
                // The first due tick is the one whose first stage already ran.
                TickAhead::InFlight => {
                    self.finish_pending_tick();
                    fixed_steps -= 1;
                }
                // A tick completed ahead of the clock counts as the first.
                TickAhead::Hidden { .. } | TickAhead::Revealed => {
                    self.tick_ahead = TickAhead::None;
                    fixed_steps -= 1;
                }
            }
        }
        // Boundaries while a tick is in flight would do its remaining
        // propagation early; its second stage refreshes when it is due.
        if self.tick_ahead != TickAhead::InFlight {
            self.refresh_transforms();
        }
        {
            let mut frame = self.app.world_mut().resource_mut::<FrameTime>();
            frame.delta_seconds = delta_seconds as f32;
            frame.elapsed_seconds += delta_seconds;
            frame.frame = frame.frame.saturating_add(1);
        }
        for _ in 0..fixed_steps {
            self.run_fixed_tick();
        }
        {
            let _profile = engine_profile::span("ecs", "Presentation");
            self.app.world_mut().run_schedule(Presentation);
        }
        if self.tick_ahead != TickAhead::InFlight {
            self.refresh_transforms();
        }
        self.app.world_mut().clear_trackers();
    }

    /// Runs one stage of the next fixed tick ahead of its due frame, with
    /// frames `predicted_delta_seconds` apart. A tick's first stage
    /// (pre-simulation and gameplay schedules) starts when the tick would be
    /// due within two frames; its second stage (transform propagation,
    /// post-simulation and interval capture) completes it when it would be due
    /// by the next frame, and the clock counts it when it becomes due. At
    /// display rates of at least twice the tick rate each frame then runs at
    /// most one stage after submitting, and the due frame only gathers and
    /// presents the completed tick. Stages still pending when the tick is due
    /// run then, as without running ahead. At most one tick completes ahead.
    /// Returns whether a stage ran.
    ///
    /// Call after extracting and submitting a frame. Until the tick completes,
    /// ordinary frames present the previous interval without refreshing
    /// transforms; exact extraction and explicit controls complete it first.
    /// Input for the tick is the input present when its first stage runs, up
    /// to two frames earlier than without running ahead. Its simulated time is
    /// unchanged.
    pub fn run_ahead(&mut self, predicted_delta_seconds: f64) -> bool {
        self.finalize_plugins();
        let in_flight = match self.tick_ahead {
            TickAhead::None => false,
            TickAhead::InFlight => true,
            TickAhead::Hidden { .. } | TickAhead::Revealed => return false,
        };
        let predicted = clock::sanitize_delta(predicted_delta_seconds).min(0.25);
        let accumulator = self.clock.accumulator_seconds();
        let step = self.clock.step_seconds();
        // Within a quarter frame of due counts as due at that frame.
        if in_flight {
            if accumulator + predicted * 1.25 < step {
                return false;
            }
            self.complete_pending_tick();
            // With GPU presentation, gather the completed tick's inputs in
            // this lighter frame: the due frame finds them current and only
            // presents.
            let _profile = engine_profile::span("cpu", "gather_ahead");
            let gathered = interpolation::gather_ahead(self.app.world_mut());
            self.tick_ahead = TickAhead::Hidden { gathered };
            return true;
        }
        if accumulator + predicted * 2.25 < step {
            return false;
        }
        let _profile = engine_profile::span("cpu", "fixed_tick_begin");
        self.begin_fixed_tick();
        self.tick_ahead = TickAhead::InFlight;
        true
    }

    /// Completes a tick started by [`Self::run_ahead`] now, before it is due;
    /// the clock counts it when it becomes due. Until then ordinary GPU frames
    /// keep presenting the previous interval; [`Self::reveal_ticks_ahead`] lets
    /// them present the completed tick instead. Exact extraction, stepping and
    /// suspension call this themselves.
    pub fn complete_pending_tick(&mut self) {
        if self.tick_ahead == TickAhead::InFlight {
            self.finish_pending_tick();
            self.tick_ahead = TickAhead::Hidden { gathered: false };
        }
    }

    /// Completes a tick in flight, which the clock then no longer owes:
    /// steps and clock resets count ticks from now.
    fn forget_ticks_ahead(&mut self) {
        self.complete_pending_tick();
        self.tick_ahead = TickAhead::None;
    }

    /// Lets GPU frames present ticks completed ahead of the clock, e.g. for an
    /// explicit preview of the latest completed interval. Ordinary frames then
    /// continue from that interval.
    pub fn reveal_ticks_ahead(&mut self) {
        if let TickAhead::Hidden { .. } = self.tick_ahead {
            self.tick_ahead = TickAhead::Revealed;
        }
    }

    /// Whether a completed tick awaits the frame the clock makes it due in.
    pub fn tick_completed_ahead(&self) -> bool {
        matches!(
            self.tick_ahead,
            TickAhead::Hidden { .. } | TickAhead::Revealed
        )
    }

    /// GPU presentation's current poses — the hierarchy's cached locals in
    /// upload layout — with their [`GpuPresentation::current_version`] now and
    /// how many the latest gather of that version stamped; `None` before GPU
    /// presentation first gathered. A renderer can stage them between frames,
    /// e.g. once a tick completes ahead of its due frame, and copy them when a
    /// gather with the same version uploads every pose. Gathers that stamped
    /// few poses upload only those, so staging every pose ahead would be wasted.
    pub fn gpu_current(&self) -> Option<GpuCurrent<'_>> {
        let world = self.app.world();
        world
            .get_resource::<PresentedTransforms>()?
            .gpu_current(world.get_resource::<GlobalTransforms>()?)
    }

    fn finish_pending_tick(&mut self) {
        let _profile = engine_profile::span("cpu", "fixed_tick_end");
        self.tick_ahead = TickAhead::None;
        self.finish_fixed_tick();
    }

    fn run_fixed_tick(&mut self) {
        let _profile = engine_profile::span("cpu", "fixed_tick");
        self.begin_fixed_tick();
        self.finish_fixed_tick();
    }

    /// Interval start, pre-simulation and gameplay schedules of one tick.
    fn begin_fixed_tick(&mut self) {
        interpolation::begin_tick(self.app.world_mut());
        {
            let mut time = self.app.world_mut().resource_mut::<FixedTime>();
            time.elapsed_seconds += self.clock.step_seconds();
            time.tick = time.tick.saturating_add(1);
        }
        self.app
            .world_mut()
            .resource_mut::<ActionInput>()
            .begin_tick();
        // Frame start or the preceding tick already propagated authoritative
        // transforms. Empty extension boundaries add no scans or schedule work.
        if self
            .app
            .get_schedule(FixedPreSimulation)
            .expect("engine pre schedule exists")
            .systems_len()
            != 0
        {
            {
                let _profile = engine_profile::span("ecs", "FixedPreSimulation");
                self.app.world_mut().run_schedule(FixedPreSimulation);
            }
            self.refresh_transforms();
        }
        {
            let _profile = engine_profile::span("ecs", "FixedSimulation");
            self.app.world_mut().run_schedule(FixedSimulation);
        }
    }

    /// Propagation, post-simulation schedule and interval end of one tick.
    fn finish_fixed_tick(&mut self) {
        self.refresh_transforms();
        if self
            .app
            .get_schedule(FixedPostSimulation)
            .expect("engine post schedule exists")
            .systems_len()
            != 0
        {
            {
                let _profile = engine_profile::span("ecs", "FixedPostSimulation");
                self.app.world_mut().run_schedule(FixedPostSimulation);
            }
            self.refresh_transforms();
        }
        // Deformers press tracks where the tick left them.
        {
            let _profile = engine_profile::span("ecs", "FixedTickEnd");
            self.app.world_mut().run_schedule(FixedTickEnd);
        }
        interpolation::end_tick(self.app.world_mut());
        self.app
            .world_mut()
            .resource_mut::<ActionInput>()
            .end_tick();
    }

    /// Discards the fractional accumulator after suspension without rewinding
    /// simulation ticks or elapsed time. The host should also reset its clock.
    pub fn reset_clock(&mut self) {
        self.forget_ticks_ahead();
        self.clock.reset();
        reset_all_interpolation(self.app.world_mut());
        self.app.world_mut().resource_mut::<ActionInput>().clear();
        self.app
            .world_mut()
            .resource_mut::<FrameTime>()
            .delta_seconds = 0.0;
    }

    /// Streams terrains toward the camera for a frame's work time without
    /// simulating or extracting a frame: hosts call it while
    /// the GPU compiles pipelines, so loading goes on meanwhile.
    pub fn stream_terrain(&mut self) {
        self.app
            .world_mut()
            .run_system_cached(terrain::stream_terrains)
            .expect("terrain streaming takes no input");
    }

    /// Loads terrain tiles the last frame found missing for `milliseconds`
    /// between frames, where frames leave time free while the
    /// GPU draws them; whether tiles are still to load.
    pub fn work_terrain(&mut self, milliseconds: f64) -> bool {
        terrain::work_between_frames(self.app.world_mut(), milliseconds)
    }

    /// Refreshes world matrices without advancing simulation or frame counters.
    ///
    /// Call after direct world edits when inspecting derived data before the
    /// next engine frame. Extraction also calls this boundary automatically.
    pub fn refresh_transforms(&mut self) {
        self.app.world_mut().run_schedule(TransformPropagation);
    }

    /// Enables or disables culling without advancing gameplay or replacing assets.
    pub fn set_frustum_culling(&mut self, enabled: bool) {
        self.retained.set_culling(enabled);
    }

    /// The renderer slot `entity` draws in, while it draws: what
    /// development tools read the GPU's per-instance state by.
    pub fn render_slot(&self, entity: Entity) -> Option<u32> {
        self.retained.slot_of(entity)
    }

    pub fn frustum_culling(&self) -> bool {
        self.retained.culling()
    }

    /// Metadata for the last prepared extraction, independent of GPU completion.
    pub fn presentation(&self) -> &PresentationState {
        self.app.world().resource::<PresentationState>()
    }

    /// Copies the CPU-prepared frame [`Self::extract_frame`] prepares into a
    /// reusable, owned CPU snapshot that borrows nothing.
    ///
    /// The snapshot keeps its vector capacity. Its instances are the frame's
    /// visible ones within the configured instance budget, in ECS iteration
    /// order, with their displayed models; `dropped_instances` reports the
    /// visible remainder. Invalid transforms skip before asset checks.
    /// Missing assets skip the entity and increment `missing_assets`;
    /// `Mesh3d` takes precedence over `Cube`; meshes with joints never cull.
    /// Each mesh, material with maps and texture used by an emitted or culled
    /// instance is listed once via `Arc`, without copying payloads, so
    /// removing library assets later does not invalidate this snapshot.
    /// Culled sources preserve existing GPU residency but do not request
    /// uploads on their own. Joints, decals, particle emitters and reflection
    /// probes are the frame's; local lights are in world space. GPU record
    /// conversion and uploads belong to the renderer.
    pub fn extract(&mut self, aspect: f32, snapshot: &mut RenderSnapshot) {
        self.extract_with_presentation(aspect, snapshot, PresentationMode::Current)
            .expect("current presentation is always valid");
    }

    /// [`Self::extract`] of current or interpolated display matrices, without
    /// changing gameplay.
    ///
    /// Opt-in local poses are captured around fixed ticks; descendants inherit
    /// their displayed ancestors. Culling and instance extraction use the same
    /// displayed matrix. Invalid fractions fail before any engine/snapshot change.
    /// [`Self::extract`] always selects exact authoritative current poses.
    pub fn extract_with_presentation(
        &mut self,
        aspect: f32,
        snapshot: &mut RenderSnapshot,
        mode: PresentationMode,
    ) -> Result<(), InterpolationError> {
        self.extract_frame(aspect, mode)?.copy_to(snapshot);
        Ok(())
    }

    /// Prepares a bounded, borrowed render frame with stable slots and trusted
    /// change revisions. The borrow prevents world edits until rendering ends.
    /// Unlike an owned snapshot, instance records include inactive slots;
    /// consumers must use [`RenderFrame::visible_indices`] and read poses with
    /// [`RenderFrame::model`].
    pub fn extract_frame(
        &mut self,
        aspect: f32,
        mode: PresentationMode,
    ) -> Result<RenderFrame<'_>, InterpolationError> {
        let _profile = engine_profile::span("cpu", "extraction");
        mode.validate()?;
        self.complete_pending_tick();
        self.mark_presented_time(match mode {
            PresentationMode::Current => None,
            PresentationMode::Interpolated { alpha } => Some(alpha),
        });
        // Before transforms refresh, so paused frames draw it too.
        self.sync_world();
        self.refresh_transforms();
        interpolation::prepare(self.app.world_mut(), mode);
        self.sync_view();
        Ok(self.extract_retained(aspect, false))
    }

    /// Prepares a retained frame whose interpolated display poses are blended,
    /// composed, validated and culled on the GPU.
    ///
    /// Inputs are gathered only when a boundary changed them, normally once per
    /// fixed tick, and the per-renderable pass is skipped while renderables,
    /// sources, assets and topology are unchanged. The frame's `gpu` inputs,
    /// `slot_nodes` and candidate `visible_indices` replace CPU poses and
    /// culling. [`PresentedTransforms`] keeps its previous CPU-prepared values;
    /// call [`Self::prepare_presented_transforms`] for CPU reads of this frame.
    /// Exact captures and previews use [`Self::extract_frame`].
    pub fn extract_gpu_frame(
        &mut self,
        aspect: f32,
        alpha: f32,
    ) -> Result<RenderFrame<'_>, InterpolationError> {
        let _profile = engine_profile::span("cpu", "extraction");
        PresentationMode::Interpolated { alpha }.validate()?;
        self.mark_presented_time(Some(alpha));
        if let Some(gathered) = self.tick_ahead.presents_previous() {
            // Mid-tick state, and a tick completed before it is due, are not
            // gathered: present the previous interval's inputs at this fraction.
            // Inputs already gathered ahead for the completed tick are held:
            // consumers keep presenting the inputs they applied before.
            interpolation::present_gpu_fraction(self.app.world_mut(), alpha, gathered);
        } else {
            self.sync_world();
            self.refresh_transforms();
            interpolation::prepare_gpu(self.app.world_mut(), alpha);
        }
        self.sync_view();
        Ok(self.extract_retained(aspect, true))
    }

    /// The world follows the camera, before the frame's transforms refresh:
    /// [`WorldSync`].
    fn sync_world(&mut self) {
        let _profile = engine_profile::span("ecs", "WorldSync");
        self.app.world_mut().run_schedule(WorldSync);
    }

    /// What follows the camera's view, after the frame's transforms
    /// refresh: [`ViewSync`].
    fn sync_view(&mut self) {
        let _profile = engine_profile::span("ecs", "ViewSync");
        self.app.world_mut().run_schedule(ViewSync);
    }

    /// Records the simulation time this frame presents ([`PresentedTime`]): the
    /// fixed clock less the part of a step interpolated poses at `alpha` stop
    /// short of it, and a step more while GPU frames present the interval
    /// before a tick in flight.
    fn mark_presented_time(&mut self, alpha: Option<f32>) {
        let step = self.clock.step_seconds();
        let behind = alpha.map_or(0.0, |alpha| (1.0 - f64::from(alpha)) * step);
        let ahead = if self.tick_ahead.presents_previous().is_some() {
            step
        } else {
            0.0
        };
        let world = self.app.world_mut();
        let fixed = world.resource::<FixedTime>().elapsed_seconds;
        world.resource_mut::<PresentedTime>().seconds = fixed - behind - ahead;
    }

    /// The retained frame of the poses just prepared: its instances, lights,
    /// particles, joints, decals and reflection probes. `gpu` frames leave
    /// poses and culling to the GPU and skip the instance pass while
    /// nothing structural changed.
    fn extract_retained(&mut self, aspect: f32, gpu: bool) -> RenderFrame<'_> {
        let world = self.app.world();
        self.retained.extract_deformation(world);
        self.retained
            .prepare(world, &mut self.render_query, aspect, gpu);
        self.retained.extract_lights_and_particles(
            world,
            &mut self.lights,
            &mut self.particles,
            gpu,
        );
        self.retained.extract_skins(world, &mut self.skins, gpu);
        self.retained.extract_decals(world, &mut self.decals, gpu);
        self.retained.extract_probes(world, &mut self.probes);
        self.retained.extract_terrain(world, aspect);
        self.retained
            .set_cells(world.resource::<WorldCells>().stats());
        if gpu {
            // Later source edits must be newer than the stamp window just
            // consumed.
            self.app.world_mut().increment_change_tick();
        }
        let world = self.app.world();
        self.retained.frame(
            world.resource::<PresentedTransforms>(),
            world.resource::<GlobalTransforms>(),
        )
    }

    /// CPU-prepares [`PresentedTransforms`] for the last GPU-presented frame,
    /// exactly as [`Self::extract_frame`] would have prepared it. Returns false
    /// and changes nothing when the last extraction was already CPU-prepared,
    /// or once a tick, refresh or reset changed that frame's inputs; then
    /// [`PresentationState::gpu_pending`] stays true. O(nodes).
    pub fn prepare_presented_transforms(&mut self) -> bool {
        if self.tick_ahead.presents_previous().is_some() {
            // The frame's inputs predate the in-flight or early tick's changes.
            return false;
        }
        interpolation::prepare_pending(self.app.world_mut())
    }

    fn finalize_plugins(&mut self) {
        if self.plugins_finalized {
            return;
        }
        self.refuse_unrun_schedules();
        match self.app.plugins_state() {
            PluginsState::Adding => {
                panic!("engine plugins must be ready before the first simulation frame")
            }
            PluginsState::Ready => {
                self.app.finish();
                self.app.cleanup();
            }
            PluginsState::Finished => self.app.cleanup(),
            PluginsState::Cleaned => {}
        }
        self.plugins_finalized = true;
    }
}

impl Engine {
    /// Panics on systems in a schedule the engine never runs, such as Bevy's
    /// `Update` or `Startup`, or the `First` that message queues add.
    fn refuse_unrun_schedules(&self) {
        let run = [
            FixedPreSimulation.intern(),
            FixedSimulation.intern(),
            FixedPostSimulation.intern(),
            Presentation.intern(),
            TransformPropagation.intern(),
            WorldSync.intern(),
            ViewSync.intern(),
            FixedTickEnd.intern(),
        ];
        let schedules = self.app.world().resource::<Schedules>();
        for (label, schedule) in schedules.iter() {
            assert!(
                schedule.systems_len() == 0 || run.contains(&schedule.label()),
                "{label:?} has systems, but the engine never runs it: add gameplay systems to \
                 FixedPreSimulation, FixedSimulation, FixedPostSimulation or Presentation"
            );
        }
    }
}

/// Panics on Bevy's `ChildOf`, which the engine does not read: entities get
/// their parents as [`Parent`] components from [`set_parent`].
fn refuse_bevy_hierarchy(add: On<Add, bevy_ecs::hierarchy::ChildOf>) {
    panic!(
        "{} was given Bevy's ChildOf, which the engine ignores: give it a parent with set_parent",
        add.entity
    );
}

impl Default for Engine {
    fn default() -> Self {
        Self::new(EngineConfig::default())
    }
}

/// The bounds GPU presentation's shaders share with the CPU reference
/// (`presentation.wgsl`): culling's coefficient and rounding limits, the
/// hierarchy validator's common admission path and interpolation's linear
/// rotation threshold.
pub mod presentation_bounds {
    pub use crate::culling::{MAX_COEFFICIENT, MAX_PRODUCT_BOUND, ROUNDING_MARGIN};
    pub use crate::hierarchy::{COEFFICIENT_LIMIT, COMMON_DETERMINANT_MARGIN, MIN_LINEAR_SCALE};
    pub use crate::interpolation::LINEAR_ROTATION_DOT_THRESHOLD;
}
