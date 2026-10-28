#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod antialiasing;
mod atmosphere;
mod clock;
mod clouds;
mod commands;
mod debug_view;
mod effects;
mod entity_indices;
mod environment;
mod extraction;
mod features;
mod foliage;
mod frame_stats;
mod gpu_culling;
mod grounded;
mod hierarchy;
mod highlights;
mod input;
mod interpolation;
mod jobs;
mod level_of_detail;
mod lights;
pub mod math;
mod mesh_lods;
pub use engine_math::collections;
mod packed;
mod particles;
mod physics;
pub mod presentation_bounds;
mod probes;
mod records;
mod resolution;
mod retained_render;
mod scene;
mod sky;
mod sky_occlusion;
mod spline_mesh;
mod statistics;
mod tables;
mod ticks;
mod transform_edits;
mod water;

pub use antialiasing::*;
pub use atmosphere::*;
pub use bevy_app::{App, Plugin, PluginGroup, PluginGroupBuilder};
pub use bevy_ecs;
pub use clock::*;
pub use clouds::*;
pub use commands::*;
pub use debug_view::*;
pub use effects::*;
pub use engine_assets::{
    Aabb, AlphaImage, AlphaMode, AssetBudget, AssetError, AssetLibrary, Filter, Grounding,
    LOD_RECORD, LodInput, LodLevel, MAX_TEXTURE_DIMENSION, Material, MaterialBinding,
    MaterialHandle, Mesh, MeshHandle, MeshLods, MeshVertex, Sampler, Surface, Texture,
    TextureFormat, TextureHandle, TextureRef, TopLayer, UvTransform, Wrap, lod_input, trim_cards,
};
pub use engine_settings::ErrorCode;
pub use engine_tables as table_data;
pub use engine_terrain as terrain_data;
pub use engine_terrain::{
    AUTHORED_LAYER, AuthoredPlacement, Biome, BiomeError, BiomeLayer, BiomeRules, Biomes, Blend,
    Control, Erosion, ErosionLayer, ErosionStage, ErosionStats, Facing, FarPlacement, GivenRiver,
    HeightImage, HeightmapLayer, HummocksLayer, Hydrology, HydrologyLayer, HydrologySample,
    HydrologyStage, HydrologyStats, Layer, LayerError, Layers, Mask, NoiseLayer, NoiseMask,
    PathLayer, PathProfile, PathStage, PathStats, Placed, PlacementChoice, PlacementLayer, Pyramid,
    PyramidError, Remap, RiversLayer, Route, RouteOptions, Stamp, StampOp, StepsLayer, Tile,
    TileKey,
};
pub use environment::*;
pub use features::*;
pub use foliage::*;
pub use frame_stats::*;
pub use glam;
pub use gpu_culling::*;
pub use grounded::*;
pub use hierarchy::*;
pub use highlights::*;
pub use input::*;
pub use interpolation::*;
pub use jobs::*;
pub use level_of_detail::*;
pub use lights::*;
pub use particles::*;
pub use physics::{Physics, PhysicsStats};
pub use probes::*;
pub use records::*;
pub use resolution::*;
pub use retained_render::*;
pub use scene::*;
pub use sky::*;
pub use sky_occlusion::*;
pub use spline_mesh::*;
pub use statistics::{Distribution, median};
pub use tables::*;
pub use transform_edits::*;
pub use water::{WaterKind, WaterObstacle, WaterObstacles, WaterSample, WaterSource, WaterSources};

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

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixedSimulation;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixedPreSimulation;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixedPostSimulation;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Presentation;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransformPropagation;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorldSync;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ViewSync;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FixedSimulationSystems {
    Move,
    Simulate,
    Ride,
    Pose,
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorldSyncSystems {
    Stream,
    Ground,
    Spawn,
    Build,
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewSyncSystems {
    Stream,
}

pub const MAX_DEV_TICKS: u32 = 120;

#[cfg(feature = "dev")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevStepError {
    pub ticks: u32,
}

#[cfg(feature = "dev")]
impl DevStepError {
    pub const MAX_TICKS: u32 = MAX_DEV_TICKS;

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

pub struct Engine {
    pub app: App,
    clock: FixedClock,
    render_query: RenderQuery,
    retained: retained_render::RetainedRender,
    extractions: extraction::Extractions,
    plugins_finalized: bool,
    tick_ahead: TickAhead,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TickAhead {
    None,
    InFlight,
    Hidden { gathered: bool },
    Revealed,
}

impl TickAhead {
    fn presents_previous(self) -> Option<bool> {
        match self {
            Self::InFlight => Some(false),
            Self::Hidden { gathered } => Some(gathered),
            Self::None | Self::Revealed => None,
        }
    }
}

fn single_threaded(schedule: &mut Schedule) {
    schedule.set_executor(SingleThreadedExecutor::new());
}

impl Engine {
    pub fn new(config: EngineConfig) -> Self {
        let clock = FixedClock::new(config);
        let mut assets = AssetLibrary::default();
        let cube_mesh = assets
            .insert_mesh(Mesh::cube())
            .expect("an unlimited library holds a cube");
        assets.set_budget(config.asset_budget);
        let mut app = App::empty();
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
            .init_schedule(WorldSync)
            .init_schedule(ViewSync)
            .add_systems(
                WorldSync,
                (
                    grounded::settle.in_set(WorldSyncSystems::Ground),
                    spline_mesh::build.in_set(WorldSyncSystems::Build),
                ),
            )
            .configure_sets(
                WorldSync,
                (
                    WorldSyncSystems::Stream,
                    WorldSyncSystems::Ground,
                    WorldSyncSystems::Spawn,
                    WorldSyncSystems::Build,
                )
                    .chain(),
            )
            .configure_sets(
                FixedSimulation,
                (
                    FixedSimulationSystems::Move,
                    FixedSimulationSystems::Simulate,
                    FixedSimulationSystems::Ride,
                    FixedSimulationSystems::Pose,
                )
                    .chain(),
            )
            .edit_schedule(FixedSimulation, single_threaded)
            .edit_schedule(FixedPreSimulation, single_threaded)
            .edit_schedule(FixedPostSimulation, single_threaded)
            .edit_schedule(Presentation, single_threaded)
            .edit_schedule(TransformPropagation, single_threaded)
            .edit_schedule(WorldSync, single_threaded)
            .edit_schedule(ViewSync, single_threaded)
            .insert_resource(FixedTime {
                delta_seconds: clock.step_seconds() as f32,
                elapsed_seconds: 0.0,
                tick: 0,
            })
            .init_resource::<FrameTime>()
            .init_resource::<PresentedTime>()
            .init_resource::<Camera>()
            .init_resource::<Atmosphere>()
            .init_resource::<Features>()
            .init_resource::<WaterSources>()
            .init_resource::<Jobs>()
            .init_resource::<Records>()
            .init_resource::<mesh_lods::MeshLevels>()
            .init_resource::<StaticTables>()
            .init_resource::<Viewport>()
            .init_resource::<InputState>()
            .init_resource::<ActionInput>()
            .init_resource::<PresentationState>()
            .init_resource::<InterpolationState>()
            .insert_resource(TransformNodeLimit(config.max_transform_nodes))
            .init_resource::<GlobalTransforms>()
            .insert_resource(assets);
        environment::init_sections(app.world_mut(), None);
        let render_query = app.world_mut().query_filtered::<(
            Entity,
            Has<Transform>,
            Option<Ref<Cube>>,
            Option<Ref<Mesh3d>>,
            Option<Ref<Tint>>,
        ), Or<(With<Cube>, With<Mesh3d>)>>();
        let extractions = extraction::Extractions::new(app.world_mut());
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
            extractions,
            plugins_finalized: false,
            tick_ahead: TickAhead::None,
        }
    }

    pub fn advance(&mut self, delta_seconds: f64) -> FrameReport {
        self.finalize_plugins();
        let delta_seconds = clock::sanitize_delta(delta_seconds);
        let report = self.clock.advance(delta_seconds);
        self.run_frame(report.fixed_steps, delta_seconds.min(0.25));
        report
    }

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
                TickAhead::InFlight => {
                    self.finish_pending_tick();
                    fixed_steps -= 1;
                }
                TickAhead::Hidden { .. } | TickAhead::Revealed => {
                    self.tick_ahead = TickAhead::None;
                    fixed_steps -= 1;
                }
            }
        }
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
        records::exchange(self.app.world_mut());
        {
            let _profile = engine_profile::span("ecs", "Presentation");
            self.app.world_mut().run_schedule(Presentation);
        }
        if self.tick_ahead != TickAhead::InFlight {
            self.refresh_transforms();
        }
        self.app.world_mut().clear_trackers();
    }

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
        if in_flight {
            if accumulator + predicted * 1.25 < step {
                return false;
            }
            self.complete_pending_tick();
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

    pub fn complete_pending_tick(&mut self) {
        if self.tick_ahead == TickAhead::InFlight {
            self.finish_pending_tick();
            self.tick_ahead = TickAhead::Hidden { gathered: false };
        }
    }

    fn forget_ticks_ahead(&mut self) {
        self.complete_pending_tick();
        self.tick_ahead = TickAhead::None;
    }

    pub fn reveal_ticks_ahead(&mut self) {
        if let TickAhead::Hidden { .. } = self.tick_ahead {
            self.tick_ahead = TickAhead::Revealed;
        }
    }

    pub fn tick_completed_ahead(&self) -> bool {
        matches!(
            self.tick_ahead,
            TickAhead::Hidden { .. } | TickAhead::Revealed
        )
    }

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
        interpolation::end_tick(self.app.world_mut());
        self.app
            .world_mut()
            .resource_mut::<ActionInput>()
            .end_tick();
    }

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

    pub fn stream_terrain(&mut self) {
        self.app
            .world_mut()
            .run_system_cached(stream_terrains)
            .expect("terrain streaming takes no input");
    }

    pub fn work_terrain(&mut self, milliseconds: f64) -> bool {
        work_between_frames(self.app.world_mut(), milliseconds)
    }

    pub fn refresh_transforms(&mut self) {
        self.app.world_mut().run_schedule(TransformPropagation);
    }

    pub fn set_frustum_culling(&mut self, enabled: bool) {
        self.retained.set_culling(enabled);
    }

    pub fn render_slot(&self, entity: Entity) -> Option<u32> {
        self.retained.slot_of(entity)
    }

    pub fn render_entity(&self, slot: u32) -> Option<Entity> {
        self.retained
            .entities
            .iter()
            .find_map(|(&entity, &held)| (held as u32 == slot).then_some(entity))
    }

    pub fn frustum_culling(&self) -> bool {
        self.retained.culling()
    }

    pub fn presentation(&self) -> &PresentationState {
        self.app.world().resource::<PresentationState>()
    }

    pub fn extract(&mut self, aspect: f32, snapshot: &mut RenderSnapshot) {
        self.extract_frame(aspect).copy_to(snapshot);
    }

    pub fn extract_frame(&mut self, aspect: f32) -> RenderFrame<'_> {
        let _profile = engine_profile::span("cpu", "extraction");
        self.finalize_plugins();
        self.complete_pending_tick();
        self.mark_presented_time(None);
        self.sync_world();
        self.refresh_transforms();
        interpolation::prepare(self.app.world_mut());
        self.sync_view();
        self.extract_retained(aspect, false)
    }

    pub fn extract_gpu_frame(
        &mut self,
        aspect: f32,
        alpha: f32,
    ) -> Result<RenderFrame<'_>, InterpolationError> {
        let _profile = engine_profile::span("cpu", "extraction");
        PresentationMode::Interpolated { alpha }.validate()?;
        self.finalize_plugins();
        self.mark_presented_time(Some(alpha));
        if let Some(gathered) = self.tick_ahead.presents_previous() {
            interpolation::present_gpu_fraction(self.app.world_mut(), alpha, gathered);
        } else {
            self.sync_world();
            self.refresh_transforms();
            interpolation::prepare_gpu(self.app.world_mut(), alpha);
        }
        self.sync_view();
        Ok(self.extract_retained(aspect, true))
    }

    fn sync_world(&mut self) {
        let _profile = engine_profile::span("ecs", "WorldSync");
        records::exchange(self.app.world_mut());
        mesh_lods::settle(self.app.world_mut());
        self.app.world_mut().run_schedule(WorldSync);
    }

    fn sync_view(&mut self) {
        let _profile = engine_profile::span("ecs", "ViewSync");
        let world = self.app.world_mut();
        world.run_schedule(ViewSync);
        tables::update_tables(world);
    }

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

    fn extract_retained(&mut self, aspect: f32, gpu: bool) -> RenderFrame<'_> {
        let world = self.app.world();
        self.retained
            .prepare(world, &mut self.render_query, aspect, gpu);
        self.extractions
            .extract(&mut self.retained, world, aspect, gpu);
        if gpu {
            self.app.world_mut().increment_change_tick();
        }
        let world = self.app.world();
        self.retained.frame(
            world.resource::<PresentedTransforms>(),
            world.resource::<GlobalTransforms>(),
        )
    }

    fn finalize_plugins(&mut self) {
        if self.plugins_finalized {
            return;
        }
        self.refuse_unrun_schedules();
        let state = self.app.plugins_state();
        assert!(
            !matches!(state, PluginsState::Adding),
            "engine plugins must be ready before the first frame"
        );
        features::check_needs(self.app.world());
        match state {
            PluginsState::Ready => {
                self.app.finish();
                self.app.cleanup();
            }
            PluginsState::Finished => self.app.cleanup(),
            PluginsState::Adding | PluginsState::Cleaned => {}
        }
        self.plugins_finalized = true;
    }
}

impl Engine {
    fn refuse_unrun_schedules(&self) {
        let run = [
            FixedPreSimulation.intern(),
            FixedSimulation.intern(),
            FixedPostSimulation.intern(),
            Presentation.intern(),
            TransformPropagation.intern(),
            WorldSync.intern(),
            ViewSync.intern(),
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
