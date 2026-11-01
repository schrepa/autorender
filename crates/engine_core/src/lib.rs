#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

extern crate self as engine_core;

mod antialiasing;
pub mod bounds;
pub mod brick_smokes;
mod bricks;
mod clock;
mod clouds;
pub mod command_line;
mod debug_view;
pub mod debug_view_smoke;
mod effects;
mod entity_indices;
mod extraction;
mod foliage;
mod frame_stats;
mod gpu_culling;
mod grid;
mod hierarchy;
mod highlights;
mod input;
mod interpolation;
mod jobs;
mod level_of_detail;
mod light;
mod lights;
pub mod math;
mod mesh_lods;
pub use engine_math::collections;
mod particles;
pub mod presentation_bounds;
mod probes;
mod provided;
mod records;
mod resolution;
mod retained_render;
mod scene;
mod schedules;
mod section;
mod services;
mod sky;
mod sky_occlusion;
pub mod smoke;
mod source;
mod spline_mesh;
mod statistics;
mod tables;
mod ticks;
mod transform_edits;
mod wind;

pub use antialiasing::*;
pub use bevy_app::App;
pub use bricks::*;
pub use clock::*;
pub use clouds::*;
pub use debug_view::*;
pub use effects::*;
pub use engine_assets::{
    Aabb, AlphaImage, AlphaMode, AssetBudget, AssetLibrary, Filter, Grounding, LOD_RECORD,
    LodInput, LodLevel, MAX_TEXTURE_DIMENSION, Material, MaterialBinding, Mesh, MeshLods,
    MeshVertex, Sampler, Surface, Texture, TextureFormat, TextureRef, TopLayer, UvTransform, Wrap,
    lod_input, trim_cards,
};
pub use engine_place::{
    DrawPoint, Error, ErrorCode, GpuStruct, InterpolateTransform, JobRef, List, MaterialHandle,
    Mesh3d, MeshHandle, PlaceDecl, Runs, Slot, Smoke, SmokeFailure, SmokeSession, Stamp,
    TextureHandle, gpu_consts, smoke_eq,
};
pub use engine_settings::{settings, stats};
pub use foliage::*;
pub use frame_stats::*;
pub use glam;
pub use gpu_culling::*;
pub use grid::*;
pub use hierarchy::*;
pub use highlights::*;
pub use input::*;
pub use interpolation::*;
pub use jobs::*;
pub use level_of_detail::*;
pub use light::*;
pub use lights::*;
pub use particles::*;
pub use probes::*;
pub use provided::*;
pub use records::*;
pub use resolution::*;
pub use retained_render::*;
pub use scene::*;
pub use section::{Cut, Operation, Readback, Section};
pub use services::{
    AssetResidency, ComponentKind, Detail, DeviceLimits, Poses, SceneComponents, View,
};
pub use sky::*;
pub use sky_occlusion::*;
pub use source::{Entry, Module, ShaderText, Source, Var};
pub use spline_mesh::*;
pub use statistics::{Distribution, median};
pub use tables::*;
pub use transform_edits::*;
pub use wind::*;

pub mod codec {
    pub use engine_codec::{decode, encode};
}

#[doc(hidden)]
pub mod __private {
    pub use ::bytemuck;
    pub use ::schemars;
    pub use ::serde;
    pub use engine_place::gpu_struct as place_gpu_struct;
}

#[macro_export]
macro_rules! gpu_struct {
    ($($body:tt)*) => {
        $crate::__private::place_gpu_struct! { @crate "::engine_core::__private::bytemuck"; $($body)* }
    };
}

use bevy_ecs::change_detection::Ref;
use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::hierarchy::ChildOf;
use bevy_ecs::lifecycle::Add;
use bevy_ecs::observer::On;
use bevy_ecs::query::{Has, Or, QueryState, With};
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::{ScheduleLabel, Schedules, SingleThreadedExecutor};
use bevy_ecs::world::World;
use clock::FixedClock;
use interpolation::{InterpolationState, ResetWatch};
use schedules::{
    BetweenFrames, FixedMove, FixedPostSimulation, FixedPreSimulation, FixedSimulation,
    Presentation, TransformPropagation, ViewSync, WorldSync,
};

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

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamingFocus;

pub const MAX_DEV_TICKS: u32 = 120;

#[cfg(feature = "dev")]
pub fn check_dev_ticks(ticks: u32) -> Result<(), Error> {
    if (1..=MAX_DEV_TICKS).contains(&ticks) {
        Ok(())
    } else {
        Err(Error::new(ErrorCode::InvalidStep, "$.ticks")
            .range(1.0, f64::from(MAX_DEV_TICKS))
            .got(ticks))
    }
}

pub struct Engine {
    pub app: App,
    clock: FixedClock,
    render_query: RenderQuery,
    retained: retained_render::RetainedRender,
    resets: ResetWatch,
    finalized: bool,
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

fn single_threaded(schedule: &mut bevy_ecs::schedule::Schedule) {
    schedule.set_executor(SingleThreadedExecutor::new());
}

fn modify<R: Resource>(world: &mut World, change: impl FnOnce(&mut R)) {
    let changed = world.modify_resource(change).ok().flatten();
    assert!(
        changed.is_some(),
        "core_resource_held {}",
        std::any::type_name::<R>()
    );
}

fn has_systems(world: &World, label: impl ScheduleLabel) -> bool {
    world
        .resource::<Schedules>()
        .get(label)
        .is_some_and(|schedule| schedule.systems_len() != 0)
}

impl Engine {
    pub fn new(config: EngineConfig) -> Self {
        let clock = FixedClock::new(config);
        let mut assets = AssetLibrary::default();
        let cube_mesh = assets
            .insert_mesh(Mesh::cube())
            .expect("unlimited_library_holds_cube");
        assets.set_budget(config.asset_budget);
        let mut app = App::empty();
        app.world_mut()
            .resource_mut::<Schedules>()
            .remove(bevy_app::First);
        interpolation::track_markers(app.world_mut());
        app.add_observer(refuse_bevy_hierarchy);
        for schedule in Schedule::ALL {
            app.init_schedule(schedule.label())
                .edit_schedule(schedule.label(), single_threaded);
        }
        app.init_schedule(TransformPropagation)
            .add_systems(TransformPropagation, |world: &mut World| {
                propagate_transforms(world);
            })
            .edit_schedule(TransformPropagation, single_threaded)
            .add_systems(WorldSync, spline_mesh::build)
            .insert_resource(FixedTime {
                delta_seconds: clock.step_seconds() as f32,
                elapsed_seconds: 0.0,
                tick: 0,
            })
            .init_resource::<FrameTime>()
            .init_resource::<Camera>()
            .init_resource::<Detail>()
            .init_resource::<CookMode>()
            .init_resource::<JobQueue>()
            .init_resource::<engine_registry::JobListed>()
            .insert_resource(Records::empty())
            .init_resource::<RecordExchange>()
            .insert_resource(View::empty())
            .insert_resource(Poses::empty())
            .insert_resource(AssetResidency::empty())
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
        let render_query = app.world_mut().query_filtered::<(
            Entity,
            Has<Transform>,
            Option<Ref<Cube>>,
            Option<Ref<Mesh3d>>,
            Option<Ref<Tint>>,
        ), Or<(With<Cube>, With<Mesh3d>)>>();
        let resets = ResetWatch::new(app.world_mut());
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
            resets,
            finalized: false,
            tick_ahead: TickAhead::None,
        }
    }

    pub fn advance(&mut self, delta_seconds: f64) -> FrameReport {
        self.finalize();
        let delta_seconds = clock::sanitize_delta(delta_seconds);
        let report = self.clock.advance(delta_seconds);
        self.run_frame(report.fixed_steps, delta_seconds.min(0.25));
        report
    }

    #[cfg(feature = "dev")]
    pub fn step_fixed(&mut self, ticks: u32) -> Result<FrameReport, Error> {
        check_dev_ticks(ticks)?;
        self.finalize();
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

    pub fn between_frames(&mut self, micros: u32) {
        self.finalize();
        let world = self.app.world_mut();
        modify(world, |frame: &mut FrameTime| {
            frame.between_frames_micros = micros;
        });
        if has_systems(world, BetweenFrames) {
            let _profile = engine_profile::span("ecs", "BetweenFrames");
            world.run_schedule(BetweenFrames);
        }
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
        modify(self.app.world_mut(), |frame: &mut FrameTime| {
            frame.delta_seconds = delta_seconds as f32;
            frame.elapsed_seconds += delta_seconds;
            frame.frame = frame.frame.saturating_add(1);
        });
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
        self.finalize();
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
        let step = self.clock.step_seconds();
        modify(self.app.world_mut(), |time: &mut FixedTime| {
            time.elapsed_seconds += step;
            time.tick = time.tick.saturating_add(1);
        });
        modify(self.app.world_mut(), ActionInput::begin_tick);
        if has_systems(self.app.world(), FixedPreSimulation) {
            {
                let _profile = engine_profile::span("ecs", "FixedPreSimulation");
                self.app.world_mut().run_schedule(FixedPreSimulation);
            }
            self.refresh_transforms();
        }
        if has_systems(self.app.world(), FixedMove) {
            let _profile = engine_profile::span("ecs", "FixedMove");
            self.app.world_mut().run_schedule(FixedMove);
        }
        {
            let _profile = engine_profile::span("ecs", "FixedSimulation");
            self.app.world_mut().run_schedule(FixedSimulation);
        }
        self.resets.snap(self.app.world_mut());
    }

    fn finish_fixed_tick(&mut self) {
        self.refresh_transforms();
        if has_systems(self.app.world(), FixedPostSimulation) {
            {
                let _profile = engine_profile::span("ecs", "FixedPostSimulation");
                self.app.world_mut().run_schedule(FixedPostSimulation);
            }
            self.refresh_transforms();
        }
        interpolation::end_tick(self.app.world_mut());
        modify(self.app.world_mut(), ActionInput::end_tick);
    }

    pub fn reset_clock(&mut self) {
        self.forget_ticks_ahead();
        self.clock.reset();
        let world = self.app.world_mut();
        reset_all_interpolation(world);
        modify(world, ActionInput::clear);
        modify(world, |frame: &mut FrameTime| frame.delta_seconds = 0.0);
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
        self.retained.entity_in(slot)
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
        self.finalize();
        self.complete_pending_tick();
        self.mark_presented_time(None);
        self.sync_world();
        self.refresh_transforms();
        interpolation::prepare(self.app.world_mut());
        self.sync_view();
        self.extract_retained(aspect, false)
    }

    pub fn extract_gpu_frame(&mut self, aspect: f32, alpha: f32) -> Result<RenderFrame<'_>, Error> {
        let _profile = engine_profile::span("cpu", "extraction");
        PresentationMode::Interpolated { alpha }.validate()?;
        self.finalize();
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
        modify(world, |frame: &mut FrameTime| {
            frame.presented_seconds = fixed - behind - ahead;
        });
    }

    fn extract_retained(&mut self, aspect: f32, gpu: bool) -> RenderFrame<'_> {
        let world = self.app.world();
        self.retained
            .prepare(world, &mut self.render_query, aspect, gpu);
        if gpu {
            self.app.world_mut().increment_change_tick();
        }
        let world = self.app.world();
        self.retained.frame(
            world.resource::<PresentedTransforms>(),
            world.resource::<GlobalTransforms>(),
        )
    }

    fn finalize(&mut self) {
        if self.finalized {
            return;
        }
        self.refuse_unrun_schedules();
        self.finalized = true;
    }

    fn refuse_unrun_schedules(&self) {
        let mut run: Vec<_> = Schedule::ALL.iter().map(|schedule| schedule.label()).collect();
        run.push(TransformPropagation.intern());
        let schedules = self.app.world().resource::<Schedules>();
        for (label, schedule) in schedules.iter() {
            assert!(
                schedule.systems_len() == 0 || run.contains(&schedule.label()),
                "schedule_never_runs {label:?}"
            );
        }
    }
}

fn refuse_bevy_hierarchy(add: On<Add, ChildOf>) {
    panic!("child_of_ignored {}", add.entity);
}

impl Default for Engine {
    fn default() -> Self {
        Self::new(EngineConfig::default())
    }
}
