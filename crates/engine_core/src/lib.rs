#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

extern crate self as engine_core;

mod assets;
pub mod bounds;
mod bricks;
mod clock;
pub mod command_line;
mod entity_indices;
mod frame_stats;
mod hierarchy;
mod input;
mod interpolation;
mod jobs;
pub mod math;
mod mesh_lods;
pub use engine_math::collections;
mod nearest;
pub mod presentation_bounds;
mod provided;
mod records;
mod scene;
mod schedules;
mod section;
mod services;
pub mod smoke;
mod source;
mod statistics;
mod ticks;
mod transform_edits;

mod sources {
    include!(concat!(env!("OUT_DIR"), "/sources.rs"));
}

pub mod profile {
    pub use engine_profile::{Aggregate, Span, active, counter, now_ms, span, tally, waited};
}

pub use assets::*;
pub use bevy_app::App;
pub use bricks::*;
pub use clock::*;
pub use engine_assets::{
    Aabb, AlphaImage, AlphaMode, AssetBudget, AssetLibrary, Filter, Grounding, LodInput, LodLevel,
    MAX_TEXTURE_DIMENSION, Material, MaterialBinding, Mesh, MeshLods, MeshVertex, Sampler,
    Surface, Texture, TextureFormat, TextureRef, TopLayer, UvTransform, Wrap, lod_input,
    trim_cards,
};
pub use engine_place::{
    DrawPoint, Error, ErrorCode, GpuStruct, InterpolateTransform, JobRef, List, MaterialHandle,
    Mesh3d, MeshHandle, PlaceDecl, Runs, Slot, Smoke, SmokeFailure, SmokeSession, Stamp,
    TextureHandle, gpu_consts, smoke_check, smoke_eq,
};
pub use engine_registry::{AuthoredId, ExplainAsk, ExplainRecord, FrameFacts, RecordParity};
pub use engine_settings::{
    Allowed, Expected, ExtraRules, ID_PATTERN, InvalidSetting, MAX_ID_BYTES, Rule, Setting,
    fields, is_id, parse_value, parse_value_at, rule, settings, state, stated, stats, valid_uri,
};
pub use frame_stats::*;
pub use glam;
pub use hierarchy::*;
pub use input::*;
pub use interpolation::*;
pub use jobs::*;
pub use nearest::Nearest;
pub use provided::*;
pub use records::*;
pub use scene::*;
pub use section::{Cut, Explain, Operation, Readback, Section};
pub use services::{
    Authored, ComponentKind, ComponentKindOf, Detail, DeviceLimits, PartDraws, PartFaults, Placed,
    Placement, Poses, Prepared, Reach, SceneBuffers, SceneComponents, SceneContext, SceneCx,
    SceneIds, ScenePath, View, ViewWind, WorkClock, WorkRemaining, scene_component,
    scene_components,
};
pub use source::{Entry, Module, ShaderText, Source, Var};
pub use statistics::{Distribution, median};
pub use ticks::ScanClock;
pub use transform_edits::*;

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

use std::sync::Arc;

use bevy_ecs::change_detection::CheckChangeTicks;
use bevy_ecs::component::Immutable;
use bevy_ecs::hierarchy::ChildOf;
use bevy_ecs::lifecycle::Add;
use bevy_ecs::observer::On;
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::{InternedScheduleLabel, ScheduleLabel, Schedules, SingleThreadedExecutor};
use bevy_ecs::system::ResMut;
use bevy_ecs::world::World;
use clock::FixedClock;
use engine_registry::{
    BrickHooks, GateDriver, JobList, JobListed, MembersOf, OperationEntry, Registry,
};
use interpolation::{InterpolationState, ResetWatch};
use schedules::{
    BetweenFrames, FixedMove, FixedPostSimulation, FixedPreSimulation, FixedSimulation, FrameStart,
    Presentation, Rendered, Stream, TransformPropagation, ViewSync, Waiting, WorldSync,
};

pub const MAX_DEV_TICKS: u32 = 120;

static CORE_PLACES: &[PlaceDecl] = &[
    PlaceDecl::list::<SceneComponents>(),
    PlaceDecl::slot::<ImpostorPolicy>(),
    PlaceDecl::slot::<ViewWind>(),
];

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
    resets: ResetWatch,
    gates: Option<GateDriver>,
    facts: Option<FrameFacts>,
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

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct RenderFrame<'a> {
    world: &'a World,
    gpu: bool,
}

impl<'a> RenderFrame<'a> {
    pub fn world(self) -> &'a World {
        self.world
    }

    pub fn gpu(self) -> Option<GpuPresentation<'a>> {
        if !self.gpu {
            return None;
        }
        let poses = self.world.get_resource::<PresentedTransforms>()?;
        let hierarchy = self.world.get_resource::<GlobalTransforms>()?;
        Some(poses.gpu_presentation(hierarchy))
    }
}

#[derive(Resource, Default)]
struct TickCheck(Option<CheckChangeTicks>);

fn note_tick_check(check: On<CheckChangeTicks>, mut held: ResMut<TickCheck>) {
    held.0 = Some(*check.event());
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

fn core_entries(registry: &mut Registry) {
    registry.core_settings(bricks::settings_entry::<Detail>());
    registry.core_settings(bricks::settings_entry::<JobSettings>());
    registry.core_operation(OperationEntry::of::<
        <services::ViewControl as Operation>::Params,
    >(
        services::ViewControl::NAME,
        true,
        bricks::run_operation::<services::ViewControl>,
    ));
    registry.core_operation(OperationEntry::of::<()>(
        transform_edits::EDIT_TRANSFORM,
        true,
        transform_edits::run_edit_transform,
    ));
    registry.writers_first::<Camera>();
}

fn refused(problems: &[Error]) -> Result<(), Error> {
    match problems {
        [] => Ok(()),
        [only] => Err(only.clone()),
        [first, ..] => Err(Error::new(first.code, "$.bricks").got(problems)),
    }
}

fn publish_view(world: &mut World, aspect: f32, facts: FrameFacts) {
    let camera = *world.resource::<Camera>();
    let view_projection = camera.view_projection(aspect);
    let eye = camera.eye();
    let held = world.resource::<View>().state();
    if held.facts() == facts && held.view_projection() == view_projection && held.eye() == eye {
        return;
    }
    modify(world, |view: &mut View| {
        view.state_mut().set(facts, view_projection, eye);
    });
}

fn publish_poses(world: &mut World) {
    let Some(mut table) = world.modify_resource(Poses::take).ok().flatten() else {
        return;
    };
    table.clear();
    let pose = |matrix: Option<glam::Mat4>| matrix.map(glam::Affine3A::from_mat4);
    match world.resource::<PresentationState>().mode() {
        PresentationMode::Interpolated { .. } => {
            let globals = world.resource::<GlobalTransforms>();
            table.reserve(globals.len());
            for (entity, global) in globals.iter() {
                table.set(entity, pose(global.matrix()));
            }
        }
        PresentationMode::Current => {
            let presented = world.resource::<PresentedTransforms>();
            table.reserve(presented.len());
            for (entity, transform) in presented.iter() {
                table.set(entity, pose(transform.matrix()));
            }
        }
    }
    modify(world, |poses: &mut Poses| poses.hold(table));
}

impl Engine {
    pub fn new(config: EngineConfig) -> Self {
        let clock = FixedClock::new(config);
        let mut assets = AssetLibrary::default();
        assets
            .insert_cube()
            .expect("unlimited_library_holds_cube");
        assets.set_budget(config.asset_budget);
        let exchange = JobExchange::default();
        let listed = JobListed::of(exchange.list());
        let cores = std::thread::available_parallelism()
            .map_or(1, |cores| u32::try_from(cores.get()).unwrap_or(u32::MAX));
        let mut app = App::empty();
        app.world_mut()
            .resource_mut::<Schedules>()
            .remove(bevy_app::First);
        interpolation::track_markers(app.world_mut());
        app.add_observer(refuse_bevy_hierarchy);
        app.add_observer(note_tick_check);
        for schedule in Schedule::ALL {
            app.init_schedule(schedule.label())
                .edit_schedule(schedule.label(), single_threaded);
        }
        app.init_schedule(TransformPropagation)
            .add_systems(TransformPropagation, |world: &mut World| {
                propagate_transforms(world);
            })
            .edit_schedule(TransformPropagation, single_threaded)
            .insert_resource(FixedTime {
                delta_seconds: clock.step_seconds() as f32,
                elapsed_seconds: 0.0,
                tick: 0,
            })
            .init_resource::<FrameTime>()
            .init_resource::<Camera>()
            .init_resource::<Detail>()
            .init_resource::<JobSettings>()
            .init_resource::<CookMode>()
            .insert_resource(exchange)
            .insert_resource(listed)
            .init_resource::<engine_registry::JobQueue>()
            .insert_resource(Records::empty())
            .init_resource::<RecordExchange>()
            .insert_resource(View::empty())
            .insert_resource(Poses::empty())
            .insert_resource(DeviceLimits::without_device(cores))
            .init_resource::<WorkClock>()
            .init_resource::<PartDraws>()
            .init_resource::<PartFaults>()
            .init_resource::<WorkRemaining>()
            .init_resource::<mesh_lods::MeshLevels>()
            .init_resource::<Viewport>()
            .init_resource::<MoveInput>()
            .init_resource::<ActionInput>()
            .init_resource::<PresentationState>()
            .init_resource::<InterpolationState>()
            .init_resource::<TickCheck>()
            .insert_resource(TransformNodeLimit(config.max_transform_nodes))
            .init_resource::<GlobalTransforms>()
            .insert_resource(assets);
        assets::install(app.world_mut());
        let resets = ResetWatch::new(app.world_mut());
        Self {
            app,
            clock,
            resets,
            gates: None,
            facts: None,
            finalized: false,
            tick_ahead: TickAhead::None,
        }
    }

    pub fn listed(list: BrickList) -> Result<Self, Error> {
        let mut engine = Self::default();
        engine.install(&list)?;
        Ok(engine)
    }

    fn install(&mut self, list: &BrickList) -> Result<(), Error> {
        let jobs = list.job_list()?;
        let mut registry = Registry::default();
        registry.places(CORE_PLACES);
        for &places in list.place_sets() {
            registry.places(places);
        }
        for register in list.core_places() {
            register(&mut registry);
        }
        core_entries(&mut registry);
        for listed in list.listed() {
            registry.add(listed.record());
        }
        for &package in list.left_out() {
            registry.leave_out(package);
        }
        let world = self.app.world_mut();
        world.insert_resource(JobListed::of(&jobs));
        world.resource_mut::<JobExchange>().set_list(jobs);
        let labels: [InternedScheduleLabel; 12] = Schedule::ALL.map(Schedule::label);
        self.gates = Some(registry.install(world, &labels));
        let twice = world
            .get_resource::<MembersOf<SceneComponents>>()
            .map(|members| members.names_twice(|kind| kind.key()))
            .unwrap_or_default();
        let mut registry = world.resource_mut::<Registry>();
        for error in twice {
            registry.report(error);
        }
        refused(registry.problems())
    }

    #[track_caller]
    pub fn settings<S: Section + Resource<Mutability = Immutable>>(&mut self) -> &mut Self {
        let entry = bricks::settings_entry::<S>();
        let world = self.app.world_mut();
        entry.owned().init(world);
        if let Some(mut registry) = world.get_resource_mut::<Registry>() {
            registry.core_settings(entry);
        }
        self
    }

    #[track_caller]
    pub fn operation<O: Operation>(&mut self) -> &mut Self {
        let entry =
            OperationEntry::of::<O::Params>(O::NAME, O::CONTROL, bricks::run_operation::<O>)
                .schemas(O::schema, O::result_schema);
        if let Some(mut registry) = self.app.world_mut().get_resource_mut::<Registry>() {
            registry.core_operation(entry);
        }
        self
    }

    #[doc(hidden)]
    pub fn set_frame(&mut self, facts: FrameFacts) {
        self.facts = Some(facts);
    }

    #[doc(hidden)]
    pub fn arrive(&mut self, key: RecordKey, record: Result<Arc<[u8]>, Error>) {
        modify(self.app.world_mut(), |records: &mut Records| {
            records.arrive(key, record);
        });
    }

    #[doc(hidden)]
    pub fn job_list(&self) -> Arc<JobList> {
        self.app.world().resource::<JobExchange>().shared_list()
    }

    pub fn pending(&self) -> u32 {
        let world = self.app.world();
        let bricks = world
            .get_resource::<Registry>()
            .map_or(0, |registry| registry.pending(world));
        let records = world
            .get_resource::<RecordExchange>()
            .map_or(0, RecordExchange::pending);
        let levels = world
            .get_resource::<mesh_lods::MeshLevels>()
            .map_or(0, mesh_lods::MeshLevels::waiting);
        let impostors = assets::assets_pending(world);
        let core = u64::try_from(records + levels)
            .unwrap_or(u64::MAX)
            .saturating_add(impostors);
        bricks.saturating_add(u32::try_from(core).unwrap_or(u32::MAX))
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

    pub fn between_frames(&mut self, micros: u32) -> bool {
        self.finalize();
        let world = self.app.world_mut();
        modify(world, |frame: &mut FrameTime| {
            frame.between_frames_micros = micros;
        });
        if world.resource::<WorkRemaining>().remaining() {
            world.resource_mut::<WorkRemaining>().clear();
        }
        if has_systems(world, BetweenFrames) {
            {
                let _profile = engine_profile::span("ecs", "BetweenFrames");
                world.run_schedule(BetweenFrames);
            }
            jobs::admit_orders(world);
        }
        world.resource::<WorkRemaining>().remaining()
    }

    pub fn start_frame(&mut self) {
        self.finalize();
        let world = self.app.world_mut();
        if has_systems(world, FrameStart) {
            world.run_schedule(FrameStart);
        }
    }

    pub fn rendered(&mut self, submitted: bool) {
        self.finalize();
        let world = self.app.world_mut();
        if world.resource::<View>().submitted() != submitted {
            modify(world, |view: &mut View| {
                let state = view.state_mut();
                let facts = FrameFacts {
                    submitted,
                    ..state.facts()
                };
                let (view_projection, eye) = (state.view_projection(), state.eye());
                state.set(facts, view_projection, eye);
            });
        }
        if has_systems(world, Rendered) {
            world.run_schedule(Rendered);
        }
    }

    pub fn stream_now(&mut self) {
        self.finalize();
        let world = self.app.world_mut();
        if !has_systems(world, Waiting) {
            return;
        }
        {
            let _profile = engine_profile::span("ecs", "Waiting");
            world.run_schedule(Waiting);
        }
        jobs::admit_orders(world);
    }

    fn run_gates(&mut self) {
        let Some(gates) = self.gates.as_mut() else {
            return;
        };
        let world = self.app.world_mut();
        {
            let _profile = engine_profile::span("ecs", "Gates");
            gates.run(world);
        }
        let faults = gates.take_faults();
        if faults.is_empty() {
            return;
        }
        if let Some(mut registry) = world.get_resource_mut::<Registry>() {
            for (brick, fault) in faults {
                registry.report(
                    Error::new(ErrorCode::MissingResource, format!("$.bricks.{brick}.when"))
                        .got(fault.to_string()),
                );
            }
        }
    }

    fn check_gate_ticks(&mut self) {
        let world = self.app.world_mut();
        if world.resource::<TickCheck>().0.is_none() {
            return;
        }
        let check = world.resource_mut::<TickCheck>().0.take();
        if let Some(check) = check
            && let Some(mut hooks) = world.get_resource_mut::<BrickHooks>()
        {
            hooks.check_change_ticks(check);
        }
        if let (Some(check), Some(gates)) = (check, self.gates.as_mut()) {
            gates.check_change_ticks(check);
        }
    }

    fn run_frame(&mut self, fixed_steps: u32, delta_seconds: f64) {
        let _profile = engine_profile::span("cpu", "simulation_frame");
        self.run_gates();
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
        jobs::exchange_orders(self.app.world_mut());
        {
            let _profile = engine_profile::span("ecs", "Presentation");
            self.app.world_mut().run_schedule(Presentation);
        }
        if self.tick_ahead != TickAhead::InFlight {
            self.refresh_transforms();
        }
        self.app.world_mut().clear_trackers();
        self.check_gate_ticks();
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

    pub fn presentation(&self) -> &PresentationState {
        self.app.world().resource::<PresentationState>()
    }

    pub fn extract_frame(&mut self, aspect: f32) -> RenderFrame<'_> {
        let _profile = engine_profile::span("cpu", "extraction");
        self.finalize();
        self.complete_pending_tick();
        self.run_gates();
        self.mark_presented_time(None);
        self.sync_world();
        self.refresh_transforms();
        interpolation::prepare(self.app.world_mut());
        self.sync_view(aspect, true);
        RenderFrame {
            world: self.app.world(),
            gpu: false,
        }
    }

    pub fn extract_gpu_frame(&mut self, aspect: f32, alpha: f32) -> Result<RenderFrame<'_>, Error> {
        let _profile = engine_profile::span("cpu", "extraction");
        PresentationMode::Interpolated { alpha }.validate()?;
        self.finalize();
        self.run_gates();
        self.mark_presented_time(Some(alpha));
        if let Some(gathered) = self.tick_ahead.presents_previous() {
            interpolation::present_gpu_fraction(self.app.world_mut(), alpha, gathered);
        } else {
            self.sync_world();
            self.refresh_transforms();
            interpolation::prepare_gpu(self.app.world_mut(), alpha);
        }
        self.sync_view(aspect, false);
        Ok(RenderFrame {
            world: self.app.world(),
            gpu: true,
        })
    }

    fn sync_world(&mut self) {
        let _profile = engine_profile::span("ecs", "WorldSync");
        records::exchange(self.app.world_mut());
        jobs::exchange_orders(self.app.world_mut());
        mesh_lods::settle(self.app.world_mut());
        if has_systems(self.app.world(), Stream) {
            self.app.world_mut().run_schedule(Stream);
        }
        self.app.world_mut().run_schedule(WorldSync);
    }

    fn sync_view(&mut self, aspect: f32, exact: bool) {
        let _profile = engine_profile::span("ecs", "ViewSync");
        let facts = FrameFacts {
            exact,
            ..self.facts.unwrap_or_else(|| self.viewport_facts())
        };
        let world = self.app.world_mut();
        publish_poses(world);
        publish_view(world, aspect, facts);
        world.run_schedule(ViewSync);
        assets::settle_assets(world);
    }

    fn viewport_facts(&self) -> FrameFacts {
        let viewport = *self.app.world().resource::<Viewport>();
        let size = glam::UVec2::new(viewport.width, viewport.height);
        FrameFacts {
            display_size: size,
            render_size: size,
            ..FrameFacts::default()
        }
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
