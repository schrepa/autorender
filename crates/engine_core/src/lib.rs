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
mod shared;
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
    MAX_TEXTURE_DIMENSION, Material, MaterialBinding, Mesh, MeshLods, MeshVertex, Sampler, Surface,
    Texture, TextureFormat, TextureRef, TopLayer, UvTransform, Wrap, lod_input, trim_cards,
};
pub use engine_place::{
    BudgetUse, Census, Cursor, CursorView, DrawPoint, Error, ErrorCode, Events, Feed, FeedView,
    FrameBudget, GpuStruct, Interest, InterpolateTransform, JobRef, Level, LevelView, List,
    MaterialHandle, Mesh3d, MeshHandle, PlaceDecl, Progress, Readiness, Request, Requests, Ring,
    Runs, Screens, Since, Slot, Smoke, SmokeFailure, SmokeSession, Stamp, StoreId, TextureHandle,
    Ticket, gpu_consts, smoke_check, smoke_eq,
};
pub use engine_registry::{AuthoredId, ExplainAsk, ExplainRecord, FrameFacts, RecordParity};
pub use engine_settings::{
    Allowed, Expected, ExtraRules, ID_PATTERN, InvalidSetting, MAX_ID_BYTES, Rule, Setting, fields,
    is_id, parse_value, parse_value_at, rule, settings, state, stated, stats, valid_uri,
};
pub use frame_stats::*;
pub use glam;
pub use hierarchy::*;
pub use input::*;
pub use interpolation::*;
pub use jobs::*;
pub use mesh_lods::MeshLevels;
pub use nearest::Nearest;
pub use provided::*;
pub use records::*;
pub use scene::*;
pub use engine_place::{
    BitField, BitValue, FrameId, GpuBits, Readable, Readback, decode_bits, gpu_bits,
};
pub use section::{Cut, Explain, Operation, RESERVED_REPLY_KEYS, Section};
pub use shared::{FeedSettings, FrameBudgetSettings};
pub use services::{
    Authored, ComponentKind, ComponentKindOf, Detail, DeviceLimits, PartDraws, Placed,
    Placement, Prepared, Reach, SceneBuffers, SceneComponents, SceneContext, SceneCx, SceneIds,
    ScenePath, View, ViewWind, WorkClock, WorkRemaining, scene_component, scene_components,
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
use bevy_ecs::component::{ComponentId, Immutable};
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
    RegistryBuilder,
};
use interpolation::{InterpolationState, ResetWatch};
use schedules::{
    BetweenFrames, FixedMove, FixedPostSimulation, FixedPreSimulation, FixedSimulation, FrameStart,
    Presentation, Stream, TransformPropagation, ViewSync, Waiting, WorldSync,
};

pub const MAX_DEV_TICKS: u32 = 120;

static CORE_PLACES: &[PlaceDecl] = &[
    PlaceDecl::list::<SceneComponents>(),
    PlaceDecl::list::<CursorRequests>(),
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
    good_camera: Camera,
    finalized: bool,
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct RenderFrame<'a> {
    world: &'a World,
}

impl<'a> RenderFrame<'a> {
    pub fn world(self) -> &'a World {
        self.world
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

fn core_entries(registry: &mut RegistryBuilder) {
    registry.core_settings(bricks::settings_entry::<Detail>());
    registry.core_settings(bricks::settings_entry::<FeedSettings>());
    registry.core_settings(bricks::settings_entry::<FrameBudgetSettings>());
    registry.core_settings(bricks::settings_entry::<FrameLimits>());
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
    registry.core_operation(OperationEntry::of::<
        <clock::TimeOperation as Operation>::Params,
    >(
        clock::TimeOperation::NAME,
        clock::TimeOperation::CONTROL,
        bricks::run_operation::<clock::TimeOperation>,
    ));
    registry.core_operation(
        OperationEntry::of::<<clock::TimeState as Operation>::Params>(
            clock::TimeState::NAME,
            clock::TimeState::CONTROL,
            bricks::run_operation::<clock::TimeState>,
        ),
    );
    registry.writers_first::<Camera>();
    registry.writers_first::<TimeControl>();
    registry.writers_first::<Screens>();
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

impl Engine {
    pub fn new(config: EngineConfig) -> Self {
        let clock = FixedClock::new(config);
        let mut assets = AssetLibrary::default();
        assets.insert_cube().expect("unlimited_library_holds_cube");
        assets.set_budget(config.asset_budget);
        let exchange = JobExchange::default();
        let listed = JobListed::of(exchange.list());
        let cores = std::thread::available_parallelism()
            .map_or(1, |cores| u32::try_from(cores.get()).unwrap_or(u32::MAX));
        let mut app = App::empty();
        app.world_mut()
            .resource_mut::<Schedules>()
            .remove(bevy_app::First);
        hierarchy::tracking::track_switches(app.world_mut());
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
            .init_resource::<TimeControl>()
            .init_resource::<FrameLimits>()
            .init_resource::<Screens>()
            .init_resource::<Level>()
            .init_resource::<Readiness>()
            .init_resource::<FrameBudget>()
            .init_resource::<Camera>()
            .init_resource::<CameraRefusals>()
            .init_resource::<Detail>()
            .init_resource::<FeedSettings>()
            .init_resource::<FrameBudgetSettings>()
            .init_resource::<CookMode>()
            .insert_resource(exchange)
            .insert_resource(listed)
            .init_resource::<engine_registry::JobQueue>()
            .init_resource::<Records>()
            .init_resource::<RecordExchange>()
            .insert_resource(View::empty())
            .insert_resource(DeviceLimits::without_device(cores))
            .init_resource::<WorkClock>()
            .init_resource::<PartDraws>()
            .init_resource::<WorkRemaining>()
            .init_resource::<MeshLevels>()
            .init_resource::<Viewport>()
            .init_resource::<MoveInput>()
            .init_resource::<ActionInput>()
            .init_resource::<AxisInput>()
            .init_resource::<PointerInput>()
            .init_resource::<MenuInput>()
            .init_resource::<CursorRequests>()
            .init_resource::<ProvidedAssets>()
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
            good_camera: Camera::default(),
            finalized: false,
        }
    }

    pub fn listed(list: BrickList) -> Result<Self, Error> {
        let mut engine = Self::default();
        engine.install(&list)?;
        Ok(engine)
    }

    fn install(&mut self, list: &BrickList) -> Result<(), Error> {
        let jobs = list.job_list()?;
        let mut registry = RegistryBuilder::default();
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
        for left in list.left() {
            registry.leave_out(&left.record());
        }
        let world = self.app.world_mut();
        world.insert_resource(JobListed::of(&jobs));
        world.resource_mut::<JobExchange>().set_list(jobs);
        let labels: [InternedScheduleLabel; 11] = Schedule::ALL.map(Schedule::label);
        let mut gates = registry.install(world, &labels);
        let twice = world
            .get_resource::<MembersOf<SceneComponents>>()
            .map(|members| members.names_twice(|kind| kind.key()))
            .unwrap_or_default();
        let registrars: Vec<fn(&mut World) -> ComponentId> = world
            .get_resource::<MembersOf<SceneComponents>>()
            .map(|members| members.all().iter().map(ComponentKind::registrar).collect())
            .unwrap_or_default();
        let authored: Vec<ComponentId> = registrars
            .into_iter()
            .map(|register| register(world))
            .collect();
        let mut registry = world.resource_mut::<Registry>();
        registry.keep_authored(&authored);
        for error in twice {
            registry.report(error);
        }
        let checked = refused(registry.problems());
        gates.run(world);
        self.gates = Some(gates);
        checked
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

    pub fn ask_on(&mut self, package: &str) -> Result<(), Error> {
        let world = self.app.world_mut();
        let brick = world
            .get_resource::<Registry>()
            .and_then(|registry| registry.id(package))
            .ok_or_else(|| Error::new(ErrorCode::UnknownValue, "$.brick").got(package))?;
        engine_registry::ask_on(world, brick);
        Ok(())
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

    pub fn readiness(&self) -> &Readiness {
        self.app.world().resource::<Readiness>()
    }

    pub fn budget(&self) -> &FrameBudget {
        self.app.world().resource::<FrameBudget>()
    }

    pub fn set_compiling(&mut self, compiling: u32) {
        let world = self.app.world_mut();
        if world.resource::<Readiness>().compiling != compiling {
            world.resource_mut::<Readiness>().compiling = compiling;
        }
    }

    pub fn advance(&mut self, delta_seconds: f64) -> FrameReport {
        self.finalize();
        let delta_seconds = clock::sanitize_delta(delta_seconds);
        let rate = self.time_rate();
        let limits = *self.app.world().resource::<FrameLimits>();
        let report = self
            .clock
            .advance(delta_seconds * rate, limits.max_fixed_steps);
        let shown = delta_seconds.min(limits.max_frame_seconds);
        self.run_frame(report.fixed_steps, shown, shown * rate);
        FrameReport {
            cut_seconds: delta_seconds - shown,
            ..report
        }
    }

    fn time_rate(&self) -> f64 {
        self.app.world().resource::<TimeControl>().rate()
    }

    #[cfg(feature = "dev")]
    pub fn step_fixed(&mut self, ticks: u32) -> Result<FrameReport, Error> {
        check_dev_ticks(ticks)?;
        self.finalize();
        self.clock.reset();
        let delta_seconds = self.clock.step_seconds();
        let mut fixed_steps = 0;
        for _ in 0..ticks {
            let stopped = self.time_rate() == 0.0;
            let (steps, game_seconds) = if stopped {
                (0, 0.0)
            } else {
                (1, delta_seconds)
            };
            self.run_frame(steps, delta_seconds, game_seconds);
            fixed_steps += steps;
        }
        Ok(FrameReport {
            fixed_steps,
            dropped_seconds: 0.0,
            cut_seconds: 0.0,
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
            jobs::exchange_orders(world);
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
        jobs::exchange_orders(world);
    }

    fn run_gates(&mut self) {
        let Some(gates) = self.gates.as_mut() else {
            return;
        };
        let world = self.app.world_mut();
        let _profile = engine_profile::span("ecs", "Gates");
        gates.run(world);
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

    fn run_frame(&mut self, fixed_steps: u32, delta_seconds: f64, game_seconds: f64) {
        let _profile = engine_profile::span("cpu", "simulation_frame");
        self.run_gates();
        self.refresh_transforms();
        modify(self.app.world_mut(), |frame: &mut FrameTime| {
            frame.delta_seconds = delta_seconds as f32;
            frame.elapsed_seconds += delta_seconds;
            frame.game_delta_seconds = game_seconds as f32;
            frame.game_elapsed_seconds += game_seconds;
            frame.frame = frame.frame.saturating_add(1);
        });
        for _ in 0..fixed_steps {
            self.run_fixed_tick();
        }
        if self.time_rate() == 0.0 {
            modify(self.app.world_mut(), ActionInput::drop_tick_edges);
        }
        records::exchange(self.app.world_mut());
        jobs::exchange_orders(self.app.world_mut());
        modify(self.app.world_mut(), ActionInput::begin_frame);
        {
            let _profile = engine_profile::span("ecs", "Presentation");
            self.app.world_mut().run_schedule(Presentation);
        }
        self.keep_camera();
        self.refresh_transforms();
        self.app.world_mut().clear_trackers();
        self.check_gate_ticks();
    }

    fn keep_camera(&mut self) {
        let world = self.app.world_mut();
        let camera = *world.resource::<Camera>();
        match camera.check() {
            Ok(()) => self.good_camera = camera,
            Err(error) => {
                let good = self.good_camera;
                modify(world, |held: &mut Camera| *held = good);
                modify(world, |refusals: &mut CameraRefusals| {
                    refusals.count += 1;
                    refusals.latest = Some(error.path);
                });
            }
        }
    }

    pub fn gpu_current(&self) -> Option<GpuCurrent<'_>> {
        let world = self.app.world();
        world
            .get_resource::<PresentedTransforms>()?
            .gpu_current(world.get_resource::<GlobalTransforms>()?)
    }

    fn run_fixed_tick(&mut self) {
        let _profile = engine_profile::span("cpu", "fixed_tick");
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
        self.refresh_transforms();
        if has_systems(self.app.world(), FixedPostSimulation) {
            {
                let _profile = engine_profile::span("ecs", "FixedPostSimulation");
                self.app.world_mut().run_schedule(FixedPostSimulation);
            }
            self.refresh_transforms();
        }
        interpolation::end_tick(self.app.world_mut());
        modify(self.app.world_mut(), ActionInput::hide_edges);
    }

    pub fn reset_clock(&mut self) {
        self.clock.reset();
        let world = self.app.world_mut();
        reset_all_interpolation(world);
        modify(world, ActionInput::drop_tick_edges);
        modify(world, |frame: &mut FrameTime| {
            frame.delta_seconds = 0.0;
            frame.game_delta_seconds = 0.0;
        });
    }

    pub fn refresh_transforms(&mut self) {
        self.app.world_mut().run_schedule(TransformPropagation);
    }

    pub fn presentation(&self) -> &PresentationState {
        self.app.world().resource::<PresentationState>()
    }

    pub fn extract_frame(
        &mut self,
        aspect: f32,
        presented: Presented,
    ) -> Result<RenderFrame<'_>, Error> {
        let _profile = engine_profile::span("cpu", "extraction");
        let alpha = presented.alpha();
        interpolation::check_alpha(alpha)?;
        self.finalize();
        self.run_gates();
        self.mark_presented_time(alpha);
        self.sync_world();
        self.refresh_transforms();
        interpolation::prepare(self.app.world_mut(), presented);
        self.sync_view(aspect);
        shared::settle_readiness(self.app.world_mut());
        modify(self.app.world_mut(), ActionInput::hide_edges);
        Ok(RenderFrame {
            world: self.app.world(),
        })
    }

    fn sync_world(&mut self) {
        let _profile = engine_profile::span("ecs", "WorldSync");
        shared::start_frame(self.app.world_mut());
        records::exchange(self.app.world_mut());
        jobs::exchange_orders(self.app.world_mut());
        mesh_lods::settle(self.app.world_mut());
        if has_systems(self.app.world(), Stream) {
            self.app.world_mut().run_schedule(Stream);
        }
        self.app.world_mut().run_schedule(WorldSync);
    }

    fn sync_view(&mut self, aspect: f32) {
        let _profile = engine_profile::span("ecs", "ViewSync");
        self.keep_camera();
        let facts = self.facts.unwrap_or_else(|| self.viewport_facts());
        let world = self.app.world_mut();
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

    fn mark_presented_time(&mut self, alpha: f32) {
        let behind = (1.0 - f64::from(alpha)) * self.clock.step_seconds();
        let world = self.app.world_mut();
        let fixed = world.resource::<FixedTime>().elapsed_seconds;
        modify(world, |frame: &mut FrameTime| {
            frame.presented_seconds = fixed - behind;
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
        let mut run: Vec<_> = Schedule::ALL
            .iter()
            .map(|schedule| schedule.label())
            .collect();
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
