#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

extern crate self as engine_core;

mod assets;
mod authoring;
pub mod bounds;
mod bricks;
mod clock;
pub mod command_line;
mod drawn;
mod entity_indices;
mod frame_stats;
mod hierarchy;
mod input;
mod jobs;
pub mod math;
mod mesh_lods;
mod moves;
pub use engine_math::collections;
mod nearest;
mod presentation;
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
mod views;

mod sources {
    include!(concat!(env!("OUT_DIR"), "/sources.rs"));
}

pub mod profile {
    pub use engine_profile::{Aggregate, Span, active, counter, now_ms, span, tally, waited};
}

pub use assets::*;
pub use authoring::{DrawnForm, ViewForm, core_components};
pub use bevy_app::App;
pub use bricks::*;
pub use clock::*;
pub use drawn::{
    DRAWN_OWN_PROJECTION, DRAWN_UNBOUNDED, DrawnBatch, DrawnCensus,
    DrawnRecord, DrawnSettings, DrawnState, DrawnTable, PlanState, drawn_wgsl,
};
pub use engine_assets::{
    Aabb, AlphaImage, AlphaMode, Asset, AssetBudget, AssetLibrary, Attribute, Attributes,
    BASE_COLOR_MAP, ColorSpace, Cull, DepthBias, EMISSIVE_MAP, Filter, LightingModel, LodInput,
    LodLevel, MAX_TEXTURE_DIMENSION, METALLIC_ROUGHNESS_MAP, Material, MaterialBinding, Mesh,
    MeshLods, MeshVertex, NORMAL_MAP, OCCLUSION_MAP, RenderState, STANDARD_MAPS, Sampler,
    Surface, SurfaceSlot, Texture, TextureFormat, TextureRef, UvTransform, Wrap, attributes_wgsl,
    lod_input, trim_cards,
};
pub use engine_textures::{IMAGE_FORMATS, Image, ImageFormat, decode_image, image_format};
pub use engine_place::{
    Applied, ApplyAt, Ask, Asked, AttachPoint, BlendedPass, BudgetUse, CORE_PASSES, Census,
    Clock, ColorPass, PositionReach,
    Cursor, CursorView, DrawKind, Drawn, DrawnProjection, Error, ErrorCode, Events,
    Feed, FeedView, FrameBudget, GeometryHandle, GpuStruct, InstanceValues, Interest, JobRef,
    KindDecl, Layers, Level, LevelView, List, MaterialHandle, MeshHandle, Mobility, Moment,
    Mover, MoverId, Node, NodeId, NodeWord, PASS_BITS, Pace, Pass, PassBit, PassCull, PassEntry,
    PassMask, PassName, PassRaster, PassViews, PlaceDecl, Progress, Readiness, Request, Requests, Ring, Runs,
    ScenePass, ScenePasses, Since, Slot, SlotSet, Smoke,
    SmokeFailure, SmokeSession,
    Stamp, StoreId, TargetName, TextureHandle, Ticket, Writes, every, gpu_consts, never,
    smoke_check, smoke_eq,
};
pub use engine_place::{
    Clear, Jitter, OwnedView, OwnedViews, Rect, Target, View, ViewCut, ViewHistory, ViewJitter,
    ViewLayer,
};
pub use engine_registry::{
    AuthoredId, ExplainAsk, ExplainRecord, FrameFacts, RecordParity, fault_of,
};
pub use engine_settings::{
    Allowed, Expected, ExtraRules, ID_PATTERN, InvalidSetting, MAX_ID_BYTES, Rule, Setting, fields,
    is_id, parse_value, parse_value_at, rule, settings, state, stated, stats, valid_uri,
};
pub use frame_stats::*;
pub use glam;
pub use hierarchy::*;
pub use input::*;
pub use jobs::*;
pub use mesh_lods::MeshLevels;
pub use moves::{
    Applies, Asks, CameraHandoff, HandOver, Moves, Teleport, mover_of, send_ask, take_asks,
};
pub use nearest::Nearest;
pub use presentation::*;
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
    Placement, Prepared, Reach, SceneBuffers, SceneComponents, SceneContext, SceneCx, SceneForm,
    SceneIds, PrimaryView, SceneShare, WorkClock, WorkRemaining, scene_component, scene_components,
};
pub use source::{Entry, Module, ShaderText, Source, Var};
pub use statistics::{Distribution, median};
pub use ticks::ScanClock;
pub(crate) use transform_edits::edit_transform;
pub use views::{DrawnView, Hit, Picking, Ray, ViewFacts, Views, WholeFrames, main_camera};

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
use clock::{Advanced, FixedClock, Interval};
use engine_registry::{
    BrickHooks, GateDriver, JobList, JobListed, MembersOf, OperationEntry, Registry,
    RegistryBuilder,
};
use schedules::{
    BetweenFrames, FixedMove, FixedPostSimulation, FixedPreSimulation, FixedSimulation, FrameStart,
    Presentation, PreWorldSync, TransformPropagation, ViewSync, Waiting, WorldSync,
};

pub const MAX_DEV_TICKS: u32 = 120;

static CORE_PLACES: &[PlaceDecl] = &[
    PlaceDecl::list::<SceneComponents>(),
    PlaceDecl::list::<CursorRequests>(),
    PlaceDecl::list::<WholeFrames>(),
    PlaceDecl::list::<Pause>(),
    PlaceDecl::slot::<ImpostorPolicy>(),
    PlaceDecl::ledger::<FrameBudget>("frame_budget"),
    PlaceDecl::ledger::<Readiness>("readiness"),
    PlaceDecl::ledger::<AssetLibrary>("asset_library"),
    PlaceDecl::component::<Drawn>("drawn"),
    PlaceDecl::component::<OwnedViews>("owned_views"),
    PlaceDecl::ask::<Teleport>(),
    PlaceDecl::ask::<HandOver>(),
    PlaceDecl::ask::<CameraHandoff>(),
    PlaceDecl::ask::<ViewCut>(),
    PlaceDecl::component::<View>("view"),
    PlaceDecl::slot::<Jitter>(),
    PlaceDecl::component::<Camera>("camera"),
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
    interval: Interval,
    gates: Option<GateDriver>,
    facts: Option<FrameFacts>,
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
    registry.core_settings(bricks::settings_entry::<DrawnSettings>());
    registry.core_settings(bricks::settings_entry::<FrameLimits>());
    registry.core_operation(OperationEntry::of::<
        <services::ViewControl as Operation>::Params,
    >(
        services::ViewControl::NAME,
        true,
        bricks::run_operation::<services::ViewControl>,
    ));
    registry.core_operation(OperationEntry::of::<
        <moves::TeleportControl as Operation>::Params,
    >(
        moves::TeleportControl::NAME,
        moves::TeleportControl::CONTROL,
        bricks::run_operation::<moves::TeleportControl>,
    ));
    registry.core_operation(OperationEntry::of::<()>(
        authoring::COMPONENT,
        true,
        authoring::run_component,
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
    registry.core_operation(OperationEntry::of::<
        <moves::MovesState as Operation>::Params,
    >(
        moves::MovesState::NAME,
        moves::MovesState::CONTROL,
        bricks::run_operation::<moves::MovesState>,
    ));
    registry.core_operation(OperationEntry::of::<
        <views::ViewsState as Operation>::Params,
    >(
        views::ViewsState::NAME,
        views::ViewsState::CONTROL,
        bricks::run_operation::<views::ViewsState>,
    ));
    registry.mover_rule(moves::rule());
    registry.despawn_rule(hierarchy::despawn_owned);
    registry.fill_core::<Asked<Teleport>>();
    registry.fill_core::<Asked<HandOver>>();
    registry.fill_core::<Asked<CameraHandoff>>();
    registry.fill_core::<Asked<ViewCut>>();
    registry.writers_first::<Pause>();
    registry.writers_first::<TimeScale>();
}

fn install_passes(world: &mut World) {
    let registered: Vec<(&'static str, PassEntry)> = world
        .get_resource::<Registry>()
        .into_iter()
        .flat_map(|registry| registry.iter())
        .flat_map(|(_, record)| {
            record
                .render_entries::<PassEntry>()
                .map(move |entry| (record.package, *entry))
        })
        .collect();
    let (passes, problems) = ScenePasses::of(registered);
    world.insert_resource(passes);
    if let Some(mut registry) = world.get_resource_mut::<Registry>() {
        for problem in problems {
            registry.report(problem);
        }
    }
}

fn refused(problems: &[Error]) -> Result<(), Error> {
    match problems {
        [] => Ok(()),
        [only] => Err(only.clone()),
        [first, ..] => Err(Error::new(first.code, "$.bricks").got(problems)),
    }
}

fn publish_view(world: &mut World, facts: FrameFacts) {
    let scene = world.resource::<PrimaryView>().scene();
    views::settle(world, facts, scene);
    let main = world.resource::<Views>().main().copied();
    let held = world.resource::<PrimaryView>();
    let (shown, view_projection, eye) = match main {
        Some(main) => (
            FrameFacts {
                display_size: main.size(),
                render_size: main.render,
                ..facts
            },
            main.view_projection,
            main.eye,
        ),
        None => (facts, held.view_projection(), held.eye()),
    };
    let state = held.state();
    if state.facts() == shown
        && state.view_projection() == view_projection
        && state.eye() == eye
        && held.same(main.as_ref())
    {
        return;
    }
    modify(world, |view: &mut PrimaryView| {
        view.state_mut().set(shown, view_projection, eye);
        view.shown(main);
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
        hierarchy::tracking::track_nodes(app.world_mut());
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
            .init_resource::<Pause>()
            .init_resource::<TimeScale>()
            .init_resource::<FrameLimits>()
            .init_resource::<Level>()
            .init_resource::<Readiness>()
            .init_resource::<FrameBudget>()
            .init_resource::<Views>()
            .init_resource::<Detail>()
            .init_resource::<FeedSettings>()
            .init_resource::<FrameBudgetSettings>()
            .init_resource::<DrawnSettings>()
            .init_resource::<CookMode>()
            .insert_resource(exchange)
            .insert_resource(listed)
            .init_resource::<engine_registry::JobQueue>()
            .init_resource::<Records>()
            .init_resource::<RecordExchange>()
            .insert_resource(PrimaryView::empty())
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
            .init_resource::<WholeFrames>()
            .init_resource::<ProvidedAssets>()
            .init_resource::<PresentationState>()
            .init_resource::<DrawnPoses>()
            .init_resource::<TickCheck>()
            .init_resource::<moves::Key>()
            .init_resource::<moves::MovingKinds>()
            .init_resource::<Asked<Teleport>>()
            .init_resource::<Asked<HandOver>>()
            .init_resource::<Asked<CameraHandoff>>()
            .init_resource::<Asked<ViewCut>>()
            .insert_resource(TransformNodeLimit(config.max_transform_nodes))
            .init_resource::<GlobalTransforms>()
            .insert_resource(assets);
        assets::install(app.world_mut());
        drawn::install(app.world_mut());
        Self {
            app,
            clock,
            interval: Interval::Fresh,
            gates: None,
            facts: None,
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
        moves::install_kinds(world);
        install_passes(world);
        drawn::install_kinds(world);
        scene::spawn_main_view(world);
        let mut twice = world
            .get_resource::<MembersOf<SceneComponents>>()
            .map(|members| members.names_twice(|kind| kind.key()))
            .unwrap_or_default();
        twice.extend(authoring::keys_taken(world));
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

    pub fn frame(&mut self, timestamp_seconds: f64) -> Result<(), Error> {
        if !timestamp_seconds.is_finite() {
            return Err(Error::new(ErrorCode::InvalidInput, "$.timestamp")
                .expected(Expected::Finite)
                .got(timestamp_seconds));
        }
        self.finalize();
        let (interval, real_seconds) = self.interval.next(Some(timestamp_seconds));
        self.interval = interval;
        let rate = self.time_rate();
        let limits = *self.app.world().resource::<FrameLimits>();
        let advanced = self
            .clock
            .advance(real_seconds * rate, limits.max_fixed_steps);
        let shown = real_seconds.min(limits.max_frame_seconds);
        self.run_frame(
            Presented::Paced {
                alpha: advanced.alpha,
            },
            advanced,
            (shown, shown * rate),
            real_seconds - shown,
        );
        Ok(())
    }

    pub fn present(&mut self, presented: Presented) -> Result<(), Error> {
        presentation::check_alpha(presented.alpha())?;
        self.finalize();
        if moves::start_frame(self.app.world_mut()) {
            self.refresh_transforms();
        }
        presentation::present(self.app.world_mut(), presented);
        Ok(())
    }

    pub fn pause(&mut self, reason: PauseReason) {
        let world = self.app.world_mut();
        if world.resource::<Pause>().holds(reason) {
            return;
        }
        world.resource_mut::<Pause>().add(reason);
        if reason.stops_frames() {
            self.interval = self.interval.next(None).0;
        }
    }

    pub fn resume(&mut self, reason: PauseReason) {
        let world = self.app.world_mut();
        if !world.resource::<Pause>().holds(reason) {
            return;
        }
        world.resource_mut::<Pause>().remove(reason);
        if reason.stops_frames() {
            self.interval = self.interval.next(None).0;
        }
    }

    pub fn frame_time(&self) -> FrameTime {
        *self.app.world().resource::<FrameTime>()
    }

    pub fn drawn_pose(&self, node: bevy_ecs::entity::Entity) -> Option<glam::Mat4> {
        let world = self.app.world();
        world
            .resource::<DrawnPoses>()
            .drawn_pose(world.get_resource::<GlobalTransforms>()?, node)
    }

    fn time_rate(&self) -> f64 {
        let world = self.app.world();
        clock::game_rate(world.resource::<Pause>(), *world.resource::<TimeScale>())
    }

    #[cfg(feature = "dev")]
    pub fn step_fixed(&mut self, ticks: u32) -> Result<(), Error> {
        check_dev_ticks(ticks)?;
        self.finalize();
        self.clock.reset();
        self.interval = self.interval.next(None).0;
        let step = self.clock.step_seconds();
        for _ in 0..ticks {
            let ticked = u32::from(self.time_rate() != 0.0);
            let advanced = Advanced {
                ticks: ticked,
                dropped_seconds: 0.0,
                alpha: 1.0,
            };
            self.run_frame(
                Presented::Tick,
                advanced,
                (step, step * f64::from(ticked)),
                0.0,
            );
        }
        Ok(())
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

    fn run_frame(
        &mut self,
        presented: Presented,
        advanced: Advanced,
        (real_seconds, seconds): (f64, f64),
        cut_seconds: f64,
    ) {
        let _profile = engine_profile::span("cpu", "simulation_frame");
        self.run_gates();
        self.refresh_transforms();
        let step = self.clock.step_seconds();
        let world = self.app.world_mut();
        let ticked =
            world.resource::<FixedTime>().elapsed_seconds + f64::from(advanced.ticks) * step;
        let alpha = presented.alpha();
        modify(world, |time: &mut FrameTime| {
            time.frame = time.frame.saturating_add(1);
            time.ticks = advanced.ticks;
            time.alpha = alpha;
            time.seconds = seconds;
            time.real_seconds = real_seconds;
            time.elapsed = ticked - (1.0 - f64::from(alpha)) * step;
            time.real_elapsed += real_seconds;
            time.dropped_seconds += advanced.dropped_seconds;
            time.cut_seconds += cut_seconds;
        });
        if moves::start_frame(world) {
            self.refresh_transforms();
        }
        for _ in 0..advanced.ticks {
            self.run_fixed_tick();
        }
        if self.app.world().resource::<Pause>().paused() {
            modify(self.app.world_mut(), ActionInput::drop_tick_edges);
        }
        records::exchange(self.app.world_mut());
        jobs::exchange_orders(self.app.world_mut());
        modify(self.app.world_mut(), ActionInput::begin_frame);
        presentation::present(self.app.world_mut(), presented);
        {
            let _profile = engine_profile::span("ecs", "Presentation");
            self.app.world_mut().run_schedule(Presentation);
        }
        self.refresh_transforms();
        self.app.world_mut().clear_trackers();
        self.check_gate_ticks();
    }

    fn run_fixed_tick(&mut self) {
        let _profile = engine_profile::span("cpu", "fixed_tick");
        let step = self.clock.step_seconds();
        let mut tick = 0;
        modify(self.app.world_mut(), |time: &mut FixedTime| {
            time.elapsed_seconds += step;
            time.tick = time.tick.saturating_add(1);
            tick = time.tick;
        });
        modify(self.app.world_mut(), |input: &mut ActionInput| {
            input.begin_tick(tick);
        });
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
        self.refresh_transforms();
        if has_systems(self.app.world(), FixedPostSimulation) {
            {
                let _profile = engine_profile::span("ecs", "FixedPostSimulation");
                self.app.world_mut().run_schedule(FixedPostSimulation);
            }
            self.refresh_transforms();
        }
        if moves::end_tick(self.app.world_mut()) {
            self.refresh_transforms();
        }
        presentation::capture(self.app.world_mut());
        modify(self.app.world_mut(), ActionInput::hide_edges);
    }

    pub fn reset_clock(&mut self) {
        self.clock.reset();
        self.interval = self.interval.next(None).0;
        let world = self.app.world_mut();
        presentation::fresh_all(world);
        modify(world, ActionInput::drop_tick_edges);
        modify(world, |time: &mut FrameTime| {
            time.ticks = 0;
            time.seconds = 0.0;
            time.real_seconds = 0.0;
        });
    }

    pub fn refresh_transforms(&mut self) {
        self.app.world_mut().run_schedule(TransformPropagation);
    }

    pub fn presentation(&self) -> &PresentationState {
        self.app.world().resource::<PresentationState>()
    }

    pub fn extract_frame(&mut self) -> Result<RenderFrame<'_>, Error> {
        let _profile = engine_profile::span("cpu", "extraction");
        self.finalize();
        self.run_gates();
        self.sync_world();
        self.refresh_transforms();
        presentation::gather(self.app.world_mut());
        self.sync_view();
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
        if has_systems(self.app.world(), PreWorldSync) {
            self.app.world_mut().run_schedule(PreWorldSync);
        }
        self.app.world_mut().run_schedule(WorldSync);
    }

    fn sync_view(&mut self) {
        let _profile = engine_profile::span("ecs", "ViewSync");
        let facts = self.facts.unwrap_or_else(|| self.viewport_facts());
        let world = self.app.world_mut();
        publish_view(world, facts);
        world.run_schedule(ViewSync);
        views::settle_owned(world);
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
