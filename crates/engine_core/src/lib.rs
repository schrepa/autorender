#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

extern crate self as engine_core;

mod assets;
mod authoring;
mod background;
mod bricks;
mod clock;
mod cook;
mod cull;
mod cursor;
mod drawn;
mod drawn_areas;
mod entity_indices;
mod files;
mod folds;
mod frame_timing;
mod hierarchy;
mod input;
mod jobs;
mod loads;
pub mod math;
mod mesh_lods;
mod moves;
pub use engine_math::collections;
mod nearest;
mod presentation;
pub mod presentation_bounds;
mod provided;
mod records;
mod refusals;
mod render_stats;
mod scene;
mod schedules;
mod section;
mod services;
mod shared;
mod source;
mod statistics;
mod ticks;
mod transform_edits;
mod views;
mod why;

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
pub use cook::{Cook, CookAsks, CookList, cook};
pub use cull::{
    CullPlan, Culling, OcclusionCulling, SubpixelCulling, frustum_bounded, frustum_planes,
    frustum_sees,
};
pub use cursor::*;
pub use drawn_areas::{DrawnArea, DrawnAreas};
pub use drawn::{
    DRAWN_PART, DrawnBatch, DrawnCensus, DrawnFlags,
    DrawnRecord, DrawnSettings, DrawnState, DrawnTable, PlanState, drawn_wgsl,
};
pub use engine_assets::{
    Aabb, AlphaMode, Asset, AssetBudget, AssetLibrary, Attribute, Attributes, GeometryStream,
    AttributeFormat, InstanceInput, VertexSkin,
    ColorSpace, Compare, Cull, DepthBias, Filter, LightingModel, LodInput,
    LodLevel, Material, MaterialBinding, Mesh,
    MeshLods, MeshVertex, BASE_COLOR_MAP, EMISSIVE_MAP, MAPPED, MAPPED_MAPS, MAPPED_SLOTS,
    NORMAL_MAP, RenderState, Sampler, SurfaceSlot, Texture, TextureFormat, TextureRef, UvTransform, Wrap, attributes_wgsl,
    lod_input,
};
pub use engine_textures::{IMAGE_FORMATS, Image, ImageFormat, decode_image, image_format};
pub use engine_place::{
    Applied, ApplyAt, Ask, Asked, AttachPoint, Awaits, BlendedPass, Board, BudgetUse, Census,
    Clock, ColorPass, PositionReach,
    Cursor, DrawKind, Drawn, DrawnBounds, DrawnProjection, Error, ErrorCode, Events,
    Feed, FeedView, FrameBudget, GpuStruct, Hold, InstanceValues, Interest, JobRef,
    KindCell, KindDecl, KindJoin, KindSkip, Layers, Level, LevelView, List, MaterialHandle, GeometryHandle, Mobility,
    Moment, Mover, MoverId, Node, NodeId, NodeWord, Pace, Pass, PassBit, PassCull, PassEntry,
    PassMask, PassName, PassPurpose, PassRaster, PassRole, PassViews, PlaceDecl, Progress, Readiness, Request, Requests, Ring,
    ScenePass, ScenePasses, Since, Slot, SlotSet, Smoke,
    SmokeFailure, SmokeSession,
    Settling, Stamp, StoreId, TargetName, TextureHandle, Ticket, Unsettled, Writes, every,
    gpu_consts,
    never,
    smoke_check, smoke_eq,
};
pub use engine_place::{
    Clear, Jitter, OwnedView, OwnedViews, Rect, Target, View, ViewCut, ViewHistory, ViewJitter,
    ViewLayer,
};
pub use engine_registry::{AuthoredId, FrameFacts, RecordParity, fault_of};
pub use engine_settings::{
    Allowed, Expected, ExtraRules, ID_PATTERN, InvalidSetting, MAX_ID_BYTES, Rule, Setting, fields,
    is_id, parse_value, parse_value_at, rule, settings, state, stated, stats, valid_uri,
};
pub use background::{Background, Delivers, Landed, Taken};
pub use files::Files;
pub use folds::FoldOrder;
pub use frame_timing::{FrameTiming, GpuFrameTime, RenderScale};
pub use glam;
pub use hierarchy::*;
pub use input::*;
pub use jobs::*;
pub use loads::{Load, Loaded, Loads};
pub use mesh_lods::MeshLevels;
pub use moves::{Applies, Asks, HandOver, MOVER_PART, Moves, Teleport, mover_of, send_ask, take_asks};
pub use nearest::Nearest;
pub use presentation::*;
pub use provided::*;
pub use records::*;
pub use refusals::{Refusals, Refused};
pub use render_stats::{KindPass, KindPassState, LinkedStats, Names, RenderStats};
pub use scene::*;
pub use schedules::{RunPace, RunPoint};
pub use engine_place::{
    BitField, BitValue, FrameId, GpuBits, Readable, Readback, decode_bits, gpu_bits,
};
pub use engine_place::{
    AppliedView, AskedView, Call, Each, EntityIds, Event, EventDecl, EventPage, Lands, Operation,
    PendingView, RESERVED_REPLY_KEYS, Raises, ReadEvents, Refusal, Rows, SHAPED_BY,
    missing_events, open_parts, open_schema, page_of, shaped_by,
};
pub use engine_registry::SceneNames;
pub use engine_registry::call;
pub use section::{Cut, EventsQuery, Section};
pub use shared::{FeedSettings, FrameBudgetSettings};
pub use services::{
    Authored, ComponentKind, ComponentKindOf, Detail, DeviceLimits, PartDraws, Placed,
    Placement, Prepared, SceneBuffers, SceneComponents, SceneContext, SceneCx, SceneForm,
    SceneIds, SubScene, SubSceneEntity, WorkClock, scene_component, scene_components,
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
pub mod textures {
    pub use engine_assets::srgb_decoded;
    pub use engine_textures::{bc7, decode, decompress, plain, record};
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
    BRICK_STATE_EVENT, BrickChangeRow, BrickHooks, EventEntry, GateDriver, JobList, JobListed,
    MembersOf, OperationEntry, Registry, RegistryBuilder, brick_changes,
};
use schedules::{
    FixedMove, FixedPostSimulation, FixedPreSimulation, FixedSimulation, FrameStart, Presentation,
    TransformPropagation, ViewSync, WorldSync,
};

static CORE_PLACES: &[PlaceDecl] = &[
    PlaceDecl::list::<SceneComponents>(),
    PlaceDecl::list::<CursorRequests>(),
    PlaceDecl::list::<Contexts>(),
    PlaceDecl::list::<Verbs>(),
    PlaceDecl::list::<WholeFrames>(),
    PlaceDecl::list::<Pause>(),
    PlaceDecl::list::<CookList>(),
    PlaceDecl::list::<TextureJobs>(),
    PlaceDecl::ledger::<FrameBudget>("frame_budget"),
    PlaceDecl::ledger::<Readiness>("readiness"),
    PlaceDecl::ledger::<AssetLibrary>("asset_library"),
    PlaceDecl::ledger::<RecordExchange>("record_exchange"),
    PlaceDecl::ledger::<Refusals>("refusals"),
    PlaceDecl::ledger::<Files>("files"),
    PlaceDecl::ledger::<DrawnAreas>("drawn_areas"),
    PlaceDecl::component::<Drawn>("drawn"),
    PlaceDecl::component::<OwnedViews>("owned_views"),
    PlaceDecl::ask::<Teleport>(),
    PlaceDecl::ask::<HandOver>(),
    PlaceDecl::ask::<ViewCut>(),
    PlaceDecl::component::<View>("view"),
    PlaceDecl::slot::<Jitter>(),
    PlaceDecl::slot::<RenderScale>(),
    PlaceDecl::component::<Camera>("camera"),
];

#[cfg(feature = "dev")]
pub fn check_dev_ticks(ticks: u32) -> Result<(), Error> {
    if ticks >= 1 {
        Ok(())
    } else {
        Err(Error::new(ErrorCode::InvalidStep, "$.ticks")
            .at_least(1)
            .got(ticks))
    }
}

pub struct Engine {
    pub app: App,
    clock: FixedClock,
    interval: Interval,
    gates: Option<GateDriver>,
    facts: Option<FrameFacts>,
    spare_ms: f64,
    finalized: bool,
    board: Board,
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
    registry.core_settings(bricks::settings_entry::<Culling>());
    registry.core_settings(bricks::settings_entry::<OcclusionCulling>());
    registry.core_settings(bricks::settings_entry::<SubpixelCulling>());
    registry.core_settings(bricks::settings_entry::<FrameLimits>());
    registry.core_settings(bricks::settings_entry::<InputSettings>());
    registry.core_settings(bricks::settings_entry::<FoldOrder>());
    registry.core_operation(OperationEntry::of(Call::of::<views::ViewControl>()));
    registry.core_operation(OperationEntry::of(Call::of::<moves::TeleportControl>()));
    registry.core_operation(OperationEntry::of(authoring::CHANGE_CALL));
    registry.core_operation(OperationEntry::of(Call::of::<moves::MovesState>()));
    registry.core_operation(OperationEntry::of(Call::of::<views::ViewsState>()));
    registry.core_operation(OperationEntry::of(Call::of::<input::Rebind>()));
    registry.core_operation(OperationEntry::of(Call::of::<why::Why>().observed()));
    registry.core_operation(OperationEntry::of(Call::of::<EventsQuery>()));
    registry.core_event(EventEntry::of(EventDecl::custom::<BrickChangeRow>(
        BRICK_STATE_EVENT,
        brick_changes,
    )));
    registry.core_event(EventEntry::of(EventDecl::of::<InputChange, InputChanges>()));
    registry.core_event(EventEntry::of(EventDecl::of::<Load, Loads>()));
    registry.mover_rule(moves::rule());
    registry.despawn_rule(hierarchy::despawn_owned);
    registry.fill_core::<Asked<Teleport>>();
    registry.fill_core::<Asked<HandOver>>();
    registry.fill_core::<Asked<ViewCut>>();
    registry.writers_first::<Pause>();
    registry.writers_first::<TimeScale>();
    registry.writers_first::<JobExchange>();
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
    let kinds = world
        .get_resource::<DrawnTable>()
        .map(|table| table.kinds().to_vec())
        .unwrap_or_default();
    let (passes, problems) = ScenePasses::of(registered, kinds);
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

impl Engine {
    pub fn new(config: EngineConfig) -> Self {
        let board = Board::default();
        board.scope(|| Self::on_board(config, board.clone()))
    }

    fn on_board(config: EngineConfig, board: Board) -> Self {
        let clock = FixedClock::new(config);
        let assets = AssetLibrary::new(config.asset_budget);
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
        app.add_observer(services::scene_entity_inserted);
        app.add_observer(services::scene_entity_discarded);
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
            .init_resource::<FrameTiming>()
            .init_resource::<Pause>()
            .init_resource::<TimeScale>()
            .init_resource::<FrameLimits>()
            .init_resource::<Level>()
            .init_resource::<Readiness>()
            .init_resource::<Refusals>()
            .init_resource::<FrameBudget>()
            .init_resource::<Views>()
            .init_resource::<Detail>()
            .init_resource::<FeedSettings>()
            .init_resource::<FrameBudgetSettings>()
            .init_resource::<DrawnSettings>()
            .init_resource::<Culling>()
            .init_resource::<OcclusionCulling>()
            .init_resource::<SubpixelCulling>()
            .init_resource::<CullPlan>()
            .insert_resource(exchange)
            .insert_resource(listed)
            .init_resource::<engine_registry::JobQueue>()
            .init_resource::<Records>()
            .init_resource::<Files>()
            .init_resource::<RecordExchange>()
            .insert_resource(DeviceLimits::without_device(cores))
            .init_resource::<WorkClock>()
            .init_resource::<PartDraws>()
            .init_resource::<MeshLevels>()
            .init_resource::<Input>()
            .init_resource::<InputChanges>()
            .init_resource::<Loads>()
            .init_resource::<Devices>()
            .init_resource::<Contexts>()
            .init_resource::<InputSettings>()
            .init_resource::<FoldOrder>()
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
            spare_ms: 0.0,
            finalized: false,
            board,
        }
    }

    fn scoped<R>(&mut self, run: impl FnOnce(&mut Self) -> R) -> R {
        let board = self.board.clone();
        board.scope(|| run(self))
    }

    pub fn listed(list: BrickList) -> Result<Self, Error> {
        let mut engine = Self::default();
        engine.scoped(|engine| engine.install(&list))?;
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
        let labels: [InternedScheduleLabel; 8] = Schedule::ALL.map(Schedule::label);
        let mut gates = registry.install(world, &labels);
        for default in list.defaults() {
            default(world);
        }
        moves::install_kinds(world);
        drawn::install_kinds(world);
        install_passes(world);
        records::install_jobs(world);
        let mut twice = world
            .get_resource::<MembersOf<SceneComponents>>()
            .map(|members| members.names_twice(|kind| kind.key()))
            .unwrap_or_default();
        twice.extend(authoring::keys_taken(world));
        twice.extend(input::install(world));
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
        self.core_operation(OperationEntry::of(Call::of::<O>()))
    }

    pub fn core_operation(&mut self, entry: OperationEntry) -> &mut Self {
        if let Some(mut registry) = self.app.world_mut().get_resource_mut::<Registry>() {
            registry.core_operation(entry);
        }
        self
    }

    pub fn ask_on(&mut self, package: &str) -> Result<(), Error> {
        self.scoped(|engine| {
            let world = engine.app.world_mut();
            let brick = world
                .get_resource::<Registry>()
                .and_then(|registry| registry.id(package))
                .ok_or_else(|| Error::new(ErrorCode::UnknownValue, "$.brick").got(package))?;
            engine_registry::ask_on(world, brick);
            Ok(())
        })
    }

    #[doc(hidden)]
    pub fn set_frame(&mut self, facts: FrameFacts) {
        self.facts = Some(facts);
    }

    #[doc(hidden)]
    pub fn land_job(&mut self, id: JobId, output: Result<Arc<[u8]>, String>) {
        jobs::land_job(self.app.world_mut(), id, output);
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
        self.scoped(|engine| engine.frame_at(timestamp_seconds));
        Ok(())
    }

    fn frame_at(&mut self, timestamp_seconds: f64) {
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
    }

    pub fn present(&mut self, presented: Presented) -> Result<(), Error> {
        presentation::check_alpha(presented.alpha())?;
        self.scoped(|engine| {
            engine.finalize();
            if moves::start_frame(engine.app.world_mut()) {
                engine.refresh_transforms();
            }
            presentation::present(engine.app.world_mut(), presented);
        });
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
        self.scoped(|engine| engine.step_ticks(ticks));
        Ok(())
    }

    #[cfg(feature = "dev")]
    pub fn frame_without_ticks(&mut self) {
        self.scoped(|engine| {
            engine.finalize();
            engine.clock.reset();
            engine.interval = engine.interval.next(None).0;
            let step = engine.clock.step_seconds();
            let advanced = Advanced {
                ticks: 0,
                dropped_seconds: 0.0,
                alpha: 1.0,
            };
            engine.run_frame(Presented::Tick, advanced, (step, 0.0), 0.0);
        });
    }

    #[cfg(feature = "dev")]
    fn step_ticks(&mut self, ticks: u32) {
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
    }

    pub fn set_spare_time(&mut self, milliseconds: f64) {
        self.spare_ms = if milliseconds.is_finite() {
            milliseconds.max(0.0)
        } else {
            0.0
        };
    }

    pub fn start_frame(&mut self) {
        self.scoped(|engine| {
            engine.finalize();
            let world = engine.app.world_mut();
            if has_systems(world, FrameStart) {
                world.run_schedule(FrameStart);
            }
        });
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
            modify(self.app.world_mut(), Input::drop_tick_edges);
        }
        records::exchange(self.app.world_mut());
        files::read_files(self.app.world_mut());
        jobs::exchange_orders(self.app.world_mut());
        modify(self.app.world_mut(), Input::begin_frame);
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
        modify(self.app.world_mut(), |input: &mut Input| {
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
        modify(self.app.world_mut(), Input::hide_edges);
    }

    pub fn reset_clock(&mut self) {
        self.clock.reset();
        self.interval = self.interval.next(None).0;
        let world = self.app.world_mut();
        presentation::fresh_all(world);
        modify(world, Input::drop_tick_edges);
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
        self.scoped(|engine| {
            engine.finalize();
            engine.run_gates();
            engine.sync_world();
            engine.refresh_transforms();
            presentation::gather(engine.app.world_mut());
            engine.sync_view();
            shared::settle_readiness(engine.app.world_mut(), &engine.board);
            modify(engine.app.world_mut(), Input::hide_edges);
        });
        Ok(RenderFrame {
            world: self.app.world(),
        })
    }

    fn sync_world(&mut self) {
        let _profile = engine_profile::span("ecs", "WorldSync");
        let spare_ms = std::mem::take(&mut self.spare_ms);
        shared::start_frame(self.app.world_mut(), spare_ms);
        records::exchange(self.app.world_mut());
        files::read_files(self.app.world_mut());
        jobs::exchange_orders(self.app.world_mut());
        mesh_lods::settle(self.app.world_mut());
        self.app.world_mut().run_schedule(WorldSync);
    }

    fn sync_view(&mut self) {
        let _profile = engine_profile::span("ecs", "ViewSync");
        let facts = self.facts.unwrap_or_default();
        let world = self.app.world_mut();
        views::settle(world, facts);
        assets::settle_assets(world);
        cull::settle(world);
        world.run_schedule(ViewSync);
        views::settle_owned(world);
        assets::settle_assets(world);
        cull::settle(world);
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
