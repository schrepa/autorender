#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

extern crate self as engine_core;

mod admission;
mod assets;
mod authoring;
mod background;
mod bricks;
mod clock;
mod cook;
mod cursor;
mod drawn;
mod entity_indices;
mod files;
mod hierarchy;
mod input;
mod jobs;
pub mod math;
mod mesh_lods;
mod moves;
pub use engine_math::collections;
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

pub use admission::admit;
pub use assets::*;
pub use authoring::{DrawnForm, ViewForm, core_components};
pub use background::{Background, Delivers, Landed, Taken};
pub use bevy_app::App;
pub use bricks::*;
pub use clock::*;
pub use cook::{Cook, CookAsks, CookList, cook};
pub use cursor::*;
pub use drawn::{
    DRAWN_PART, DrawnBatch, DrawnRecord, DrawnSettings, DrawnState, DrawnTable, drawn_wgsl,
};
pub use engine_assets::{
    Aabb, AlphaMode, Asset, AssetBudget, AssetLibrary, Attribute, AttributeFormat, Attributes,
    BASE_COLOR_MAP, ColorSpace, Compare, Cull, DepthBias, EMISSIVE_MAP, FIRST_NAMED_LOCATION,
    Filter, GeometryStream, InstanceInput, LightingModel, LodInput, LodLevel, MAPPED, MAPPED_MAPS,
    MAPPED_SLOTS, Material, MaterialBinding, Mesh, MeshLods, MeshVertex, NORMAL_MAP, NamedStream,
    RenderState, Sampler, StreamShape, SurfaceSlot, Texture, TextureFormat, TextureRef,
    UvTransform, Wrap, attributes_wgsl, lod_input,
};
pub use engine_place::{
    AuthoredId, Named, OperationScene, SCENE_SCHEMA, SceneEntity, entity_removed, schema_of,
};
pub use engine_place::{
    Asked, Awaits, BlendedPass, Board, Census, Changed, Changes, Clock, ColorPass, Cursor,
    DrawKind, Drawn, DrawnBounds, DrawnProjection, Error, ErrorCode, Events, Feed, FeedView,
    GeometryHandle, GpuStruct, InstanceValues, Interest, JobRef, KindCell, KindDecl, KindJoin,
    Layers, List, MaterialHandle, Mobility, Mover, MoverId, Node, NodeId, NodeWord, Pace, Pass,
    PassBit, PassEntry, PassMask, PassName, PassRole, PassViews, PositionReach, Progress,
    Readiness, Request, Requests, Ring, ScenePass, ScenePasses, Settling, Since, SlotSet, Stamp,
    StoreId, TargetName, TextureHandle, Ticket, Unsettled, Writes, gpu_consts,
};
pub use engine_place::{
    BitField, BitValue, FrameId, GpuBits, Readable, Readback, decode_bits, gpu_bits,
};
pub use engine_place::{Clear, Rect, Target, View, ViewCut, ViewHistory, ViewLayer};
pub use engine_registry::{FrameFacts, RecordParity};
#[cfg(any(feature = "dev", feature = "profiling"))]
pub use engine_registry::{
    Answer, Kind, Lands, NameEntry, Reach, ReachMut, register, unavailable,
};
pub use engine_settings::{
    Allowed, Expected, ExtraRules, ID_PATTERN, InvalidSetting, MAX_ID_BYTES, Rule, Setting, fields,
    is_id, parse_value, parse_value_at, rule, settings, state, stated, stats, valid_uri,
};
pub use engine_textures::{IMAGE_FORMATS, Image, ImageFormat, decode_image, image_format};
pub use files::Files;
pub use glam;
pub use hierarchy::*;
pub use input::*;
pub use jobs::*;
pub use mesh_lods::MeshLevels;
pub use moves::{Moves, Teleport, mover_of, send_ask, take_asks};
pub use presentation::*;
pub use provided::*;
pub use records::*;
pub use refusals::{Refusals, Refused};
pub use render_stats::DrawReports;
pub use render_stats::{DropReason, DroppedDraws, KindPass, KindPassState, Names, RenderStats};
pub use scene::*;
pub use schedules::{RunPace, RunPoint};
pub use section::{Cut, Section, Switch};
pub use services::{
    Authored, ComponentKind, ComponentKindOf, Detail, DeviceLimits, Levels, Placed,
    Placement, Prepared, SceneBuffers, SceneComponents, SceneContext, SceneCx, SceneForm,
    scene_component, scene_components,
};
pub use shared::FeedSettings;
pub use source::{Entry, Module, ShaderText, Source, Var};
pub use statistics::{Distribution, median};
pub use ticks::ScanClock;
pub(crate) use transform_edits::edit_transform;
pub use views::{ViewFacts, Views, main_camera};

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

use bevy_ecs::component::Immutable;
use bevy_ecs::hierarchy::ChildOf;
use bevy_ecs::lifecycle::Add;
use bevy_ecs::observer::On;
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::{InternedScheduleLabel, ScheduleLabel, Schedules, SingleThreadedExecutor};
use bevy_ecs::world::World;
use clock::{Advanced, FixedClock, Interval};
use engine_registry::{JobList, JobListed, MembersOf, Registry, RegistryBuilder, Switches};
use schedules::{FixedSimulation, FrameStart, Presentation, TransformPropagation};

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

#[cfg(feature = "dev")]
#[derive(Resource, Default)]
pub struct Held {
    pub paused: bool,
    camera: Option<(Transform, Camera)>,
}

#[cfg(feature = "dev")]
impl Held {
    pub fn camera(&self) -> Option<(Transform, Camera)> {
        self.camera
    }

    pub fn hold_camera(&mut self, camera: Option<(Transform, Camera)>) {
        self.camera = camera;
    }
}

#[cfg(feature = "dev")]
fn place_held(world: &mut World) {
    let held = world.get_resource::<Held>().and_then(Held::camera);
    if let Some((at, lens)) = held
        && let Some(camera) = main_camera(world)
    {
        place_camera(world, camera, at, lens);
    }
}

pub struct Engine {
    pub app: App,
    clock: FixedClock,
    interval: Interval,
    gates: Option<Switches>,
    facts: Option<FrameFacts>,
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
    registry.core_settings(bricks::settings_entry::<DrawnSettings>());
    registry.core_settings(bricks::settings_entry::<FrameLimits>());
    registry.despawn_rule(hierarchy::despawn_owned);
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

#[cfg(any(feature = "dev", feature = "profiling"))]
#[derive(Resource, Clone, Copy, Default)]
pub struct StageTimes {
    pub simulation: u64,
    pub extraction: u64,
    pub render: u64,
}

#[cfg(any(feature = "dev", feature = "profiling"))]
fn stages_time(world: &World) -> serde_json::Value {
    world.get_resource::<StageTimes>().map_or_else(
        || engine_registry::unavailable("stages_not_timed"),
        |times| {
            serde_json::json!({
                "simulation": times.simulation,
                "extraction": times.extraction,
                "render": times.render,
            })
        },
    )
}

#[cfg(any(feature = "dev", feature = "profiling"))]
fn core_names(world: &mut World) {
    world.init_resource::<StageTimes>();
    let named = |name: &'static str, kind: Kind, read: fn(&World) -> serde_json::Value| {
        NameEntry::read(name, kind, move |reach, _| Ok(read(reach.world)))
    };
    for entry in [
        named("stages.time", Kind::Time, stages_time).unit("ns"),
        named("jobs.asked", Kind::Count, jobs::jobs_asked),
        named("jobs.landed", Kind::Count, jobs::jobs_landed),
        named("jobs.failed", Kind::Count, jobs::jobs_failed),
        named("jobs.cancelled", Kind::Count, jobs::jobs_cancelled),
        named("jobs.time", Kind::Time, jobs::jobs_time).unit("ns"),
        named("jobs.exchange", Kind::Count, jobs::jobs_exchange),
        named("jobs.bytes", Kind::Bytes, jobs::jobs_bytes).unit("bytes"),
        named("jobs.state", Kind::State, jobs::jobs_state),
    ] {
        register(world, entry);
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
            .init_resource::<Pause>()
            .init_resource::<TimeScale>()
            .init_resource::<FrameLimits>()
            .init_resource::<Readiness>()
            .init_resource::<Refusals>()
            .init_resource::<Views>()
            .init_resource::<Detail>()
            .init_resource::<FeedSettings>()
            .init_resource::<DrawnSettings>()
            .insert_resource(exchange)
            .insert_resource(listed)
            .init_resource::<engine_registry::JobQueue>()
            .init_resource::<Records>()
            .init_resource::<Files>()
            .init_resource::<RecordExchange>()
            .insert_resource(DeviceLimits::without_device(cores))
            .init_resource::<MeshLevels>()
            .init_resource::<Devices>()
            .init_resource::<CursorMode>()
            .init_resource::<ProvidedAssets>()
            .init_resource::<PresentationState>()
            .init_resource::<DrawnPoses>()
            .init_resource::<Asked<Teleport>>()
            .init_resource::<Asked<ViewCut>>()
            .insert_resource(TransformNodeLimit(config.max_transform_nodes))
            .init_resource::<GlobalTransforms>()
            .insert_resource(assets);
        assets::install(app.world_mut());
        drawn::install(app.world_mut());
        #[cfg(feature = "dev")]
        app.init_resource::<Held>();
        Self {
            app,
            clock,
            interval: Interval::Fresh,
            gates: None,
            facts: None,
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
        core_entries(&mut registry);
        for listed in list.listed() {
            registry.add(listed.record());
        }
        let world = self.app.world_mut();
        world.insert_resource(JobListed::of(&jobs));
        world.resource_mut::<JobExchange>().set_list(jobs);
        world.register_component::<Camera>();
        world.register_component::<Drawn>();
        world.register_component::<View>();
        let labels: [InternedScheduleLabel; 3] = Schedule::ALL.map(Schedule::label);
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
        #[cfg(any(feature = "dev", feature = "profiling"))]
        core_names(world);
        #[cfg(any(feature = "dev", feature = "profiling"))]
        drawn::object_names(world);
        let mut registry = world.resource_mut::<Registry>();
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

    #[cfg(any(feature = "dev", feature = "profiling"))]
    pub fn name(&mut self, entry: NameEntry) -> &mut Self {
        register(self.app.world_mut(), entry);
        self
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

    pub fn notices(&self) -> &[Error] {
        self.app
            .world()
            .get_resource::<Registry>()
            .map_or(&[], Registry::notices)
    }

    pub fn set_compiling(&mut self, compiling: u32) {
        let world = self.app.world_mut();
        if world.resource::<Readiness>().compiling != compiling {
            world.resource_mut::<Readiness>().compiling = compiling;
        }
    }

    pub fn report_drawing(&mut self, reports: DrawReports) {
        let world = self.app.world_mut();
        if world.get_resource::<DrawReports>() == Some(&reports) {
            return;
        }
        world.insert_resource(reports);
        shared::settle_readiness(world, &self.board);
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
            engine.run_gates();
            if moves::start_frame(engine.app.world_mut()) {
                engine.refresh_transforms();
            }
            #[cfg(feature = "dev")]
            place_held(engine.app.world_mut());
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

    fn run_frame(
        &mut self,
        presented: Presented,
        advanced: Advanced,
        (real_seconds, seconds): (f64, f64),
        cut_seconds: f64,
    ) {
        let _profile = engine_profile::span("cpu", "simulation_frame");
        self.run_gates();
        hierarchy::settle_changes(self.app.world_mut());
        moves::start_frame(self.app.world_mut());
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
        for _ in 0..advanced.ticks {
            self.run_fixed_tick();
        }
        self.exchange_now();
        presentation::present(self.app.world_mut(), presented);
        {
            let _profile = engine_profile::span("ecs", "Presentation");
            self.app.world_mut().run_schedule(Presentation);
        }
        #[cfg(feature = "dev")]
        place_held(self.app.world_mut());
        self.app.world_mut().clear_trackers();
    }

    fn run_fixed_tick(&mut self) {
        let _profile = engine_profile::span("cpu", "fixed_tick");
        let step = self.clock.step_seconds();
        modify(self.app.world_mut(), |time: &mut FixedTime| {
            time.elapsed_seconds += step;
            time.tick = time.tick.saturating_add(1);
        });
        {
            let _profile = engine_profile::span("ecs", "FixedSimulation");
            self.app.world_mut().run_schedule(FixedSimulation);
        }
        moves::end_tick(self.app.world_mut());
        self.refresh_transforms();
        presentation::capture(self.app.world_mut());
    }

    pub fn reset_clock(&mut self) {
        self.clock.reset();
        self.interval = self.interval.next(None).0;
        let world = self.app.world_mut();
        presentation::fresh_all(world);
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
            mesh_lods::settle(engine.app.world_mut());
            engine.refresh_transforms();
            presentation::gather(engine.app.world_mut());
            engine.sync_view();
            shared::settle_readiness(engine.app.world_mut(), &engine.board);
        });
        Ok(RenderFrame {
            world: self.app.world(),
        })
    }

    #[doc(hidden)]
    pub fn exchange(&mut self) {
        self.scoped(Self::exchange_now);
    }

    fn exchange_now(&mut self) {
        let world = self.app.world_mut();
        records::exchange(world);
        files::read_files(world);
        jobs::exchange_orders(world);
    }

    fn sync_view(&mut self) {
        let _profile = engine_profile::span("ecs", "ViewSync");
        let facts = self.facts.unwrap_or_default();
        let world = self.app.world_mut();
        views::settle(world, facts);
        drawn::ask(world);
        assets::settle_assets(world);
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
