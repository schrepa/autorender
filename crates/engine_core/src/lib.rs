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
mod closing;
mod dev_doors;
mod drawn;
mod entity_indices;
mod files;
mod hierarchy;
mod host;
mod input;
mod kept;
pub mod math;
mod moves;
pub use engine_math::collections;
mod refusals;
mod render_stats;
mod scene;
mod schedules;
mod section;
mod services;
mod shared;
mod sound;
mod source;
mod ticks;
mod transform_edits;
mod views;

pub use assets::AssetKinds;
pub use authoring::core_components;
pub use background::{Background, Delivers, Landed, Taken};
pub use bevy_app::App;
pub use bricks::*;
pub use clock::*;
pub use closing::Closing;
pub use drawn::{DrawnBatch, DrawnRecord, DrawnSettings, DrawnTable, drawn_wgsl};
pub use engine_assets::{
    Aabb, AlphaMode, Asset, AssetId, AssetKind, Assets, ColorSpace, Compare, Cull, DepthBias,
    Filter, GroupReport, GroupStage, IndexFormat, LightingModel, Material, MeshDescription,
    MeshFile, RenderState, Sampler, Texture, TextureFormat, TextureRef, UvTransform, VertexFormat,
    VertexInput, VertexStream, Wrap,
};
pub use engine_place::{
    AuthoredId, OperationScene, SCENE_SCHEMA, SceneEntity, entity_removed, schema_of,
};
pub use engine_place::{
    Asked, Board, Census, Changed, Changes, Clock, DrawKind, Drawn, Error, ErrorCode, GpuStruct,
    InstanceValues, Interest, KindCell, KindDecl, KindJoin, Layers, List, Mobility, Mover, MoverId,
    Node, NodeId, NodeWord, Pace, Pass, PassBit, PassEntry, PassMask, PassName, PassViews,
    Progress, Readiness, Request, Requests, ScenePass, ScenePasses, Settling, TargetName, Ticket,
    Unsettled, Writes,
};
pub use engine_place::{BitValue, FrameId, GpuBits, Readable, Readback, gpu_bits};
pub use engine_place::{Clear, Rect, Target, View, ViewSkip};
pub use engine_codec::content_hash;
pub use engine_registry::{FrameFacts, ViewportClass};
#[cfg(feature = "dev")]
pub use engine_registry::{
    Answer, Kind, Lands, NameEntry, Reach, ReachMut, register, unavailable,
};
pub use engine_settings::{
    Allowed, Expected, ExtraRules, ID_PATTERN, InvalidSetting, MAX_ID_BYTES, Rule, Setting, fields,
    is_id, parse_value, parse_value_at, rule, settings, state, stated, stats, valid_uri,
};
pub use dev_doors::{DevBricks, DevOnly, DevSetting, Stats};
pub use files::{Files, IoAct, IoOrder, IoReply};
pub use kept::Kept;
pub use glam;
pub use hierarchy::*;
pub use host::*;
pub use input::*;
pub use moves::{Moves, Teleport, send_ask};
pub use refusals::{Refusals, Refused};
pub use render_stats::DrawReports;
pub use render_stats::{DropReason, DroppedDraws, KindPass, KindPassState, Names, RenderStats};
pub use scene::*;
pub use schedules::Schedule;
pub use section::{Section, Switch};
pub use services::{
    ComponentKind, ComponentKindOf, DeviceLimits, Prepared, SceneComponents, SceneContext, SceneCx,
    SceneForm, scene_component, scene_components,
};
pub use sound::{
    Length, Loaded, Output, Play, Refusal, RefusalCounts, SoundCounts, SoundId, SoundOrder,
    SoundReply, SoundReport, Sounds, Tune, VoiceCounts, VoiceId,
};
pub use source::{Entry, Module, ShaderText, Source, Var};
pub(crate) use transform_edits::edit_transform;
pub use views::{ViewFacts, Views, main_camera};

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

use bevy_ecs::component::Immutable;
use bevy_ecs::hierarchy::ChildOf;
use bevy_ecs::lifecycle::Add;
use bevy_ecs::observer::On;
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::{InternedScheduleLabel, ScheduleLabel, Schedules, SingleThreadedExecutor};
use bevy_ecs::world::World;
use clock::{Advanced, FixedClock, Interval};
use engine_registry::{MembersOf, Registry, RegistryBuilder, Switches};
use schedules::{BeforeDraw, BeforeTicks, FixedSimulation, Presentation, TransformPropagation};

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
    ticked: Devices,
    gates: Option<Switches>,
    facts: Option<FrameFacts>,
    finalized: bool,
    board: Board,
    reported: Option<shared::Reported>,
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
    registry.core_settings(bricks::settings_entry::<DrawnSettings>());
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

#[cfg(feature = "dev")]
#[derive(Resource, Clone, Copy, Default)]
pub struct StageTimes {
    pub simulation: u64,
    pub extraction: u64,
    pub render: u64,
}

#[cfg(feature = "dev")]
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

#[cfg(feature = "dev")]
fn core_names(world: &mut World) {
    world.init_resource::<StageTimes>();
    register(
        world,
        NameEntry::read("stages.time", Kind::Time, |reach, _| Ok(stages_time(reach.world)))
            .unit("ns"),
    );
}

fn refused(problems: &[Error]) -> Result<(), Error> {
    match problems {
        [] => Ok(()),
        [only] => Err(only.clone()),
        [first, ..] => Err(Error::new(first.code, "$.bricks").got(problems)),
    }
}

impl Engine {
    pub fn listed(
        list: BrickList,
        ticks_per_second: u32,
        catch_up_seconds: f64,
    ) -> Result<Self, Error> {
        let board = Board::default();
        let mut engine =
            board.scope(|| Self::on_board(ticks_per_second, catch_up_seconds, board.clone()))?;
        engine.scoped(|engine| engine.install(&list))?;
        Ok(engine)
    }

    fn on_board(
        ticks_per_second: u32,
        catch_up_seconds: f64,
        board: Board,
    ) -> Result<Self, Error> {
        let clock = FixedClock::new(ticks_per_second, catch_up_seconds)?;
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
                rate: ticks_per_second,
            })
            .init_resource::<FrameTime>()
            .init_resource::<Pause>()
            .init_resource::<Readiness>()
            .init_resource::<Refusals>()
            .init_resource::<Views>()
            .init_resource::<DrawnSettings>()
            .init_resource::<Files>()
            .init_resource::<Kept>()
            .init_resource::<Closing>()
            .init_resource::<DeviceLimits>()
            .init_resource::<Devices>()
            .init_resource::<Pointer>()
            .init_resource::<InputProfile>()
            .init_resource::<CursorMode>()
            .init_resource::<Sounds>()
            .init_resource::<engine_place::DrawnPoses>()
            .init_resource::<Asked<Teleport>>()
            .init_resource::<GlobalTransforms>()
            .init_resource::<Assets>();
        drawn::install(app.world_mut());
        #[cfg(feature = "dev")]
        app.init_resource::<Held>();
        Ok(Self {
            app,
            clock,
            interval: Interval::Fresh,
            ticked: Devices::default(),
            gates: None,
            facts: None,
            finalized: false,
            board,
            reported: None,
        })
    }

    fn scoped<R>(&mut self, run: impl FnOnce(&mut Self) -> R) -> R {
        let board = self.board.clone();
        board.scope(|| run(self))
    }

    fn install(&mut self, list: &BrickList) -> Result<(), Error> {
        let mut registry = RegistryBuilder::default();
        core_entries(&mut registry);
        for listed in list.listed() {
            let (record, systems) = listed.record();
            registry.add(record, systems);
        }
        let world = self.app.world_mut();
        world.register_component::<Camera>();
        world.register_component::<Drawn>();
        world.register_component::<View>();
        let labels: [InternedScheduleLabel; 4] = Schedule::ALL.map(Schedule::label);
        let mut gates = registry.install(world, &labels);
        moves::install_kinds(world);
        drawn::install_kinds(world);
        assets::install_kinds(world);
        install_passes(world);
        let mut twice = world
            .get_resource::<MembersOf<SceneComponents>>()
            .map(|members| members.names_twice(|kind| kind.key()))
            .unwrap_or_default();
        twice.extend(authoring::keys_taken(world));
        #[cfg(feature = "dev")]
        core_names(world);
        #[cfg(feature = "dev")]
        drawn::object_names(world);
        #[cfg(feature = "dev")]
        sound::names(world);
        let mut registry = world.resource_mut::<Registry>();
        for error in twice {
            registry.report(error);
        }
        let problems: Vec<Error> = registry
            .problems()
            .iter()
            .map(|(error, _)| error.clone())
            .collect();
        let checked = refused(&problems);
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

    #[doc(hidden)]
    pub fn set_frame(&mut self, facts: FrameFacts) {
        self.facts = Some(facts);
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
        shared::settle_readiness(world, &self.board, &mut self.reported);
    }

    pub fn report_held(&mut self, ready: bool) {
        let world = self.app.world_mut();
        if ready && world.resource::<Assets>().compiling() {
            world.resource_mut::<Assets>().compiled();
            world.resource_mut::<DrawnTable>().release_held();
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
        let real_seconds = if self.shown() {
            let (interval, real_seconds) = self.interval.next(timestamp_seconds);
            self.interval = interval;
            real_seconds
        } else {
            0.0
        };
        let rate = if self.time_runs() { 1.0 } else { 0.0 };
        let shown = self.clock.capped(real_seconds);
        let advanced = self.clock.advance(shown * rate);
        self.run_frame(advanced, (shown, shown * rate), real_seconds - shown);
    }

    pub fn frame_time(&self) -> FrameTime {
        *self.app.world().resource::<FrameTime>()
    }

    fn time_runs(&mut self) -> bool {
        let runs = self.shown() && !self.app.world().resource::<Pause>().paused();
        if !runs {
            self.drop_tick_marks();
        }
        runs
    }

    #[cfg(feature = "dev")]
    pub fn step(&mut self, ticks: u32, alpha: f32) -> Result<(), Error> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(Error::new(ErrorCode::InvalidPresentationAlpha, "$.alpha")
                .range(0.0, 1.0)
                .got(alpha));
        }
        self.scoped(|engine| {
            engine.finalize();
            engine.clock.reset();
            engine.interval = Interval::Fresh;
            let step = engine.clock.step_seconds();
            let lasting = if engine.shown() {
                f64::from(ticks) * step
            } else {
                0.0
            };
            let ran = if engine.time_runs() { ticks } else { 0 };
            let advanced = Advanced { ticks: ran, alpha };
            engine.run_frame(advanced, (lasting, f64::from(ran) * step), 0.0);
        });
        Ok(())
    }

    pub fn before_draw(&mut self) {
        self.scoped(|engine| {
            engine.finalize();
            let world = engine.app.world_mut();
            if has_systems(world, BeforeDraw) {
                world.run_schedule(BeforeDraw);
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
        let alpha = advanced.alpha;
        modify(world, |time: &mut FrameTime| {
            time.frame = time.frame.saturating_add(1);
            time.ticks = advanced.ticks;
            time.alpha = alpha;
            time.seconds = seconds;
            time.real_seconds = real_seconds;
            time.elapsed = ticked - (1.0 - f64::from(alpha)) * step;
            time.real_elapsed += real_seconds;
            time.cut_seconds += cut_seconds;
        });
        if has_systems(self.app.world(), BeforeTicks) {
            let _profile = engine_profile::span("ecs", "BeforeTicks");
            self.app.world_mut().run_schedule(BeforeTicks);
            self.refresh_transforms();
        }
        self.run_ticks(advanced.ticks);
        moves::present(self.app.world_mut(), alpha);
        {
            let _profile = engine_profile::span("ecs", "Presentation");
            self.app.world_mut().run_schedule(Presentation);
        }
        #[cfg(feature = "dev")]
        place_held(self.app.world_mut());
        self.app.world_mut().clear_trackers();
    }

    pub(crate) fn run_fixed_tick(&mut self) {
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
        moves::capture(self.app.world_mut());
    }

    pub fn reset_clock(&mut self) {
        self.clock.reset();
        self.interval = Interval::Fresh;
        let world = self.app.world_mut();
        moves::fresh_all(world);
        modify(world, |time: &mut FrameTime| {
            time.ticks = 0;
            time.seconds = 0.0;
            time.real_seconds = 0.0;
        });
    }

    fn refresh_transforms(&mut self) {
        self.app.world_mut().run_schedule(TransformPropagation);
    }

    pub fn extract_frame(&mut self) -> Result<RenderFrame<'_>, Error> {
        let _profile = engine_profile::span("cpu", "extraction");
        self.scoped(|engine| {
            engine.finalize();
            engine.refresh_transforms();
            engine.sync_view();
            shared::settle_readiness(engine.app.world_mut(), &engine.board, &mut engine.reported);
        });
        Ok(RenderFrame {
            world: self.app.world(),
        })
    }

    pub fn take_io(&mut self) -> Vec<IoOrder> {
        self.scoped(|engine| files::take_io(engine.app.world_mut()))
    }

    pub fn land_io(&mut self, reply: IoReply) {
        self.scoped(|engine| files::land_io(engine.app.world_mut(), reply));
    }

    fn sync_view(&mut self) {
        let _profile = engine_profile::span("ecs", "ViewSync");
        let facts = self.facts.unwrap_or_default();
        let world = self.app.world_mut();
        drawn::resolve(world);
        moves::gather(world);
        views::settle(world, facts);
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
