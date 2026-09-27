//! Platform-independent gameplay, fixed scheduling, and owned render extraction.
//!
//! The SDK explicitly uses Bevy ECS and Bevy's [`Plugin`] contract. Register
//! systems in [`FixedSimulation`] or [`Presentation`], then drive [`Engine`]
//! from the platform event loop. Bevy's default `Main` schedules are not run.

mod atmosphere;
mod clock;
mod culling;
mod hierarchy;
mod input;
mod interpolation;
pub mod math;
mod packed;
mod retained_render;
mod scene;
mod transform_edits;
mod visibility_cache;

pub use atmosphere::Atmosphere;
pub use bevy_app::{self, App, Plugin};
pub use bevy_ecs;
pub use clock::{EngineConfig, FixedTime, FrameReport, FrameTime};
pub use engine_assets::{
    Aabb, AssetBudget, AssetError, AssetLibrary, Material, MaterialHandle, Mesh, MeshHandle,
    MeshVertex, Texture, TextureHandle,
};
pub use glam;
pub use hierarchy::{
    GlobalTransform, GlobalTransforms, HierarchyError, HierarchyReport, MAX_HIERARCHY_DEPTH,
    MAX_NORMAL_INVERSE_NORM, MAX_TRANSFORM_NODES, Parent, TransformIndex, propagate_transforms,
    set_parent, set_parents, valid_global_matrix,
};
pub use input::{ActionId, ActionInput};
pub use interpolation::{
    GpuCurrent, GpuPresentation, InterpolateTransform, InterpolationError, NO_PARENT, NODE_ACTIVE,
    NODE_BLEND, NODE_PARENT_ACTIVE, NODE_VALID, NodeInfo, NodePose, PresentationMode,
    PresentationState, PresentedTransform, PresentedTransforms, reset_all_interpolation,
    reset_interpolation,
};
pub use retained_render::{PoseRows, RenderFrame};
pub use scene::{
    Camera, Cube, InputState, Mesh3d, RenderInstance, RenderMesh, RenderSnapshot, RenderTexture,
    Transform,
};
pub use transform_edits::{
    MAX_TRANSFORM_EDITS, TransformEditError, TransformEdits, edit_transform,
};

use bevy_app::PluginsState;
use bevy_ecs::{
    prelude::*,
    schedule::{ScheduleLabel, SingleThreadedExecutor},
};
use clock::FixedClock;
use interpolation::InterpolationState;
use std::{collections::HashSet, sync::Arc};
use visibility_cache::VisibilityCache;

type RenderQuery = QueryState<
    (
        Entity,
        Has<Transform>,
        Option<Ref<'static, Cube>>,
        Option<Ref<'static, Mesh3d>>,
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

/// An invalid development step request. Valid requests contain 1–120 ticks.
#[cfg(feature = "dev")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevStepError {
    /// Rejected tick count. Validation occurs before any engine mutation.
    pub ticks: u32,
}

#[cfg(feature = "dev")]
impl DevStepError {
    /// Maximum fixed ticks in one synchronous development request.
    pub const MAX_TICKS: u32 = 120;

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
    max_render_instances: usize,
    frustum_culling: bool,
    render_query: RenderQuery,
    cube_mesh: MeshHandle,
    extracted_meshes: HashSet<MeshHandle>,
    extracted_textures: HashSet<TextureHandle>,
    visibility: VisibilityCache,
    retained: retained_render::RetainedRender,
    plugins_finalized: bool,
    // A fixed tick whose first stage ran ahead of its due frame (see
    // `Self::run_ahead`), and completed ticks the clock has not yet counted.
    tick_in_flight: bool,
    ticks_ahead: u32,
    // A tick `Self::run_ahead` completed before its due frame: GPU frames keep
    // presenting the previous interval until the clock counts it.
    tick_ahead_hidden: bool,
    // GPU inputs of that tick were gathered when it completed.
    gathered_ahead: bool,
}

impl Engine {
    /// Creates gameplay and transform schedules and default camera/input resources.
    pub fn new(config: EngineConfig) -> Self {
        let clock = FixedClock::new(config);
        let mut assets = AssetLibrary::default();
        let cube_mesh = assets
            .insert_mesh(Mesh::cube())
            .expect("default asset budget holds a cube");
        let mut app = App::empty();
        app.init_schedule(FixedSimulation)
            .init_schedule(FixedPreSimulation)
            .init_schedule(FixedPostSimulation)
            .init_schedule(Presentation)
            .init_schedule(TransformPropagation)
            .add_systems(TransformPropagation, |world: &mut World| {
                propagate_transforms(world);
            })
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
            .insert_resource(FixedTime {
                delta_seconds: clock.step_seconds() as f32,
                elapsed_seconds: 0.0,
                tick: 0,
            })
            .init_resource::<FrameTime>()
            .init_resource::<Camera>()
            .init_resource::<Atmosphere>()
            .init_resource::<InputState>()
            .init_resource::<ActionInput>()
            .init_resource::<PresentationState>()
            .init_resource::<InterpolationState>()
            .init_resource::<GlobalTransforms>()
            .insert_resource(assets);
        let render_query = app.world_mut().query_filtered::<(
            Entity,
            Has<Transform>,
            Option<Ref<Cube>>,
            Option<Ref<Mesh3d>>,
        ), Or<(With<Cube>, With<Mesh3d>)>>();
        Self {
            app,
            clock,
            max_render_instances: config.max_render_instances,
            frustum_culling: config.frustum_culling,
            render_query,
            cube_mesh,
            extracted_meshes: HashSet::new(),
            extracted_textures: HashSet::new(),
            visibility: VisibilityCache::default(),
            retained: retained_render::RetainedRender::default(),
            plugins_finalized: false,
            tick_in_flight: false,
            ticks_ahead: 0,
            tick_ahead_hidden: false,
            gathered_ahead: false,
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
        self.complete_pending_tick();
        self.ticks_ahead = 0;
        self.tick_ahead_hidden = false;
        self.gathered_ahead = false;
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
        // Ticks completed ahead of the clock count against the due ticks.
        let ahead = fixed_steps.min(self.ticks_ahead);
        self.ticks_ahead -= ahead;
        if self.ticks_ahead == 0 {
            self.tick_ahead_hidden = false;
            self.gathered_ahead = false;
        }
        let mut fixed_steps = fixed_steps - ahead;
        // The first due tick is the one whose first stage already ran.
        if self.tick_in_flight && fixed_steps > 0 {
            self.finish_pending_tick();
            fixed_steps -= 1;
        }
        // Boundaries while a tick is in flight would do its remaining
        // propagation early; its second stage refreshes when it is due.
        if !self.tick_in_flight {
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
        if !self.tick_in_flight {
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
        if self.ticks_ahead != 0 {
            return false;
        }
        let predicted = clock::sanitize_delta(predicted_delta_seconds).min(0.25);
        let accumulator = self.clock.accumulator_seconds();
        let step = self.clock.step_seconds();
        // Within a quarter frame of due counts as due at that frame.
        if self.tick_in_flight {
            if accumulator + predicted * 1.25 < step {
                return false;
            }
            self.complete_pending_tick();
            // With GPU presentation, gather the completed tick's inputs in
            // this lighter frame: the due frame finds them current and only
            // presents.
            let _profile = engine_profile::span("cpu", "gather_ahead");
            self.gathered_ahead = interpolation::gather_ahead(self.app.world_mut());
            return true;
        }
        if accumulator + predicted * 2.25 < step {
            return false;
        }
        let _profile = engine_profile::span("cpu", "fixed_tick_begin");
        self.begin_fixed_tick();
        self.tick_in_flight = true;
        true
    }

    /// Whether a tick started by [`Self::run_ahead`] awaits its second stage.
    pub fn tick_in_flight(&self) -> bool {
        self.tick_in_flight
    }

    /// Completes a tick started by [`Self::run_ahead`] now, before it is due;
    /// the clock counts it when it becomes due. Until then ordinary GPU frames
    /// keep presenting the previous interval; [`Self::reveal_ticks_ahead`] lets
    /// them present the completed tick instead. Exact extraction, stepping and
    /// suspension call this themselves.
    pub fn complete_pending_tick(&mut self) {
        if self.tick_in_flight {
            self.finish_pending_tick();
            self.ticks_ahead += 1;
            self.tick_ahead_hidden = true;
        }
    }

    /// Lets GPU frames present ticks completed ahead of the clock, e.g. for an
    /// explicit preview of the latest completed interval. Ordinary frames then
    /// continue from that interval.
    pub fn reveal_ticks_ahead(&mut self) {
        self.tick_ahead_hidden = false;
        self.gathered_ahead = false;
    }

    /// Whether a completed tick awaits the frame the clock makes it due in.
    pub fn tick_completed_ahead(&self) -> bool {
        self.ticks_ahead != 0
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
        self.tick_in_flight = false;
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
        interpolation::end_tick(self.app.world_mut());
        self.app
            .world_mut()
            .resource_mut::<ActionInput>()
            .end_tick();
    }

    /// Discards the fractional accumulator after suspension without rewinding
    /// simulation ticks or elapsed time. The host should also reset its clock.
    pub fn reset_clock(&mut self) {
        self.complete_pending_tick();
        self.ticks_ahead = 0;
        self.tick_ahead_hidden = false;
        self.gathered_ahead = false;
        self.clock.reset();
        reset_all_interpolation(self.app.world_mut());
        self.app.world_mut().resource_mut::<ActionInput>().clear();
        self.app
            .world_mut()
            .resource_mut::<FrameTime>()
            .delta_seconds = 0.0;
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
        self.frustum_culling = enabled;
    }

    pub fn frustum_culling(&self) -> bool {
        self.frustum_culling
    }

    /// Metadata for the last prepared extraction, independent of GPU completion.
    pub fn presentation(&self) -> &PresentationState {
        self.app.world().resource::<PresentationState>()
    }

    /// Copies renderable instances into a reusable, owned CPU snapshot.
    ///
    /// Extraction retains vector capacity and copies at most the configured
    /// instance budget, in ECS iteration order. `dropped_instances` reports the
    /// visible remainder. Invalid transforms skip before asset checks. Missing
    /// assets skip the entity and increment
    /// `missing_assets`; `Mesh3d` takes precedence over `Cube`. Each referenced
    /// mesh and texture used by an emitted or culled instance is retained once
    /// via `Arc`, without copying payloads. Culled sources preserve existing GPU
    /// residency but do not request uploads on their own. Sources used only by
    /// invalid, missing-asset or dropped entities are not retained. Removing
    /// library assets later does not invalidate this snapshot. GPU record
    /// conversion and uploads belong to the renderer.
    pub fn extract(&mut self, aspect: f32, snapshot: &mut RenderSnapshot) {
        self.extract_with_presentation(aspect, snapshot, PresentationMode::Current)
            .expect("current presentation is always valid");
    }

    /// Extracts current or interpolated display matrices without changing gameplay.
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
        let _profile = engine_profile::span("cpu", "extraction");
        mode.validate()?;
        self.complete_pending_tick();
        self.refresh_transforms();
        interpolation::prepare(self.app.world_mut(), mode);
        let world = self.app.world();
        snapshot.view_projection = world.resource::<Camera>().view_projection(aspect);
        snapshot.camera_position = world.resource::<Camera>().eye();
        snapshot.atmosphere = atmosphere::current(world);
        snapshot.instances.clear();
        snapshot.meshes.clear();
        snapshot.textures.clear();
        snapshot.dropped_instances = 0;
        snapshot.missing_assets = 0;
        snapshot.invalid_transforms = 0;
        snapshot.visible_instances = 0;
        snapshot.culled_instances = 0;
        self.extracted_meshes.clear();
        self.extracted_textures.clear();
        let assets = world.resource::<AssetLibrary>();
        let instances = self.render_query.iter(world);
        let total = instances.len();
        snapshot.render_candidates = total;
        self.visibility
            .begin(snapshot.view_projection, self.frustum_culling, total);
        let count = total.min(self.max_render_instances);
        if snapshot.instances.capacity() < count {
            snapshot.instances.reserve_exact(count);
        }
        let extraction_loop = engine_profile::span("cpu", "extraction_scan_assets_culling_copy");
        let globals = world.resource::<GlobalTransforms>();
        let presented = world.resource::<PresentedTransforms>();
        for (entity, has_transform, cube, mesh3d) in instances {
            let model = if has_transform
                && globals
                    .get(entity)
                    .is_some_and(|global| global.matrix().is_some())
            {
                presented.get(entity).and_then(|pose| pose.matrix())
            } else {
                None
            };
            let Some(model) = model else {
                snapshot.invalid_transforms += 1;
                continue;
            };
            let (handle, color, texture_handle) = if let Some(mesh3d) = mesh3d {
                let Some(material) = assets.material(mesh3d.material) else {
                    snapshot.missing_assets += 1;
                    continue;
                };
                (mesh3d.mesh, material.color(), material.texture())
            } else if let Some(cube) = cube {
                (self.cube_mesh, cube.color, None)
            } else {
                continue;
            };
            let Some(mesh) = assets.mesh(handle) else {
                snapshot.missing_assets += 1;
                continue;
            };
            let texture = if let Some(handle) = texture_handle {
                let Some(texture) = assets.texture(handle) else {
                    snapshot.missing_assets += 1;
                    continue;
                };
                Some((handle, texture))
            } else {
                None
            };
            let culled = self.visibility.outside(handle, model, mesh.bounds());
            if culled {
                snapshot.culled_instances += 1;
            } else {
                snapshot.visible_instances += 1;
                if snapshot.instances.len() == self.max_render_instances {
                    snapshot.dropped_instances += 1;
                    continue;
                }
            }
            if self.extracted_meshes.insert(handle) {
                snapshot.meshes.push(RenderMesh {
                    handle,
                    mesh: Arc::clone(mesh),
                });
            }
            if let Some((handle, texture)) = texture
                && self.extracted_textures.insert(handle)
            {
                snapshot.textures.push(RenderTexture {
                    handle,
                    texture: Arc::clone(texture),
                });
            }
            if !culled {
                snapshot.instances.push(RenderInstance {
                    mesh: handle,
                    texture: texture_handle,
                    model,
                    color,
                });
            }
        }
        drop(extraction_loop);
        self.visibility.finish();
        engine_profile::counter("extraction.candidates", total as f64);
        engine_profile::counter("extraction.instances", snapshot.instances.len() as f64);
        engine_profile::counter("extraction.culled", snapshot.culled_instances as f64);
        engine_profile::counter(
            "extraction.instance_capacity_bytes",
            (snapshot.instances.capacity() * size_of::<RenderInstance>()) as f64,
        );
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
        self.refresh_transforms();
        interpolation::prepare(self.app.world_mut(), mode);
        self.retained.prepare(
            self.app.world(),
            &mut self.render_query,
            self.cube_mesh,
            aspect,
            self.frustum_culling,
            self.max_render_instances,
            false,
        );
        Ok(self.retained.frame(
            self.app.world().resource::<PresentedTransforms>(),
            self.app.world().resource::<GlobalTransforms>(),
        ))
    }

    /// Prepares a retained frame whose interpolated display poses are blended,
    /// composed, validated and culled on the GPU (decision 014).
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
        if self.tick_in_flight || self.tick_ahead_hidden {
            // Mid-tick state, and a tick completed before it is due, are not
            // gathered: present the previous interval's inputs at this fraction.
            // Inputs already gathered ahead for the completed tick are held:
            // consumers keep presenting the inputs they applied before.
            interpolation::present_gpu_fraction(self.app.world_mut(), alpha, self.gathered_ahead);
        } else {
            self.refresh_transforms();
            interpolation::prepare_gpu(self.app.world_mut(), alpha);
        }
        self.retained.ensure_source_scan(self.app.world_mut());
        self.retained.prepare(
            self.app.world(),
            &mut self.render_query,
            self.cube_mesh,
            aspect,
            self.frustum_culling,
            self.max_render_instances,
            true,
        );
        // Later source edits must be newer than the stamp window just consumed.
        self.app.world_mut().increment_change_tick();
        Ok(self.retained.frame(
            self.app.world().resource::<PresentedTransforms>(),
            self.app.world().resource::<GlobalTransforms>(),
        ))
    }

    /// CPU-prepares [`PresentedTransforms`] for the last GPU-presented frame,
    /// exactly as [`Self::extract_frame`] would have prepared it. Returns false
    /// and changes nothing when the last extraction was already CPU-prepared,
    /// or once a tick, refresh or reset changed that frame's inputs; then
    /// [`PresentationState::gpu_pending`] stays true. O(nodes).
    pub fn prepare_presented_transforms(&mut self) -> bool {
        if self.tick_in_flight || self.tick_ahead_hidden {
            // The frame's inputs predate the in-flight or early tick's changes.
            return false;
        }
        interpolation::prepare_pending(self.app.world_mut())
    }

    fn finalize_plugins(&mut self) {
        if self.plugins_finalized {
            return;
        }
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

impl Default for Engine {
    fn default() -> Self {
        Self::new(EngineConfig::default())
    }
}

/// Common gameplay imports; Bevy types are intentionally part of the SDK.
pub mod prelude {
    #[cfg(feature = "dev")]
    pub use crate::DevStepError;
    pub use crate::math;
    pub use crate::{
        Aabb, App, AssetBudget, AssetError, AssetLibrary, Atmosphere, Camera, Cube, Engine,
        EngineConfig, FixedSimulation, FixedTime, FrameReport, FrameTime, GlobalTransform,
        GlobalTransforms, HierarchyError, HierarchyReport, InputState, InterpolateTransform,
        InterpolationError, MAX_HIERARCHY_DEPTH, MAX_TRANSFORM_NODES, Material, MaterialHandle,
        Mesh, Mesh3d, MeshHandle, MeshVertex, Parent, Plugin, Presentation, PresentationMode,
        PresentationState, PresentedTransform, PresentedTransforms, RenderFrame, RenderInstance,
        RenderMesh, RenderSnapshot, RenderTexture, Texture, TextureHandle, Transform,
        TransformIndex, TransformPropagation, propagate_transforms, reset_all_interpolation,
        reset_interpolation, set_parent, set_parents, valid_global_matrix,
    };
    pub use crate::{ActionId, ActionInput};
    pub use crate::{FixedPostSimulation, FixedPreSimulation};
    pub use crate::{MAX_TRANSFORM_EDITS, TransformEditError, TransformEdits, edit_transform};
    // Bevy derives resolve this name in client scope when only the engine
    // facade is a direct dependency, so expose the crate alongside its prelude.
    pub use bevy_ecs::{self, prelude::*};
    pub use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
}

#[cfg(test)]
mod tests;
