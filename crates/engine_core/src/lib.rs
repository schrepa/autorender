//! Platform-independent gameplay, fixed scheduling, and owned render extraction.
//!
//! The SDK explicitly uses Bevy ECS and Bevy's [`Plugin`] contract. Register
//! systems in [`FixedSimulation`] or [`Presentation`], then drive [`Engine`]
//! from the platform event loop. Bevy's default `Main` schedules are not run.

mod animation;
mod atmosphere;
mod clock;
mod culling;
mod decals;
mod effects;
mod entity_indices;
mod environment;
mod hierarchy;
mod input;
mod interpolation;
mod level_of_detail;
mod lights;
pub mod math;
mod packed;
mod particles;
mod probes;
mod retained_render;
mod scene;
mod skin;
mod transform_edits;
mod virtual_geometry;

pub use animation::{AnimationPlayer, AnimationSystems};
pub use atmosphere::Atmosphere;
pub use bevy_app::{self, App, Plugin};
pub use bevy_ecs;
pub use clock::{EngineConfig, FixedTime, FrameReport, FrameTime};
pub use decals::{Decal, DecalBlend, Decals, RenderDecal};
pub use effects::{
    AmbientOcclusion, Bloom, MAX_SHADOW_CASCADES, Reflections, Refraction, Shadows, SunShafts,
};
pub use engine_assets::{
    Aabb, AlphaMode, AssetBudget, AssetError, AssetLibrary, Filter, LodLevel,
    MAX_TEXTURE_DIMENSION, Material, MaterialBinding, MaterialHandle, Mesh, MeshHandle, MeshLods,
    MeshVertex, Sampler, Surface, Texture, TextureFormat, TextureHandle, TextureRef, Transmission,
    UvTransform, Wrap, mesh_lods,
};
pub use engine_assets::{
    AnimatedProperty, AnimationChannel, AnimationClip, AnimationSample, ClipHandle,
    KeyInterpolation, Keyframes,
};
pub use engine_assets::{
    CLUSTER_TRIANGLES, CLUSTER_VERTICES, CLUSTER_WORDS, DEFAULT_FALLBACK_TRIANGLES, GROUP_WORDS,
    NODE_WORDS, PAGE_WORDS, VIRTUAL_MESH_MAGIC, VirtualMesh,
};
pub use environment::{Environment, InvalidSetting};
pub use glam;
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
pub use level_of_detail::LevelOfDetail;
pub use lights::{LocalLight, LocalLights, RenderLight, Spot};
pub use particles::{
    MAX_EMITTER_PARTICLES, ParticleBlend, ParticleEmitter, Particles, RenderEmitter,
};
pub use probes::{ReflectionProbe, ReflectionProbes, RenderReflectionProbe};
pub use retained_render::{PoseRows, RenderFrame};
pub use scene::{
    Camera, Cube, InputState, Mesh3d, RenderInstance, RenderMaterial, RenderMesh, RenderSnapshot,
    RenderTexture, Tint, Transform,
};
pub use skin::{RenderJoint, Skin};
pub use transform_edits::{TransformEditError, edit_transform};
pub use virtual_geometry::VirtualGeometry;

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
    retained: retained_render::RetainedRender,
    // Local lights (decision 023) and particle emitters (decision 026) of
    // each extraction.
    lights: lights::LightExtraction,
    particles: particles::ParticleExtraction,
    skins: skin::SkinExtraction,
    decals: decals::DecalExtraction,
    probes: probes::ProbeExtraction,
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
        // The engine's cube is admitted before the configured budget applies.
        let mut assets = AssetLibrary::default();
        let cube_mesh = assets
            .insert_mesh(Mesh::cube())
            .expect("an unlimited library holds a cube");
        assets.set_budget(config.asset_budget);
        let mut app = App::empty();
        app.init_schedule(FixedSimulation)
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
            .init_resource::<Shadows>()
            .init_resource::<AmbientOcclusion>()
            .init_resource::<Bloom>()
            .init_resource::<LocalLights>()
            .init_resource::<Particles>()
            .init_resource::<Decals>()
            .init_resource::<ReflectionProbes>()
            .init_resource::<VirtualGeometry>()
            .init_resource::<LevelOfDetail>()
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
        let lights = lights::LightExtraction::new(app.world_mut());
        let particles = particles::ParticleExtraction::new(app.world_mut());
        let skins = skin::SkinExtraction::new(app.world_mut());
        let decals = decals::DecalExtraction::new(app.world_mut());
        let probes = probes::ProbeExtraction::new(app.world_mut());
        Self {
            app,
            clock,
            max_render_instances: config.max_render_instances,
            frustum_culling: config.frustum_culling,
            render_query,
            cube_mesh,
            retained: retained_render::RetainedRender::default(),
            lights,
            particles,
            skins,
            decals,
            probes,
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
        self.retained.extract_lights_and_particles(
            self.app.world(),
            &mut self.lights,
            &mut self.particles,
            false,
        );
        self.retained
            .extract_skins(self.app.world(), &mut self.skins, false);
        self.retained
            .extract_decals(self.app.world(), &mut self.decals, false);
        self.retained
            .extract_probes(self.app.world(), &mut self.probes);
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
        self.retained.extract_lights_and_particles(
            self.app.world(),
            &mut self.lights,
            &mut self.particles,
            true,
        );
        self.retained
            .extract_skins(self.app.world(), &mut self.skins, true);
        self.retained
            .extract_decals(self.app.world(), &mut self.decals, true);
        self.retained
            .extract_probes(self.app.world(), &mut self.probes);
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
        Aabb, AlphaMode, AmbientOcclusion, App, AssetBudget, AssetError, AssetLibrary, Atmosphere,
        Bloom, Camera, Cube, Engine, EngineConfig, Filter, FixedSimulation, FixedTime, FrameReport,
        FrameTime, GlobalTransform, GlobalTransforms, HierarchyError, HierarchyReport, InputState,
        InterpolateTransform, InterpolationError, MAX_HIERARCHY_DEPTH, MAX_TRANSFORM_NODES,
        Material, MaterialBinding, MaterialHandle, Mesh, Mesh3d, MeshHandle, MeshVertex, Parent,
        Plugin, Presentation, PresentationMode, PresentationState, PresentedTransform,
        PresentedTransforms, RenderFrame, RenderInstance, RenderMaterial, RenderMesh,
        RenderSnapshot, RenderTexture, Sampler, Shadows, Skin, Surface, Texture, TextureHandle,
        TextureRef, Tint, Transform, TransformIndex, TransformNodeLimit, TransformPropagation,
        UvTransform, Wrap, propagate_transforms, reset_all_interpolation, reset_interpolation,
        set_parent, set_parents, valid_global_matrix,
    };
    pub use crate::{ActionId, ActionInput};
    pub use crate::{
        AnimatedProperty, AnimationChannel, AnimationClip, AnimationPlayer, AnimationSystems,
        ClipHandle, KeyInterpolation, Keyframes,
    };
    pub use crate::{Decal, DecalBlend, Decals};
    pub use crate::{FixedPostSimulation, FixedPreSimulation};
    pub use crate::{
        LevelOfDetail, ReflectionProbe, ReflectionProbes, RenderReflectionProbe, VirtualGeometry,
    };
    pub use crate::{
        LocalLight, LocalLights, MAX_SHADOW_CASCADES, ParticleBlend, ParticleEmitter, Particles,
        Reflections, Refraction, RenderEmitter, RenderLight, Spot, SunShafts, Transmission,
    };
    pub use crate::{TransformEditError, edit_transform};
    // Bevy derives resolve this name in client scope when only the engine
    // facade is a direct dependency, so expose the crate alongside its prelude.
    pub use bevy_ecs::{self, prelude::*};
    pub use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
}

#[cfg(test)]
mod tests;
