//! One frame of [`Renderer`], stage by stage: planning, the surface, asset
//! uploads, instance packing and writes, each feature's preparation, the
//! frame's pipelines, encoding and submission. `prepare` prepares the
//! frame, `encode` and `opaque` encode its render graph's passes, `submit`
//! submits it, and `frame_stats` reports it.

use super::*;
use crate::decals::DecalFrame;
use crate::gpu_presentation::{CullingView, GpuPrepared, PresentationInputs, RecordsKey, Staging};
use crate::particles::ParticleFrame;
use crate::post::PostFrame;
use crate::render_graph::{
    self, Clear, Effects, FrameShape, Load, Node, Ops, Pass, Phase, Resource,
};
use crate::shadows::{MapSources, ShadowFrame, ShadowInputs};

mod encode;
mod opaque;
mod prepare;
mod submit;

/// How far switches between how objects show fade each frame:
/// the presentation time of the last displayed GPU-presented
/// frame, and this frame's share of a switch's fade.
#[derive(Default)]
pub(super) struct SwitchFades {
    presented: f32,
    pub(super) share: f32,
}

impl SwitchFades {
    /// This frame's share of a fade `fade_seconds` long, at presentation time
    /// `seconds`: displayed GPU-presented frames advance by the time since
    /// the last (none for one presenting the same moment again, as the
    /// first after a pause does); previews switch at once (1), and exact
    /// frames show objects as they stand (a negative share).
    fn advance(&mut self, kind: FrameKind, seconds: f32, fade_seconds: f32) {
        self.share = match (kind.gpu_mode, kind.display) {
            (false, _) => -1.0,
            (true, false) => 1.0,
            (true, true) => {
                let since = seconds - std::mem::replace(&mut self.presented, seconds);
                if fade_seconds > 0.0 && since < 1.0 {
                    engine_core::math::clamp(since / fade_seconds, 0.0, 1.0)
                } else {
                    1.0
                }
            }
        };
    }
}

/// What kind of frame this is, decided before any GPU work.
#[derive(Clone, Copy)]
pub(super) struct FrameKind {
    /// Presented on the GPU.
    pub(super) gpu_mode: bool,
    /// Temporally antialiased.
    pub(super) temporal: bool,
    /// Shown as the page's next frame, not a preview or capture.
    pub(super) display: bool,
    pub(super) instance_count: usize,
}

/// The work a frame submits before its own command buffer, in submission
/// order: the physically based sky's light, which everything after reads;
/// new textures' mip levels and impostor bakes; staged pose copies; GPU
/// presentation's input copies; environment maps' and reflection probes'
/// work. Profiling captures time each but the sky's, which times its own
/// passes, between marks of its own.
#[derive(Default)]
pub(super) struct Prelude {
    pub(super) sky: Option<wgpu::CommandEncoder>,
    pub(super) mips: Option<wgpu::CommandEncoder>,
    pub(super) poses: Option<wgpu::CommandBuffer>,
    pub(super) inputs: Option<wgpu::CommandEncoder>,
    pub(super) environment: Option<wgpu::CommandEncoder>,
}

impl Prelude {
    /// The span profiling captures time each part as, in submission order.
    const SPANS: [Option<&'static str>; 5] = [
        None,
        Some("pre_mips"),
        Some("pre_poses"),
        Some("pre_inputs"),
        Some("pre_environment"),
    ];

    /// Whether each part holds work, in submission order.
    fn held(&self) -> [bool; 5] {
        [
            self.sky.is_some(),
            self.mips.is_some(),
            self.poses.is_some(),
            self.inputs.is_some(),
            self.environment.is_some(),
        ]
    }

    /// Each part's command buffer, in submission order.
    fn finish(self) -> [Option<wgpu::CommandBuffer>; 5] {
        let finish =
            |encoder: Option<wgpu::CommandEncoder>| encoder.map(|encoder| encoder.finish());
        [
            finish(self.sky),
            finish(self.mips),
            self.poses,
            finish(self.inputs),
            finish(self.environment),
        ]
    }
}

/// A frame's asset uploads.
pub(super) struct AssetUploads {
    pub(super) mesh_bytes: u64,
    pub(super) texture_bytes: u64,
    pub(super) mip_levels: u32,
}

/// How a frame packed its instance storage.
pub(super) struct Packing {
    /// This frame's temporal antialiasing.
    pub(super) temporal_frame: Option<TemporalFrame>,
    write_slots: bool,
    slot_key: Option<(u64, u64)>,
    pub(super) repacked: usize,
    pub(super) poses_repacked: usize,
    pub(super) colors_repacked: usize,
    pub(super) full_upload: bool,
    pub(super) fallback: bool,
}

/// A frame's uniform and instance writes.
pub(super) struct Writes {
    pub(super) lighting: LightingUniform,
    pub(super) camera_bytes: u64,
    pub(super) instance_bytes: u64,
    pub(super) pose_bytes: u64,
    pub(super) color_bytes: u64,
    pub(super) slot_bytes: u64,
    pub(super) ranges: usize,
}

/// What each feature prepared for a frame.
pub(super) struct Features<'s> {
    /// The viewport and depth pyramid levels of occlusion culling.
    pub(super) occlusion: Option<(u32, u32, u32)>,
    pub(super) gpu_prepared: Option<GpuPrepared>,
    pub(super) shadow_frame: Option<ShadowFrame>,
    pub(super) probe_list: &'s [engine_core::RenderReflectionProbe],
    pub(super) dropped_probes: u64,
    pub(super) light_frame: Option<LightFrame>,
    pub(super) post_frame: Option<PostFrame>,
    pub(super) refraction: bool,
    pub(super) transparent_draws: bool,
    pub(super) particle_frame: Option<ParticleFrame>,
    pub(super) transparency: bool,
    pub(super) decal_frame: Option<DecalFrame>,
}

/// The mesh pass kinds a frame draws with, and its impostors' pipeline.
#[derive(Clone, Copy)]
pub(super) struct Passes {
    pub(super) opaque_kind: PassKind,
    pub(super) transparent_kind: PassKind,
    pub(super) impostor_pass: Option<(Targets, bool)>,
}

/// Everything a frame's encoding, submission and stats read.
pub(super) struct FrameState<'s> {
    pub(super) snapshot: &'s RenderSnapshot,
    pub(super) frame: RenderFrame<'s>,
    pub(super) kind: FrameKind,
    pub(super) prelude: Prelude,
    pub(super) assets: AssetUploads,
    pub(super) packing: Packing,
    pub(super) writes: Writes,
    pub(super) features: Features<'s>,
    pub(super) passes: Passes,
    /// Virtual geometry draws its clusters, and the
    /// visibility buffer resolves some of them.
    pub(super) virtual_frame: bool,
    /// Terrain draws.
    pub(super) terrain: bool,
    /// Placed objects draw.
    pub(super) scatter: bool,
    pub(super) resolving: bool,
    /// What the frame's render graph declares.
    pub(super) shape: FrameShape,
}

/// What encoding counted: the opaque passes' draws, shadow draws,
/// reflection probe faces, and the render graph's passes.
#[derive(Default)]
pub(super) struct EncodeReport {
    pub(super) draws: u32,
    pub(super) shadow_draws: u32,
    pub(super) probe_faces: u32,
    pub(super) passes: u64,
}

impl Renderer {
    /// Renders a borrowed core frame using its retained structural and pose
    /// revisions. Lost or outdated surfaces recover on a later frame; device
    /// loss and validation errors permanently fail this renderer, and frame
    /// source errors return before uploads or cache changes. The frame is
    /// exact: never jittered or blended with earlier frames.
    pub fn render_frame(&mut self, frame: RenderFrame<'_>) -> Result<RenderStats, RenderError> {
        self.render_inner(frame, false)
    }

    /// [`Self::render_frame`] for ordinary display frames. With temporal
    /// antialiasing enabled, GPU-presented frames rasterize with a sub-pixel
    /// jitter and blend with the reprojected output of the previous display
    /// frame. Any other frame ends that history.
    pub fn render_display_frame(
        &mut self,
        frame: RenderFrame<'_>,
    ) -> Result<RenderStats, RenderError> {
        self.render_inner(frame, true)
    }

    /// Copies GPU presentation's current poses into a staging buffer between
    /// frames, e.g. right after a fixed tick completes ahead of its due frame.
    /// The next GPU frame copies them on the GPU instead of staging them again
    /// when its gather still has the same version and uploads every current
    /// pose. Poses whose gather stamped fewer than three quarters of them
    /// upload record by record instead, so they are not staged here.
    pub fn prestage_gpu_current(&mut self, current: engine_core::GpuCurrent<'_>) {
        let bytes: &[u8] = bytemuck::cast_slice(current.poses);
        if self.display.suspended
            || !self.gpu_frame
            || bytes.len() < staging::MIN_STAGED_BYTES
            || !gpu_uploads::whole_stream(current.stamped, current.poses.len())
        {
            return;
        }
        let _profile = engine_profile::span("render", "prestage_current");
        if let Some(prestaged) = self.instances.pose_uploads.prestage(&self.device, bytes) {
            self.gpu_presentation.prestage(prestaged, current.version);
        }
    }

    fn render_inner(
        &mut self,
        frame: RenderFrame<'_>,
        display: bool,
    ) -> Result<RenderStats, RenderError> {
        let snapshot = frame.snapshot();
        let _profile = engine_profile::span("render", "render_snapshot");
        if let Ok(failure) = self.failure.lock()
            && let Some(message) = failure.as_ref()
        {
            return Err(RenderError(message.to_string()));
        }
        let (kind, gpu) = self.plan_frame(frame, display)?;
        if self.display.suspended {
            return Ok(self.residency_stats());
        }
        self.switch_fades.advance(
            kind,
            snapshot.elapsed_seconds,
            snapshot.environment.level_of_detail.fade_seconds,
        );
        let Some(surface) = self.display.acquire(&self.device, &mut self.depth)? else {
            return Ok(self.residency_stats());
        };
        let mut prelude = Prelude::default();
        let assets = self.upload_assets(snapshot, kind, &mut prelude.mips);
        let packing = self.pack_instances(frame, kind);
        let profile_writes = engine_profile::span("render", "queue_writes");
        let writes = self.write_frame_data(frame, kind, &packing, &mut prelude.poses);
        // Profiling captures time this frame's passes from here.
        self.profiler.begin_frame(&self.device);
        let features = self.prepare_features(
            frame,
            gpu.as_ref(),
            kind,
            (&packing.temporal_frame, &writes.lighting),
            &mut prelude,
        );
        drop(profile_writes);
        let profile_encoding = engine_profile::span("render", "command_encoding");
        let view = surface
            .texture()
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/frame-encoder"),
            });
        let passes = self.prepare_pipelines(kind, &packing.temporal_frame, &features);
        let (virtual_frame, resolving) = self.prepare_virtual_frame(
            snapshot,
            kind,
            &packing.temporal_frame,
            features.occlusion,
            passes.opaque_kind,
        );
        let terrain = self.prepare_terrain(
            snapshot,
            &packing.temporal_frame,
            passes.opaque_kind,
            (
                kind.gpu_mode && !features.probe_list.is_empty(),
                kind.gpu_mode,
            ),
        );
        // Lookups loop over the horizon records the terrain now binds.
        let horizon_records = self.terrain.horizons.record_count();
        self.shadows
            .set_horizon_records(&self.queue, horizon_records);
        let scatter =
            self.prepare_scatter(snapshot, passes.opaque_kind, passes.impostor_pass.is_some());
        // Water on terrains.
        let water = self.prepare_water(snapshot, passes.opaque_kind);
        // Terrain's horizon maps may have been replaced.
        self.refresh_camera_group();
        let mut shape = self.frame_shape(
            kind,
            &packing,
            &features,
            (virtual_frame, resolving, scatter),
        );
        shape.water = water;
        shape.clouds = self.prepare_cloud_composite(shape.targets);
        let mut state = FrameState {
            snapshot,
            frame,
            kind,
            prelude,
            assets,
            packing,
            writes,
            features,
            passes,
            virtual_frame,
            resolving,
            terrain,
            scatter,
            shape,
        };
        self.prepare_frame_groups(&state);
        // The frame's passes as a render graph: declared,
        // compiled into attachment operations, then encoded in order.
        let mut graph = std::mem::take(&mut self.graph);
        render_graph::plan(&mut graph, &state.shape);
        if let Err(invalid) = graph.compile() {
            self.graph = graph;
            return Err(RenderError(invalid.to_string()));
        }
        let mut report = EncodeReport {
            passes: graph.kept(),
            ..EncodeReport::default()
        };
        for node in graph.kept_passes() {
            // Profiling captures time every pass whole, with whatever it
            // records.
            let stage = node.pass().stage();
            if let Some(span) = self.profiler.span(stage) {
                span.mark(&mut encoder, false);
            }
            // GPU debuggers (PIX, RenderDoc, Xcode) name each pass's work;
            // ordinary page builds make no calls for them.
            #[cfg(any(not(target_arch = "wasm32"), feature = "profiling"))]
            encoder.push_debug_group(node.pass().name());
            self.encode_pass(node, &state, &mut encoder, &view, &mut report);
            #[cfg(any(not(target_arch = "wasm32"), feature = "profiling"))]
            encoder.pop_debug_group();
            if let Some(span) = self.profiler.span(stage) {
                span.mark(&mut encoder, true);
            }
        }
        self.graph = graph;
        // ...and the prelude's command buffers, whose spans must exist before
        // the frame resolves its queries.
        for (span, held) in Prelude::SPANS.into_iter().zip(state.prelude.held()) {
            if let (Some(span), true) = (span, held) {
                let _ = self.profiler.span(span);
            }
        }
        self.profiler.end_frame(&mut encoder);
        drop(profile_encoding);
        self.submit_frame(&mut state, surface, encoder);
        Ok(self.frame_stats(&state, &report))
    }
}
