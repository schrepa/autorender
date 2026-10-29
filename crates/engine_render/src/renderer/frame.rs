use super::*;
use crate::exposure::{ExposureFrame, ExposurePass};
use crate::gpu_calls::Recorded;
use crate::gpu_presentation::{CullingView, GpuPrepared, PresentationInputs, RecordsKey, Staging};
use crate::ground_cover::CoverFrame;
use crate::render_graph::{self, Effects, FeaturePass, FrameShape, Node, Pass, Phase, Resource};
use crate::shadows::{MapSources, ShadowFrame, ShadowInputs};

mod debug;
mod encode;
mod opaque;
mod prepare;
mod submit;

#[derive(Default)]
pub(super) struct SwitchFades {
    presented: f32,
    pub(super) share: f32,
}

impl SwitchFades {
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

#[derive(Clone, Copy)]
pub(super) struct FrameKind {
    pub(super) gpu_mode: bool,
    pub(super) temporal: bool,
    pub(super) display: bool,
    pub(super) debug: bool,
    pub(super) instance_count: usize,
}

#[derive(Default)]
pub(super) struct Prelude {
    pub(super) sky: Option<wgpu::CommandEncoder>,
    pub(super) mips: Option<wgpu::CommandEncoder>,
    pub(super) poses: Option<wgpu::CommandBuffer>,
    pub(super) inputs: Option<wgpu::CommandEncoder>,
    pub(super) environment: Option<wgpu::CommandEncoder>,
}

impl Prelude {
    const SPANS: [Option<&'static str>; 5] = [
        None,
        Some("pre_mips"),
        Some("pre_poses"),
        Some("pre_inputs"),
        Some("pre_environment"),
    ];

    fn parts(self) -> [Option<Recorded>; 5] {
        [
            self.sky.map(Recorded::Encoder),
            self.mips.map(Recorded::Encoder),
            self.poses.map(Recorded::Buffer),
            self.inputs.map(Recorded::Encoder),
            self.environment.map(Recorded::Encoder),
        ]
    }
}

pub(super) struct AssetUploads {
    pub(super) mesh_bytes: u64,
    pub(super) texture_bytes: u64,
    pub(super) mip_levels: u32,
}

pub(super) struct Packing {
    pub(super) temporal_frame: Option<TemporalFrame>,
    write_slots: bool,
    slot_key: Option<(u64, u64)>,
    pub(super) repacked: usize,
    pub(super) poses_repacked: usize,
    pub(super) colors_repacked: usize,
    pub(super) full_upload: bool,
    pub(super) fallback: bool,
}

pub(super) struct Writes {
    pub(super) lighting: LightingUniform,
    pub(super) camera_bytes: u64,
    pub(super) instance_bytes: u64,
    pub(super) pose_bytes: u64,
    pub(super) color_bytes: u64,
    pub(super) slot_bytes: u64,
    pub(super) ranges: usize,
}

pub(super) struct Features<'s> {
    pub(super) gpu_prepared: Option<GpuPrepared>,
    pub(super) shadow_frame: Option<ShadowFrame>,
    pub(super) probe_list: &'s [engine_core::RenderReflectionProbe],
    pub(super) frames: FeatureFrames,
}

impl Features<'_> {
    pub(super) fn light_frame(&self) -> Option<&LightFrame> {
        self.frames.lights.as_ref().map(Ready::frame)
    }

    pub(super) fn post_frame(&self) -> Option<Effects> {
        self.frames.post.as_ref().map(|post| *post.frame())
    }
}

#[derive(Clone, Copy)]
pub(super) struct Passes {
    pub(super) opaque_kind: PassKind,
    pub(super) impostor_pass: Option<(Targets, bool)>,
}

pub(super) struct FrameState<'s> {
    pub(super) snapshot: &'s RenderSnapshot,
    pub(super) frame: RenderFrame<'s>,
    pub(super) kind: FrameKind,
    pub(super) assets: AssetUploads,
    pub(super) packing: Packing,
    pub(super) writes: Writes,
    pub(super) features: Features<'s>,
    pub(super) passes: Passes,
    pub(super) terrain: bool,
    pub(super) scatter: bool,
    pub(super) shape: FrameShape,
}

#[derive(Default)]
pub(super) struct EncodeReport {
    pub(super) draws: u32,
    pub(super) shadow_draws: u32,
    pub(super) passes: u64,
}

impl Renderer {
    pub fn render_frame(&mut self, frame: RenderFrame<'_>) -> Result<RenderStats, Error> {
        self.render_inner(frame, false)
    }

    fn prepare_exposure(
        &mut self,
        snapshot: &RenderSnapshot,
        drawn: bool,
        inputs: &mut Option<wgpu::CommandEncoder>,
    ) -> Option<Ready<ExposureFrame>> {
        let _owner = crate::memory::owned(Pass::Feature(FeaturePass::Exposure));
        let needs = drawn && snapshot.environment.auto_exposure.enabled;
        let (device, queue) = (&self.device, &self.queue);
        let (post, depth) = (self.features.post.get(), &self.depth.view);
        let lighting = self.camera.lighting_buffer();
        let drawn = self.display.region.render.pixels();
        self.features.exposure.prepare(
            (device, needs),
            || ExposurePass::new(device),
            |exposure| {
                let targets = (post?.scene(), depth, drawn);
                let frame = exposure.prepare((device, queue), snapshot, targets, lighting);
                let encoder = inputs.get_or_insert_with(|| {
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("engine/exposure-copy"),
                    })
                });
                exposure.show(encoder, lighting);
                Some(frame)
            },
        )
    }

    fn frame_targets(
        &self,
        kind: FrameKind,
        temporal_frame: &Option<TemporalFrame>,
        post_frame: Option<Effects>,
    ) -> Targets {
        match (temporal_frame, post_frame) {
            _ if kind.debug => Targets::Debug,
            (Some(_), Some(_)) if self.features.temporal.get().is_some() => Targets::Temporal,
            (_, Some(_)) => Targets::Post,
            (_, None) => Targets::Canvas,
        }
    }

    pub fn render_display_frame(&mut self, frame: RenderFrame<'_>) -> Result<RenderStats, Error> {
        self.render_inner(frame, true)
    }

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

    fn mark_stages(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        end: Option<Pass>,
        begin: Option<Pass>,
    ) {
        let span = |pass: Option<Pass>| self.profiler.span(pass?.stage());
        submit::mark_between(encoder, span(end), span(begin));
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn render_inner(
        &mut self,
        frame: RenderFrame<'_>,
        display: bool,
    ) -> Result<RenderStats, Error> {
        let snapshot = frame.snapshot();
        let _profile = engine_profile::span("render", "render_snapshot");
        self.check_health()?;
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
        self.specialize_shading(snapshot);
        let mut prelude = Prelude::default();
        let assets = self.upload_assets(snapshot, kind, &mut prelude.mips);
        let packing = self.pack_instances(frame, kind);
        let profile_writes = engine_profile::span("render", "queue_writes");
        if !snapshot.environment.auto_exposure.enabled
            && self
                .features
                .exposure
                .get_mut()
                .is_some_and(ExposurePass::stop_showing)
        {
            self.camera.forget_lighting();
        }
        let writes = self.write_frame_data(frame, kind, &packing, &mut prelude.poses);
        self.profiler.begin_frame(&self.device);
        let mut features = self.prepare_features(
            frame,
            gpu.as_ref(),
            kind,
            (&packing.temporal_frame, &writes.lighting),
            &mut prelude,
        );
        let drawn = features.frames.post.is_some();
        features.frames.exposure = self.prepare_exposure(snapshot, drawn, &mut prelude.inputs);
        drop(profile_writes);
        self.hand_prelude(prelude);
        let profile_encoding = engine_profile::span("render", "command_encoding");
        let view = surface
            .texture()
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.frame_encoder();
        self.display.time_begin(&mut encoder);
        let passes = self.prepare_pipelines(kind, &packing.temporal_frame, &features);
        let terrain = self.prepare_terrain(
            snapshot,
            &packing.temporal_frame,
            passes.opaque_kind,
            (
                kind.gpu_mode && !features.probe_list.is_empty(),
                kind.gpu_mode,
            ),
        );
        self.water_settle.prepare(
            &self.device,
            &self.queue,
            (&self.mesh_shader, &self.terrain),
        );
        self.water_settle.encode(&mut encoder);
        features.frames.ground_cover = if terrain {
            self.features
                .ground_cover
                .prepare(|cover| cover.growing().then_some(CoverFrame))
        } else {
            None
        };
        let horizon_records = self.terrain.horizons.record_count();
        self.shadows
            .set_horizon_records(&self.queue, horizon_records);
        features.frames.scatter =
            self.prepare_scatter(snapshot, passes.opaque_kind, passes.impostor_pass.is_some());
        let scatter = features.frames.scatter.is_some();
        features.frames.occupancy = self.prepare_occupancy();
        features.frames.water = self.prepare_water(snapshot, passes.opaque_kind);
        self.refresh_camera_group();
        let shape = self.frame_shape(kind, (&packing, &features, &passes), (scatter, terrain));
        features.frames.clouds = self.prepare_cloud_composite(shape.targets);
        features.frames.debug_view = self.prepare_debug_view(snapshot, kind);
        let state = FrameState {
            snapshot,
            frame,
            kind,
            assets,
            packing,
            writes,
            features,
            passes,
            terrain,
            scatter,
            shape,
        };
        self.prepare_frame_groups(&state);
        let mut graph = std::mem::take(&mut self.graph);
        render_graph::plan(&mut graph, &state.shape);
        RenderFeatures::declare(&state.features.frames, &state.shape, &mut graph);
        if let Err(invalid) = graph.compile() {
            self.graph = graph;
            self.queue.settle();
            return Err(invalid.error());
        }
        let mut report = EncodeReport {
            passes: graph.kept(),
            ..EncodeReport::default()
        };
        let mut batch = 0;
        let mut timed = None;
        for node in graph.kept_passes() {
            let pass = node.pass();
            let next = submit::batch(pass);
            if next != batch {
                batch = next;
                self.mark_stages(&mut encoder, timed.take(), None);
                encoder = self.hand_batch(encoder);
            }
            self.mark_stages(&mut encoder, timed.replace(pass), Some(pass));
            #[cfg(any(not(target_arch = "wasm32"), feature = "profiling"))]
            encoder.push_debug_group(pass.name());
            self.encode_pass(node, &state, &mut encoder, &view, &mut report);
            #[cfg(any(not(target_arch = "wasm32"), feature = "profiling"))]
            encoder.pop_debug_group();
        }
        self.mark_stages(&mut encoder, timed, None);
        self.graph = graph;
        #[cfg(feature = "dev")]
        self.encode_probe(&state, &mut encoder, surface.texture());
        self.display.time_end(&self.device, &mut encoder);
        self.profiler.end_frame(&mut encoder);
        drop(profile_encoding);
        self.submit_frame(&state, surface, encoder);
        Ok(self.frame_stats(&state, &report))
    }
}
