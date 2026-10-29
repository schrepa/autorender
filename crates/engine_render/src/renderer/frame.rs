use super::*;
use crate::parts::{NodeViews, Prelude, Step};

mod encode;
mod prepare;
mod submit;
mod views;

#[derive(Default)]
pub(crate) struct SwitchFades {
    presented: f32,
    pub(crate) share: f32,
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

pub(super) struct AssetUploads {
    pub(super) mesh_bytes: u64,
    pub(super) texture_bytes: u64,
    pub(super) mip_levels: u32,
}

pub(super) struct Packing {
    write_slots: bool,
    slot_key: Option<(u64, u64)>,
    pub(super) repacked: usize,
    pub(super) poses_repacked: usize,
    pub(super) colors_repacked: usize,
    pub(super) full_upload: bool,
    pub(super) fallback: bool,
}

pub(super) struct Writes {
    pub(super) camera_bytes: u64,
    pub(super) instance_bytes: u64,
    pub(super) pose_bytes: u64,
    pub(super) color_bytes: u64,
    pub(super) slot_bytes: u64,
    pub(super) ranges: usize,
}

pub(super) struct Uploaded {
    pub(super) assets: AssetUploads,
    pub(super) packing: Packing,
    pub(super) writes: Writes,
}

pub(super) struct EncodeReport {
    pub(super) draws: u32,
}

impl Renderer {
    pub fn render_frame(&mut self, frame: RenderFrame<'_>) -> Result<RenderStats, Error> {
        self.render_inner(frame, false)
    }

    pub fn render_display_frame(&mut self, frame: RenderFrame<'_>) -> Result<RenderStats, Error> {
        self.render_inner(frame, true)
    }

    pub fn prestage_gpu_current(&mut self, current: engine_core::GpuCurrent<'_>) {
        let bytes: &[u8] = bytemuck::cast_slice(current.poses);
        let core = &mut self.core;
        if core.display.suspended
            || !core.gpu_frame
            || bytes.len() < staging::MIN_STAGED_BYTES
            || !gpu_uploads::whole_stream(current.stamped, current.poses.len())
        {
            return;
        }
        let _profile = engine_profile::span("render", "prestage_current");
        if let Some(prestaged) = core.instances.pose_uploads.prestage(&core.device, bytes) {
            core.gpu_presentation.prestage(prestaged, current.version);
        }
    }

    fn render_inner(
        &mut self,
        frame: RenderFrame<'_>,
        display: bool,
    ) -> Result<RenderStats, Error> {
        let snapshot = frame.snapshot();
        let _profile = engine_profile::span("render", "render_snapshot");
        self.check_health()?;
        let kind = self.plan_frame(frame, display)?;
        if self.core.display.suspended {
            return Ok(self.core.residency_stats());
        }
        self.core.switch_fades.advance(
            kind,
            snapshot.elapsed_seconds,
            snapshot.environment.level_of_detail.fade_seconds,
        );
        let core = &mut self.core;
        let Some(surface) = core.display.acquire(&core.device, &mut core.depth)? else {
            return Ok(self.core.residency_stats());
        };
        if kind.debug {
            let core = &mut self.core;
            core.debug.prepare(&core.device, core.display.size());
        }
        let mut info = FrameInfo {
            snapshot,
            frame,
            kind,
            temporal: None,
            passes: None,
            shape: None,
            kept: 0,
            written: 0,
        };
        let mut prelude = Prelude::default();
        prelude.attribute(crate::parts::CORE_BRICK);
        let assets = self.upload_assets(snapshot, kind, &mut prelude);
        let packing = self.pack_instances(&mut info);
        let profile_writes = engine_profile::span("render", "queue_writes");
        let mut writes = self.write_frame_data(&info, &packing, &mut prelude);
        self.core.profiler.begin_frame(&self.core.device);
        self.prepare_parts(&info, &mut prelude);
        let core = &mut self.core;
        writes.camera_bytes += core.camera.write_pieces(&core.queue, &self.parts);
        drop(profile_writes);
        self.hand_prelude(prelude);
        let profile_encoding = engine_profile::span("render", "command_encoding");
        let canvas = surface
            .texture()
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.frame_encoder(0);
        self.core.display.time_begin(&mut encoder);
        info.passes = Some(self.prepare_pipelines(&info));
        self.steps(&info, |part, cx| part.prepare_draws(cx, &mut encoder));
        self.refresh_shared(&info);
        if info.temporal.is_some() {
            let core = &mut self.core;
            core.previous
                .prepare_pose_groups(&core.device, &core.instances);
        }
        info.shape = Some(self.frame_shape(&info));
        self.steps(&info, |part, cx| part.finish(cx));
        let mut graph = std::mem::take(&mut self.graph);
        let planned = self
            .plan_graph(&mut graph, &info)
            .and_then(|()| self.node_views(&graph, &canvas));
        let views = match planned {
            Ok(views) => views,
            Err(error) => {
                self.graph = graph;
                self.core.queue.settle();
                return Err(error);
            }
        };
        info.kept = graph.kept();
        info.written = graph.written();
        let (mut encoder, draws, batch) = self.encode_passes(&graph, &views, &info, encoder);
        self.graph = graph;
        #[cfg(feature = "dev")]
        self.encode_probe(&info, &mut encoder, surface.texture());
        self.core.display.time_end(&self.core.device, &mut encoder);
        self.close_frame(&mut encoder, batch);
        drop(profile_encoding);
        self.submit_frame(kind, surface, encoder);
        self.core.previous.after_submit(info.temporal.as_ref());
        let uploaded = Uploaded {
            assets,
            packing,
            writes,
        };
        let report = EncodeReport { draws };
        let stats = self.frame_stats(&info, &uploaded, &report);
        self.parts.after_submit();
        Ok(stats)
    }

    fn prepare_parts(&mut self, info: &FrameInfo<'_>, prelude: &mut Prelude) {
        let (parts, core, table) = (&mut self.parts, &mut self.core, &self.table);
        for index in 0..parts.len() {
            let decl = parts.decl(index);
            let _owner = crate::memory::owned(decl.brick);
            prelude.attribute(decl.passes.first().map_or(decl.name, |pass| pass.name));
            parts.with_part(index, |part, parts| {
                part.prepare(
                    &mut Step {
                        core,
                        parts,
                        table,
                        frame: info,
                    },
                    prelude,
                );
            });
        }
    }
}
