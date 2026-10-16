//! Per-view preparation and current opaque phase coverage. Collection runs
//! after that phase's visibility inputs, without writing the frame's targets.

use super::super::light_depth::DepthMeshKind;
use super::*;

impl Renderer {
    pub(super) fn prepare_light_depth(
        &mut self,
        frame: RenderFrame<'_>,
        packing: &Packing,
        features: &mut Features<'_>,
        opaque: PassKind,
    ) {
        let requested = frame.snapshot().environment.local_lights.depth_occupancy;
        let Some(lights) = self.features.lights.get_mut() else {
            self.mesh_pipelines.depth = None;
            return;
        };
        if !requested && lights.depth().is_none() {
            self.mesh_pipelines.depth = None;
            return;
        }
        let eligible = self
            .draw_order
            .order
            .iter()
            .filter(|&&index| self.virtual_draws.drawn.get(index as usize) != Some(&true))
            .count() as u32;
        let lit = features.light_frame().is_some();
        let changed = lights.configure_depth(
            &self.device,
            &self.queue,
            self.display.size(),
            (frame.identity(), eligible),
            requested && lit,
        );
        if changed && lit {
            let (device, queue, size) = (&self.device, &self.queue, self.display.size());
            let parity = packing
                .temporal_frame
                .as_ref()
                .map_or(0, |frame| frame.parity);
            let poses = self.instances.frame_poses(parity);
            features.frames.lights = self.features.lights.prepare(
                (device, true),
                || LightPass::new(device),
                |lights| {
                    lights.prepare(
                        device,
                        queue,
                        frame.snapshot(),
                        size,
                        poses,
                        (frame.identity(), frame.lights_revision()),
                    )
                },
            );
        }
        self.ensure_depth_variants(opaque);
    }

    pub(super) fn encode_light_depth_collect(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        phase: usize,
    ) {
        let Some(depth) = self.features.lights.get().and_then(LightPass::depth) else {
            return;
        };
        let name = if phase == 0 {
            "light_depth_collect"
        } else {
            "light_depth_collect_phase_two"
        };
        let span = self.profiler.span(name);
        if let Some(span) = span {
            span.mark(encoder, false);
        }
        depth.reset(encoder, phase);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/light-depth-collection"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth.scratch_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.draw_ordinary(state, &mut pass, phase, Some(DepthMeshKind::Collect));
        }
        if let Some(span) = span {
            span.mark(encoder, true);
        }
    }

    pub(super) fn encode_light_depth_filter(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        phase: usize,
    ) {
        let Some(depth) = self.features.lights.get().and_then(LightPass::depth) else {
            return;
        };
        let name = if phase == 0 {
            "light_depth_filter"
        } else {
            "light_depth_filter_phase_two"
        };
        let span = self.profiler.span(name);
        if let Some(span) = span {
            span.mark(encoder, false);
        }
        depth.filter(&self.device, encoder, phase);
        if let Some(span) = span {
            span.mark(encoder, true);
        }
    }
}
