use engine_faces::environment::EnvironmentFace;
use engine_faces::lights::LightListFace;
use engine_render::gpu_data::{CameraUniform, FAR_DEPTH};
use engine_render::parts::{DrawAt, Encode};
use engine_render::renderer::DEPTH_FORMAT;

use crate::probes::{FACES, ProbePass, face_view_projection, probe_list};

impl ProbePass {
    pub(crate) fn capture(&mut self, cx: &mut Encode<'_>, encoder: &mut wgpu::CommandEncoder) {
        self.slots.tick();
        if !cx.core.views_settled() {
            return;
        }
        let odd = cx
            .frame
            .temporal
            .as_ref()
            .is_some_and(|frame| frame.parity == 1);
        let faces = self.encode_captures(cx, encoder, odd);
        self.faces = faces;
    }

    fn encode_captures(
        &mut self,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
        odd: bool,
    ) -> u32 {
        let parts = cx.parts;
        let snapshot = cx.frame.snapshot;
        let probes = probe_list(snapshot);
        let faces = snapshot.environment.reflection_probes.faces_per_frame;
        let seconds = snapshot.elapsed_seconds;
        let Some(environment) = parts.face::<dyn EnvironmentFace>() else {
            return 0;
        };
        let lights = parts.face::<dyn LightListFace>();
        let parity = usize::from(odd);
        let mut used = [false; FACES];
        let mut drawn = 0;
        let mut timed = false;
        while drawn < faces {
            let Some((cube, face, probe)) = self.slots.next_face(probes) else {
                break;
            };
            if used[face] {
                break;
            }
            used[face] = true;
            if !timed && let Some(span) = cx.core.profiler.span("probe_captures") {
                span.mark(encoder, false);
                timed = true;
            }
            let view_projection = face_view_projection(probe.center, face);
            let camera =
                CameraUniform::probe(view_projection, probe.center, seconds, environment.size());
            cx.core
                .queue
                .write_buffer(self.camera(face), 0, bytemuck::bytes_of(&camera));
            let core = &mut *cx.core;
            if !core.cull_view(encoder, face, (view_projection, environment.size()), parity) {
                break;
            }
            if let Some(lights) = lights {
                lights.encode_probe_face(
                    (&core.device, &core.queue),
                    encoder,
                    (face, view_projection, environment.size()),
                );
            }
            let depth = self
                .depth(&core.device, environment.size(), DEPTH_FORMAT)
                .clone();
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("engine/probe-face"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: environment.radiance_face(face),
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(FAR_DEPTH),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                let lights_group = lights.and_then(|lights| lights.probe_group(face));
                cx.core
                    .draw_view(&mut pass, face, parity, lights_group.as_ref());
                cx.draw_hook(
                    DrawAt::View {
                        view: face,
                        center: probe.center,
                    },
                    &mut pass,
                );
            }
            drawn += 1;
            if self.slots.face_drawn() {
                environment.finish(encoder, cube);
            }
        }
        if timed && let Some(span) = cx.core.profiler.span("probe_captures") {
            span.mark(encoder, true);
        }
        drawn
    }
}
