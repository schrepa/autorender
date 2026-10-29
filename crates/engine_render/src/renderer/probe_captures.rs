use engine_core::MeshHandle;

use super::*;
use super::meshes::{BatchBinds, relit};
use crate::entries::PassKind;
use crate::gpu_data::{CameraUniform, FAR_DEPTH};
use crate::lights::LightPass;
use crate::parts::{DrawAt, Encode};
use crate::probes::{FACES, ProbePass, face_view_projection, probe_list};
use crate::shadows::CasterDraw;
use crate::sky::SkyPass;

impl ProbePass {
    pub(crate) fn capture(&mut self, cx: &mut Encode<'_>, encoder: &mut wgpu::CommandEncoder) {
        self.slots.tick();
        if !cx.core.gpu_presentation.stillness_settled() {
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

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
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
        let settings = &snapshot.environment.local_lights;
        let Some(environment) = parts.get::<SkyPass>().map(SkyPass::environment) else {
            return 0;
        };
        let lights = parts.get::<LightPass>().filter(|lights| lights.lit());
        let kind = PassKind::probe(lights.is_some());
        let relights = cx.core.mesh_pipelines.cooking
            && parts.get::<LightPass>().is_some_and(LightPass::created);
        cx.core.ensure_probe_pipelines(kind, relights);
        if odd {
            cx.core.instances.prepare_pose_group(&cx.core.device, 1);
        }
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
            let poses = core.instances.frame_poses(usize::from(odd));
            if !core.gpu_presentation.encode_probe_face(
                &core.device,
                &core.queue,
                encoder,
                face,
                environment.size(),
                view_projection,
                poses.current(),
            ) {
                break;
            }
            if let Some(lights) = lights {
                lights.encode_probe_face(
                    (&core.device, &core.queue),
                    encoder,
                    (face, view_projection, environment.size()),
                    settings,
                    poses,
                    core.gpu_presentation.still_frames(),
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
                let core = &*cx.core;
                pass.set_bind_group(0, core.probe_group(face, &["refraction"]), &[]);
                let poses_group = if odd {
                    core.instances.pose_group(1)
                } else {
                    core.instances.pose_group(0)
                };
                pass.set_bind_group(2, poses_group, &[]);
                if let Some(group) = lights.and_then(|lights| lights.probe_group(face)) {
                    pass.set_bind_group(3, &group, &[]);
                }
                let mut binds = BatchBinds::default();
                for &index in &core.draw_order.order {
                    let index = index as usize;
                    let batch = &core.plan.batches[index];
                    core.bind_batch(&mut pass, &mut binds, (kind, true), batch);
                    let (ids, draws, offset) = core.gpu_presentation.probe_draw_source(batch, index);
                    binds.bindings.ids(&mut pass, 1, ids);
                    pass.draw_indexed_indirect(draws, offset);
                }
                cx.draw_hook(
                    DrawAt::ProbeFace {
                        face,
                        center: probe.center,
                        lit: lights.is_some(),
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

impl Core {
    pub(crate) fn ensure_probe_pipelines(&mut self, kind: PassKind, relights: bool) {
        if relights {
            self.create_probe_variants(relit(kind));
        }
        self.create_probe_variants(kind);
    }

    fn create_probe_variants(&mut self, kind: PassKind) {
        let key = (self.structure_key(), kind);
        if self.structure_keys.probes == Some(key) {
            return;
        }
        self.structure_keys.probes = Some(key);
        let mut missing = Vec::new();
        for batch in &self.plan.batches {
            let Some(mesh) = self.meshes.get(&batch.mesh) else {
                continue;
            };
            let variant = self.batch_binding(batch, Some(mesh)).1.uber();
            if !self.mesh_pipelines.variants.contains_key(&(kind, variant))
                && !missing.contains(&variant)
            {
                missing.push(variant);
            }
        }
        for variant in missing {
            self.ensure_mesh_layout(kind);
            let pipeline = self.create_variant(kind, variant);
            self.mesh_pipelines
                .variants
                .insert((kind, variant), pipeline);
        }
    }

    pub(crate) fn caster_draw(&self, mesh: MeshHandle, levels: bool) -> CasterDraw {
        self.meshes.get(&mesh).map_or(
            CasterDraw {
                index_count: 0,
                first_index: 0,
                base_vertex: 0,
                lods: Vec::new(),
            },
            |mesh| {
                let draw = mesh.draw(&self.mesh_arena);
                let lods = if !levels {
                    Vec::new()
                } else {
                    mesh.lods
                        .iter()
                        .enumerate()
                        .map(|(level, &(offset, count, error))| {
                            let (offset, count) = mesh.caster_range(level + 1, (offset, count));
                            (draw.first_index + offset, count, error)
                        })
                        .collect()
                };
                let (offset, count) = mesh.caster_range(0, (0, draw.index_count));
                CasterDraw {
                    index_count: count,
                    first_index: draw.first_index + offset,
                    base_vertex: draw.base_vertex,
                    lods,
                }
            },
        )
    }
}
