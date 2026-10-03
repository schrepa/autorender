//! Reflection probe captures (decision 043): still opaque meshes and the
//! sky's haze drawn into probe cube faces after the frame.

use super::*;

impl Renderer {
    /// Records up to `faces` reflection probe cube faces (decision 043), the
    /// nearest probes waiting first and each face at most once a frame (each
    /// has one camera): this frame's still opaque instances (drawn from the
    /// alternate pose buffer when `odd`) and the sky's haze, in linear light,
    /// then a probe's blurred cube once its last face is drawn. Returns the
    /// faces drawn.
    pub(super) fn encode_probe_captures(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        probes: &[engine_core::RenderReflectionProbe],
        faces: u32,
        odd: bool,
        seconds: f32,
        lights: Option<(&LightFrame, &engine_core::LocalLights)>,
    ) -> u32 {
        let lights = lights.filter(|_| self.lights.is_some());
        let kind = PassKind::probe(lights.is_some());
        self.ensure_probe_pipelines(kind);
        self.camera.prepare_probe_groups(
            &self.device,
            &self.shadows,
            &self.environment,
            &self.probes,
        );
        if odd {
            self.instances.prepare_alternate_group(&self.device);
        }
        let mut used = [false; FACES];
        let mut drawn = 0;
        // Profiling captures time every face captured as one span.
        let mut timed = false;
        while drawn < faces {
            let Some((cube, face, probe)) = self.probes.slots.next_face(probes) else {
                break;
            };
            if used[face] {
                break;
            }
            used[face] = true;
            if !timed && let Some(span) = self.profiler.span("probe_captures") {
                span.mark(encoder, false);
                timed = true;
            }
            let view_projection = face_view_projection(probe.center, face);
            let camera = CameraUniform::probe(view_projection, probe.center, seconds);
            self.queue
                .write_buffer(self.probes.camera(face), 0, bytemuck::bytes_of(&camera));
            let poses = self.instances.frame_poses(usize::from(odd));
            if !self.gpu_presentation.encode_probe_face(
                &self.device,
                &self.queue,
                encoder,
                face,
                self.environment.size(),
                view_projection,
                poses.current(),
            ) {
                break;
            }
            // Still lamps light the capture: lights binned for this face.
            if let (Some((frame, settings)), Some(pass)) = (lights, self.lights.as_mut()) {
                pass.encode_probe_face(
                    &self.device,
                    &self.queue,
                    encoder,
                    face,
                    view_projection,
                    self.environment.size(),
                    settings,
                    frame,
                    poses,
                    self.gpu_presentation.still_frames(),
                );
            }
            let depth = self
                .probes
                .depth(&self.device, self.environment.size(), DEPTH_FORMAT)
                .clone();
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("engine/probe-face"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: self.environment.radiance_face(face),
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
                pass.set_bind_group(0, self.camera.probe_group(face), &[]);
                let poses_group = if odd {
                    self.instances.pose_group(1)
                } else {
                    self.instances.pose_group(0)
                };
                pass.set_bind_group(2, poses_group, &[]);
                if lights.is_some() {
                    let lights = self.lights.as_ref().expect("checked above");
                    pass.set_bind_group(3, lights.probe_fragment_group(face), &[]);
                }
                let mut bound = None;
                let mut bindings = Bindings::default();
                for &index in &self.draw_order.order {
                    let index = index as usize;
                    let batch = &self.plan.batches[index];
                    // Only still opaque meshes without joints are captured.
                    let mesh = &self.meshes[&batch.mesh];
                    if mesh.skinned {
                        continue;
                    }
                    let (group, variant) = self.materials.binding(batch.material);
                    let variant = variant.with_extended(mesh.extended).uber();
                    if bound != Some(variant) {
                        pass.set_pipeline(&self.mesh_pipelines.variants[&(kind, variant)]);
                        bound = Some(variant);
                    }
                    bindings.group(&mut pass, 1, group);
                    let draw = mesh.draw(&self.mesh_arena);
                    bindings.vertex(&mut pass, 0, draw.vertices);
                    if let Some(extras) = draw.extras.filter(|_| mesh.extended) {
                        bindings.vertex(&mut pass, 2, extras);
                    }
                    bindings.index(&mut pass, draw.indices);
                    let (ids, draws, offset) =
                        self.gpu_presentation.probe_draw_source(batch, index);
                    bindings.ids(&mut pass, 1, ids);
                    pass.draw_indexed_indirect(draws, offset);
                }
                pass.set_pipeline(self.mesh_pipelines.sky(SkyTargets::Probe));
                pass.draw(0..3, 0..1);
            }
            drawn += 1;
            if self.probes.slots.face_drawn() {
                self.environment.finish(encoder, cube);
            }
        }
        if timed && let Some(span) = self.profiler.span("probe_captures") {
            span.mark(encoder, true);
        }
        drawn
    }

    /// The reflection probe captures' pipelines (decision 043): the sky's,
    /// and `kind`'s uber variants (decision 049) for this frame's still
    /// opaque meshes.
    pub(super) fn ensure_probe_pipelines(&mut self, kind: PassKind) {
        self.ensure_sky(SkyTargets::Probe);
        let mut missing = Vec::new();
        for batch in &self.plan.batches {
            let Some(mesh) = self.meshes.get(&batch.mesh) else {
                continue;
            };
            if batch.transparent || mesh.skinned {
                continue;
            }
            let (_, variant) = self.materials.binding(batch.material);
            let variant = variant.with_extended(mesh.extended).uber();
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
}
