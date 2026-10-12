//! The opaque passes: meshes, impostors' cards, virtual geometry, terrain
//! (shaded deferred where it can be), placed objects and the sky, in one
//! phase or two around occlusion culling's test, and the visibility
//! buffer's triangle IDs, tiles and resolves.

use super::*;

impl Renderer {
    /// The sky of the frame's opaque targets, drawn after its last opaque
    /// geometry.
    fn sky_pipeline(&self, state: &FrameState<'_>) -> &wgpu::RenderPipeline {
        self.mesh_pipelines
            .sky(SkyTargets::Frame(state.shape.targets))
    }

    /// The pose buffers of this frame and the previous one, as the virtual
    /// pose groups pick them.
    fn pose_parities(&self, state: &FrameState<'_>) -> (usize, usize) {
        match state
            .packing
            .temporal_frame
            .as_ref()
            .zip(self.temporal.as_ref())
        {
            Some((frame, _)) if frame.motion_valid => (frame.parity, 1 - frame.parity),
            Some((frame, _)) => (frame.parity, frame.parity),
            None => (0, 0),
        }
    }

    /// The opaque pass of `phase` (two around the occlusion test), with the sky
    /// after the last one unless the visibility buffer resolves it; returns its
    /// draws, the sky's aside.
    pub(super) fn encode_opaque(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
        phase: usize,
    ) -> u32 {
        let phases = usize::from(state.shape.phases);
        let resolve_phase = state.shape.resolves[phase];
        // The first of this phase's camera bins.
        let virtual_first = match phase {
            0 => Some(0),
            _ => self.virtual_draws.phase_two,
        }
        .filter(|_| state.virtual_frame);
        // Profiled frames time both phases as one opaque span, which the
        // visibility buffer's last resolve ends when it resolves this phase.
        let (begins, ends) = (phase == 0, phase + 1 == phases && !resolve_phase);
        // Terrain, the main occluder, draws in the first phase
        // so occlusion culling's pyramid holds it; shaded deferred
        // where it can be.
        if phase == 0 && state.shape.deferred {
            return self.encode_opaque_deferred(
                state,
                encoder,
                view,
                node,
                (phase, phases),
                (begins, ends),
                virtual_first,
            );
        }
        let timestamp_writes = (begins || ends)
            .then(|| self.profiler.span("opaque_meshes"))
            .flatten()
            .map(|span| span.render_part(begins, ends));
        let (colors, count) = self.color_attachments(node, view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/opaque-meshes"),
            color_attachments: &colors[..count],
            depth_stencil_attachment: self.depth_attachment(node),
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let mut draws = if state.kind.instance_count > 0 {
            self.draw_opaque(state, &mut pass, phase, virtual_first)
        } else {
            0
        };
        // Terrain, the main occluder: in the first phase, so
        // occlusion culling's pyramid holds it.
        if phase == 0 {
            draws += self.draw_terrain(state, &mut pass);
            draws += self.draw_scatter(state, &mut pass);
        }
        // Last, so depth testing leaves the sky only uncovered pixels.
        if phase + 1 == phases && !resolve_phase {
            self.draw_sky(state, &mut pass);
        }
        draws
    }

    /// The first opaque phase with terrain shaded deferred:
    /// the meshes; terrain's visibility, writing each pixel's terrain and
    /// depth; each terrain's resolve, shading its pixels once each; then
    /// its cover, placed objects and the sky; each in a pass of its own.
    #[expect(clippy::too_many_arguments, reason = "a pass's disjoint borrows")]
    fn encode_opaque_deferred(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
        (phase, phases): (usize, usize),
        (begins, ends): (bool, bool),
        virtual_first: Option<usize>,
    ) -> u32 {
        let resolve_phase = state.shape.resolves[phase];
        let loaded = wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        };
        let loaded_depth = wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        };
        // The meshes, as the opaque pass's start.
        let mut draws = 0;
        {
            let timestamp_writes = begins
                .then(|| self.profiler.span("opaque_meshes"))
                .flatten()
                .map(|span| span.render_part(true, false));
            let (colors, count) = self.color_attachments(node, view);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/opaque-meshes"),
                color_attachments: &colors[..count],
                depth_stencil_attachment: self.depth_attachment(node),
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if state.kind.instance_count > 0 {
                draws += self.draw_opaque(state, &mut pass, phase, virtual_first);
            }
        }
        let Some(ids) = &self.terrain_ids else {
            return draws;
        };
        let depth = wgpu::RenderPassDepthStencilAttachment {
            view: &self.depth.view,
            depth_ops: Some(loaded_depth),
            stencil_ops: None,
        };
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/terrain-ids"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &ids.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(depth.clone()),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            draws += self.draw_terrain_ids(&mut pass);
        }
        let (mut colors, count) = self.color_attachments(node, view);
        for color in colors.iter_mut().flatten() {
            color.ops = loaded;
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/terrain-resolve"),
                color_attachments: &colors[..count],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            draws += self.draw_terrain_resolve(state, &mut pass);
        }
        let timestamp_writes = ends
            .then(|| self.profiler.span("opaque_meshes"))
            .flatten()
            .map(|span| span.render_part(false, true));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/opaque-meshes"),
            color_attachments: &colors[..count],
            depth_stencil_attachment: Some(depth),
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        draws += self.draw_terrain_cover(state, &mut pass);
        draws += self.draw_scatter(state, &mut pass);
        if phase + 1 == phases && !resolve_phase {
            self.draw_sky(state, &mut pass);
        }
        draws
    }

    /// The sky over the pixels nothing covers, after the opaque surfaces.
    fn draw_sky(&self, state: &FrameState<'_>, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(self.sky_pipeline(state));
        pass.set_bind_group(0, &self.camera.group, &[]);
        pass.set_bind_group(1, self.clouds.sky_group(), &[]);
        pass.draw(0..3, 0..1);
    }

    /// Occlusion culling's test between the opaque phases: the
    /// depth pyramid of phase one's depths, the test and phase two's
    /// compaction, and virtual geometry's retest of the camera's clusters the
    /// last frame's pyramid hid.
    pub(super) fn encode_occlusion_test(
        &mut self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let virtual_frame = state.virtual_frame;
        let temporal_frame = &state.packing.temporal_frame;
        // One span from the pyramid's reduction to the test's compaction or
        // the virtual retest.
        let span = self.profiler.span("occlusion");
        if let Some((pyramid, generation)) =
            self.culling.pyramid.as_ref().and_then(DepthPyramid::view)
        {
            let odd = temporal_frame
                .as_ref()
                .is_some_and(|frame| frame.parity == 1);
            let poses = self.instances.poses(usize::from(odd));
            self.culling
                .pyramid
                .as_ref()
                .expect("viewed above")
                .encode(encoder, span);
            // The camera's clusters the last frame's pyramid hid
            // are retested against this frame's within
            // the same span.
            self.gpu_presentation.encode_occlusion(
                &self.device,
                encoder,
                (pyramid, generation),
                poses,
                span.filter(|_| !virtual_frame),
            );
            if virtual_frame {
                self.virtual_draws.pass.encode_retest(&self.device, encoder);
                if let Some(span) = span {
                    span.mark(encoder, true);
                }
            }
        }
    }

    /// The opaque meshes of `phase` (GPU-presented frames draw each level of
    /// detail's visible slots), the impostors' cards,
    /// and virtual geometry's bins the visibility buffer leaves to
    /// forward drawing; returns the draws.
    fn draw_opaque(
        &self,
        state: &FrameState<'_>,
        pass: &mut wgpu::RenderPass<'_>,
        phase: usize,
        virtual_first: Option<usize>,
    ) -> u32 {
        let gpu_mode = state.kind.gpu_mode;
        let Passes {
            opaque_kind,
            impostor_pass,
            ..
        } = state.passes;
        let temporal_pass = state
            .packing
            .temporal_frame
            .as_ref()
            .zip(self.temporal.as_ref());
        // Lit frames shade with the lit pipelines and the lights' group.
        let lit = state
            .features
            .light_frame
            .as_ref()
            .zip(self.lights.as_ref());
        pass.set_bind_group(0, &self.camera.group, &[]);
        if let Some((_, lights)) = lit {
            pass.set_bind_group(3, lights.fragment_group(), &[]);
        }
        pass.set_bind_group(
            2,
            temporal_pass.map_or_else(
                || self.instances.pose_group(0),
                |(frame, temporal)| temporal.pose_group(frame),
            ),
            &[],
        );
        if !gpu_mode {
            pass.set_vertex_buffer(1, self.instances.slots().slice(..));
        }
        let mut draws = 0;
        let mut binds = BatchBinds::default();
        for &index in &self.draw_order.order {
            let index = index as usize;
            let batch = &self.plan.batches[index];
            // Batches drawing virtual geometry draw its
            // clusters below.
            if self.virtual_draws.drawn.get(index) == Some(&true) {
                continue;
            }
            let (mesh, draw) = self.bind_batch(pass, &mut binds, (opaque_kind, false), batch);
            if gpu_mode {
                // Visible slots of this group, compacted and counted
                // on the GPU, each level of detail
                // with its own.
                for level in 0..lod_levels(batch, mesh, false) {
                    // An empty level draws nothing.
                    if level > 0 && mesh.lods[level - 1].1 == 0 {
                        continue;
                    }
                    let (ids, args, offset) = self
                        .gpu_presentation
                        .draw_source(batch, index, phase, level);
                    binds.bindings.ids(pass, 1, ids);
                    let timed = self
                        .profiler
                        .begin_draw(pass, || draw_label(batch, phase, Some(level)));
                    pass.draw_indexed_indirect(args, offset);
                    self.profiler.end_draw(pass, timed);
                    draws += 1;
                }
            } else {
                let timed = self
                    .profiler
                    .begin_draw(pass, || draw_label(batch, phase, None));
                pass.draw_indexed(
                    draw.index_range(),
                    draw.base_vertex,
                    batch.instances.clone(),
                );
                self.profiler.end_draw(pass, timed);
                draws += 1;
            }
        }
        // Impostors: every draw group's cards in one
        // draw, over the pose and light groups the meshes bound.
        if gpu_mode
            && let Some((targets, lit)) = impostor_pass
            && let (Some(pipeline), Some(group)) = (
                self.impostors.pipelines.pipeline(targets, lit),
                self.impostors.atlas.group(),
            )
        {
            let (ids, args, offset) = self.gpu_presentation.impostor_source(phase);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, group, &[]);
            pass.set_vertex_buffer(0, ids.slice(..));
            pass.draw_indirect(args, offset);
            draws += 1;
            // Cards fading over their meshes, once.
            if phase == 0 {
                let (cards, args, offset) = self.gpu_presentation.fading_cards();
                pass.set_vertex_buffer(0, cards);
                pass.draw_indirect(args, offset);
                draws += 1;
            }
        }
        // Virtual geometry: each bin's selected
        // clusters, in phase one, then each bin's twin of clusters
        // the retest found visible in phase two; bins the visibility
        // buffer resolves draw below.
        if let Some(first_bin) = virtual_first {
            let groups = self.virtual_draws.pose_groups.get().expect("built above");
            let group = match temporal_pass {
                Some((frame, _)) => {
                    let previous = if frame.motion_valid {
                        1 - frame.parity
                    } else {
                        frame.parity
                    };
                    &groups[2 * frame.parity + previous]
                }
                None => &groups[0],
            };
            pass.set_bind_group(2, group, &[]);
            for (bin, (material, variant)) in self.virtual_draws.bins.iter().enumerate() {
                if self.visibility.bins.get(bin) == Some(&true) {
                    continue;
                }
                pass.set_pipeline(&self.virtual_draws.pipelines[&(opaque_kind, *variant)]);
                pass.set_bind_group(1, self.materials.binding(*material).0, &[]);
                let (stream, args, offset) = self.virtual_draws.pass.bin_source(first_bin + bin);
                pass.set_vertex_buffer(0, stream);
                pass.draw_indirect(args, offset);
                draws += 1;
            }
        }
        draws
    }

    /// The triangle IDs of `phase`'s camera bins the visibility buffer
    /// resolves.
    pub(super) fn encode_visibility_ids(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
        phase: usize,
    ) {
        let pipelines = self.visibility.pipelines.as_ref().expect("created above");
        let (pass_groups, _) = self.visibility.groups.get().expect("built above");
        let bins = self.virtual_draws.bins.len();
        let first_bin = match phase {
            0 => 0,
            _ => self
                .virtual_draws
                .phase_two
                .expect("phase two resolves its bins' twins"),
        };
        let (current, _) = self.pose_parities(state);
        let (colors, count) = self.color_attachments(node, view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/visibility"),
            color_attachments: &colors[..count],
            depth_stencil_attachment: self.depth_attachment(node),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &self.camera.group, &[]);
        pass.set_bind_group(1, &self.mesh_pipelines.empty_group, &[]);
        for (bin, (_, variant)) in self.virtual_draws.bins.iter().enumerate() {
            if !self.visibility.bins[bin] {
                continue;
            }
            pass.set_pipeline(&pipelines[usize::from(variant.double_sided())]);
            pass.set_bind_group(
                2,
                &pass_groups[current],
                &[VisibilityBuffer::draw_offset(phase, bins, bin)],
            );
            let (stream, args, offset) = self.virtual_draws.pass.bin_source(first_bin + bin);
            pass.set_vertex_buffer(0, stream);
            pass.draw_indirect(args, offset);
        }
    }

    /// The tiles of each bin the visibility buffer resolves.
    pub(super) fn encode_visibility_tiles(&self, encoder: &mut wgpu::CommandEncoder) {
        self.visibility
            .buffer
            .as_ref()
            .expect("prepared above")
            .encode_classify(encoder, self.virtual_draws.bins.len());
    }

    /// Each resolved bin's pixels of `phase` shaded into the opaque targets,
    /// and the sky after the last phase.
    pub(super) fn encode_visibility_resolve(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
        phase: usize,
    ) {
        let phases = usize::from(state.shape.phases);
        let opaque_kind = state.passes.opaque_kind;
        let lit = state
            .features
            .light_frame
            .as_ref()
            .zip(self.lights.as_ref());
        let visibility = self.visibility.buffer.as_ref().expect("prepared above");
        let (_, resolve_groups) = self.visibility.groups.get().expect("built above");
        let bins = self.virtual_draws.bins.len();
        let (current, previous) = self.pose_parities(state);
        // The last resolve ends the opaque span.
        let timestamp_writes = (phase + 1 == phases)
            .then(|| self.profiler.span("opaque_meshes"))
            .flatten()
            .map(|span| span.render_part(false, true));
        let (colors, count) = self.color_attachments(node, view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/visibility-resolve"),
            color_attachments: &colors[..count],
            // Read-only: the resolves test no depth and the sky tests it.
            depth_stencil_attachment: self.depth_attachment(node),
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &self.camera.group, &[]);
        if let Some((_, lights)) = lit {
            pass.set_bind_group(3, lights.fragment_group(), &[]);
        }
        let (_, _, resolve_args) = visibility.buffers();
        for (bin, (material, variant)) in self.virtual_draws.bins.iter().enumerate() {
            if !self.visibility.bins[bin] {
                continue;
            }
            pass.set_pipeline(&self.visibility.resolve_pipelines[&(opaque_kind, *variant)]);
            pass.set_bind_group(1, self.materials.binding(*material).0, &[]);
            pass.set_bind_group(
                2,
                &resolve_groups[2 * current + previous],
                &[VisibilityBuffer::draw_offset(phase, bins, bin)],
            );
            pass.draw_indirect(resolve_args, bin as u64 * 16);
        }
        if phase + 1 == phases {
            pass.set_pipeline(self.sky_pipeline(state));
            pass.set_bind_group(0, &self.camera.group, &[]);
            pass.set_bind_group(1, self.clouds.sky_group(), &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// What profiling reports name a draw of `batch` by: its mesh, material,
/// opaque phase and level of detail (`None` for exact frames' draws of
/// every instance).
fn draw_label(batch: &DrawBatch, phase: usize, level: Option<usize>) -> String {
    let material = batch
        .material
        .map_or_else(|| "none".to_owned(), |material| material.slot().to_string());
    let mesh = batch.mesh.slot();
    match level {
        Some(level) => format!("mesh {mesh} material {material} phase {phase} level {level}"),
        None => format!("mesh {mesh} material {material} phase {phase}"),
    }
}
