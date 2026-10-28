#![expect(
    clippy::expect_used,
    reason = "the opaque passes read what the frame built"
)]

use super::*;
use crate::debug_view::{DrawKind, DrawRecord, bind_record};

#[derive(Clone, Copy)]
enum Opaque {
    Prepass,
    Meshes,
    Terrain,
    Cover,
    Scatter,
    Sky,
}

#[derive(Clone, Copy)]
enum Slot {
    Early,
    Terrain,
    Late,
    Last,
}

const OPAQUE: [(Opaque, Slot); 6] = [
    (Opaque::Prepass, Slot::Early),
    (Opaque::Meshes, Slot::Early),
    (Opaque::Terrain, Slot::Terrain),
    (Opaque::Cover, Slot::Late),
    (Opaque::Scatter, Slot::Late),
    (Opaque::Sky, Slot::Last),
];

impl Renderer {
    fn sky_pipeline(&self, state: &FrameState<'_>) -> &wgpu::RenderPipeline {
        self.mesh_pipelines
            .sky(SkyTargets::Frame(state.shape.targets))
    }

    fn pose_parities(&self, state: &FrameState<'_>) -> (usize, usize) {
        match state
            .packing
            .temporal_frame
            .as_ref()
            .zip(self.features.temporal.get())
        {
            Some((frame, _)) if frame.motion_valid => (frame.parity, 1 - frame.parity),
            Some((frame, _)) => (frame.parity, frame.parity),
            None => (0, 0),
        }
    }

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
        let virtual_first = match phase {
            0 => Some(0),
            _ => self.virtual_draws.phase_two,
        }
        .filter(|_| state.virtual_frame);
        let forward = phase > 0 || !state.shape.deferred;
        let (begins, ends) = (phase == 0, forward && phase + 1 == phases && !resolve_phase);
        let timestamp_writes = (begins || ends)
            .then(|| self.profiler.span("opaque_meshes"))
            .flatten()
            .map(|span| span.render_part(begins, ends));
        let mut pass = self.render_pass(
            encoder,
            node,
            view,
            "engine/opaque-meshes",
            timestamp_writes,
        );
        let (first, last) = (phase == 0, phase + 1 == phases && !resolve_phase);
        self.draw_contributors(
            state,
            &mut pass,
            (phase, virtual_first),
            |slot| match slot {
                Slot::Early => true,
                Slot::Terrain | Slot::Late => forward && first,
                Slot::Last => forward && last,
            },
        )
    }

    fn draw_contributors(
        &self,
        state: &FrameState<'_>,
        pass: &mut wgpu::RenderPass<'_>,
        (phase, virtual_first): (usize, Option<usize>),
        drawn: impl Fn(Slot) -> bool,
    ) -> u32 {
        let mut draws = 0;
        for (contributor, slot) in OPAQUE {
            if !drawn(slot) {
                continue;
            }
            draws += match contributor {
                Opaque::Prepass => self.draw_prepass(state, pass, phase),
                Opaque::Meshes if state.kind.instance_count > 0 => {
                    self.draw_opaque(state, pass, phase, virtual_first)
                }
                Opaque::Meshes => 0,
                Opaque::Terrain => self.draw_terrain_patches(state, pass),
                Opaque::Cover => self.draw_terrain_cover(state, pass),
                Opaque::Scatter => self.draw_scatter(state, pass, false),
                Opaque::Sky => {
                    self.draw_sky(state, pass);
                    0
                }
            };
        }
        draws
    }

    pub(super) fn encode_terrain_ids(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
    ) -> u32 {
        let mut pass = self.render_pass(encoder, node, view, "engine/terrain-ids", None);
        self.draw_terrain_ids(&mut pass)
    }

    pub(super) fn encode_terrain_resolve(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
    ) -> u32 {
        let mut pass = self.render_pass(encoder, node, view, "engine/terrain-resolve", None);
        self.draw_terrain_resolve(state, &mut pass)
    }

    pub(super) fn encode_opaque_late(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
    ) -> u32 {
        let ends = state.shape.phases == 1 && !state.shape.resolves[0];
        let timestamp_writes = ends
            .then(|| self.profiler.span("opaque_meshes"))
            .flatten()
            .map(|span| span.render_part(false, true));
        let mut pass = self.render_pass(
            encoder,
            node,
            view,
            "engine/opaque-meshes",
            timestamp_writes,
        );
        self.draw_contributors(state, &mut pass, (0, None), |slot| match slot {
            Slot::Late => true,
            Slot::Last => ends,
            Slot::Early | Slot::Terrain => false,
        })
    }

    fn draw_sky(&self, state: &FrameState<'_>, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(self.sky_pipeline(state));
        pass.set_bind_group(0, self.camera.group(), &[]);
        pass.set_bind_group(1, self.features.sky.get().pass_group(), &[]);
        pass.draw(0..3, 0..1);
    }

    pub(super) fn encode_occlusion_test(
        &mut self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let virtual_frame = state.virtual_frame;
        let temporal_frame = &state.packing.temporal_frame;
        let span = self.profiler.span("occlusion");
        if let Some(pyramid) = self.culling.pyramid.as_ref().and_then(DepthPyramid::view) {
            let odd = temporal_frame
                .as_ref()
                .is_some_and(|frame| frame.parity == 1);
            let poses = self.instances.poses(usize::from(odd));
            self.culling
                .pyramid
                .as_ref()
                .expect("viewed above")
                .encode(encoder, span);
            self.gpu_presentation.encode_occlusion(
                &self.device,
                encoder,
                (pyramid, pyramid.generation()),
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

    fn draw_prepass(
        &self,
        state: &FrameState<'_>,
        pass: &mut wgpu::RenderPass<'_>,
        phase: usize,
    ) -> u32 {
        let meshes = if state.kind.instance_count > 0 && !self.draw_order.masked.is_empty() {
            self.draw_meshes(state, pass, phase, true)
        } else {
            0
        };
        let placed = if phase == 0 {
            self.draw_scatter(state, pass, true)
        } else {
            0
        };
        meshes + placed
    }

    fn draw_opaque(
        &self,
        state: &FrameState<'_>,
        pass: &mut wgpu::RenderPass<'_>,
        phase: usize,
        virtual_first: Option<usize>,
    ) -> u32 {
        let mut draws = self.draw_meshes(state, pass, phase, false);
        let debug = self.debug_records(state);
        let casts = state.shape.shadows;
        if state.kind.gpu_mode
            && let Some((targets, lit)) = state.passes.impostor_pass
            && let (Some(pipeline), Some(group)) = (
                self.impostors.pipelines.pipeline(targets, lit),
                self.impostors.atlas.group(),
            )
        {
            let (ids, args, offset) = self.gpu_presentation.impostor_source(phase);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, group, &[]);
            pass.set_vertex_buffer(0, ids.slice(..));
            bind_record(
                pass,
                debug,
                DrawRecord::of(DrawKind::Card, LOD_LEVELS, casts),
            );
            pass.draw_indirect(args, offset);
            draws += 1;
            if phase == 0 {
                let (cards, args, offset) = self.gpu_presentation.fading_cards();
                pass.set_vertex_buffer(0, cards);
                pass.draw_indirect(args, offset);
                draws += 1;
            }
        }
        if let Some(first_bin) = virtual_first {
            draws += self.draw_virtual_bins(state, pass, first_bin);
        }
        draws
    }

    fn draw_meshes(
        &self,
        state: &FrameState<'_>,
        pass: &mut wgpu::RenderPass<'_>,
        phase: usize,
        prepass: bool,
    ) -> u32 {
        let gpu_mode = state.kind.gpu_mode;
        let opaque_kind = state.passes.opaque_kind;
        let temporal_pass = state
            .packing
            .temporal_frame
            .as_ref()
            .zip(self.features.temporal.get());
        let lit = state
            .features
            .light_frame()
            .zip(self.features.lights.get())
            .filter(|_| !prepass);
        pass.set_bind_group(0, self.camera.group(), &[]);
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
        let debug = self.debug_records(state);
        let casts = state.shape.shadows;
        let mut draws = 0;
        let mut binds = BatchBinds::default();
        let order = if prepass {
            &self.draw_order.masked
        } else {
            &self.draw_order.order
        };
        for &index in order {
            let index = index as usize;
            let batch = &self.plan.batches[index];
            if self.virtual_draws.drawn.get(index) == Some(&true) {
                continue;
            }
            let bound = if prepass {
                self.bind_prepass(pass, &mut binds, opaque_kind, batch)
            } else {
                Some(self.bind_batch(pass, &mut binds, (opaque_kind, false), batch))
            };
            let Some((mesh, draw)) = bound else {
                continue;
            };
            if gpu_mode {
                for level in 0..lod_levels(batch, mesh, false) {
                    if level > 0 && mesh.plan[level - 1].1 == 0 {
                        continue;
                    }
                    let (ids, args, offset) = self
                        .gpu_presentation
                        .draw_source(batch, index, phase, level);
                    binds.bindings.ids(pass, 1, ids);
                    bind_record(pass, debug, DrawRecord::of(DrawKind::Mesh, level, casts));
                    let timed = self
                        .profiler
                        .begin_draw(pass, || draw_label(batch, (phase, Some(level)), prepass));
                    pass.draw_indexed_indirect(args, offset);
                    self.profiler.end_draw(pass, timed);
                    draws += 1;
                }
            } else {
                bind_record(pass, debug, DrawRecord::of(DrawKind::Mesh, 0, casts));
                let timed = self
                    .profiler
                    .begin_draw(pass, || draw_label(batch, (phase, None), prepass));
                pass.draw_indexed(
                    draw.index_range(),
                    draw.base_vertex,
                    batch.instances.clone(),
                );
                self.profiler.end_draw(pass, timed);
                draws += 1;
            }
        }
        draws
    }

    fn draw_virtual_bins(
        &self,
        state: &FrameState<'_>,
        pass: &mut wgpu::RenderPass<'_>,
        first_bin: usize,
    ) -> u32 {
        let index = match state
            .packing
            .temporal_frame
            .as_ref()
            .zip(self.features.temporal.get())
        {
            Some((frame, _)) => {
                let previous = if frame.motion_valid {
                    1 - frame.parity
                } else {
                    frame.parity
                };
                2 * frame.parity + previous
            }
            None => 0,
        };
        let group = self.virtual_draws.pose_groups[index]
            .get()
            .expect("built above");
        pass.set_bind_group(2, group, &[]);
        let record = DrawRecord::of(DrawKind::Virtual, 0, state.shape.shadow_clusters);
        bind_record(pass, self.debug_records(state), record);
        let opaque_kind = state.passes.opaque_kind;
        let mut draws = 0;
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
        draws
    }

    pub(super) fn encode_visibility_ids(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
        phase: usize,
    ) {
        let pipelines = self.visibility.pipelines.as_ref().expect("created above");

        let bins = self.virtual_draws.bins.len();
        let first_bin = match phase {
            0 => 0,
            _ => self
                .virtual_draws
                .phase_two
                .expect("phase two resolves its bins' twins"),
        };
        let (current, _) = self.pose_parities(state);
        let mut pass = self.render_pass(encoder, node, view, "engine/visibility", None);
        pass.set_bind_group(0, self.camera.group(), &[]);
        pass.set_bind_group(1, &self.mesh_pipelines.empty_group, &[]);
        for (bin, (_, variant)) in self.virtual_draws.bins.iter().enumerate() {
            if !self.visibility.bins[bin] {
                continue;
            }
            pass.set_pipeline(&pipelines[usize::from(variant.double_sided())]);
            pass.set_bind_group(
                2,
                self.visibility.pass_groups[current]
                    .get()
                    .expect("built above"),
                &[VisibilityBuffer::draw_offset(phase, bins, bin)],
            );
            let (stream, args, offset) = self.virtual_draws.pass.bin_source(first_bin + bin);
            pass.set_vertex_buffer(0, stream);
            pass.draw_indirect(args, offset);
        }
    }

    pub(super) fn encode_visibility_tiles(&self, encoder: &mut wgpu::CommandEncoder) {
        self.visibility
            .buffer
            .as_ref()
            .expect("prepared above")
            .encode_classify(encoder, self.virtual_draws.bins.len());
    }

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
        let lit = state.features.light_frame().zip(self.features.lights.get());
        let visibility = self.visibility.buffer.as_ref().expect("prepared above");

        let bins = self.virtual_draws.bins.len();
        let (current, previous) = self.pose_parities(state);
        let timestamp_writes = (phase + 1 == phases)
            .then(|| self.profiler.span("opaque_meshes"))
            .flatten()
            .map(|span| span.render_part(false, true));
        let mut pass = self.render_pass(
            encoder,
            node,
            view,
            "engine/visibility-resolve",
            timestamp_writes,
        );
        pass.set_bind_group(0, self.camera.group(), &[]);
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
                self.visibility.resolve_groups[2 * current + previous]
                    .get()
                    .expect("built above"),
                &[VisibilityBuffer::draw_offset(phase, bins, bin)],
            );
            pass.draw_indirect(resolve_args, bin as u64 * 16);
        }
        if phase + 1 == phases {
            pass.set_pipeline(self.sky_pipeline(state));
            pass.set_bind_group(0, self.camera.group(), &[]);
            pass.set_bind_group(1, self.features.sky.get().pass_group(), &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

fn draw_label(batch: &DrawBatch, (phase, level): (usize, Option<usize>), prepass: bool) -> String {
    let material = batch
        .material
        .map_or_else(|| "none".to_owned(), |material| material.slot().to_string());
    let mesh = batch.mesh.slot();
    let stage = if prepass { " depth pre-pass" } else { "" };
    match level {
        Some(level) => {
            format!("mesh {mesh} material {material} phase {phase} level {level}{stage}")
        }
        None => format!("mesh {mesh} material {material} phase {phase}{stage}"),
    }
}
