use engine_render::renderer::*;
use engine_debug_view::debug_targets::{DrawKind, DrawRecord, bind_record, debug_records};
use engine_render::parts::View;

impl Core {
    pub(crate) fn draw_prepass(
        &self,
        cx: &View<'_>,
        pass: &mut wgpu::RenderPass<'_>,
        phase: usize,
    ) -> u32 {
        if cx.frame.kind.instance_count > 0 && !self.draw_order.masked.is_empty() {
            self.draw_meshes(cx, pass, phase, true)
        } else {
            0
        }
    }

    fn draw_opaque(&self, cx: &View<'_>, pass: &mut wgpu::RenderPass<'_>, phase: usize) -> u32 {
        let mut draws = self.draw_meshes(cx, pass, phase, false);
        let debug = debug_records(cx);
        if cx.frame.kind.gpu_mode
            && let Some(targets) = cx.frame.passes.and_then(|passes| passes.impostor_pass)
            && let (Some(pipeline), Some(group)) = (
                self.impostors
                    .pipelines
                    .pipeline(targets, self.mesh_pipelines.shading()),
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
        draws
    }

    pub(crate) fn draw_meshes(
        &self,
        cx: &View<'_>,
        pass: &mut wgpu::RenderPass<'_>,
        phase: usize,
        prepass: bool,
    ) -> u32 {
        let gpu_mode = cx.frame.kind.gpu_mode;
        let Some(passes) = cx.frame.passes else {
            return 0;
        };
        if self.mesh_entries.is_none() {
            return 0;
        }
        let opaque_kind = passes.opaque_kind;
        pass.set_bind_group(0, self.shared_group("frame", &["refraction"]), &[]);
        if let Some(lights) = self.camera.built("lights", &[]) {
            pass.set_bind_group(3, lights, &[]);
        }
        pass.set_bind_group(
            2,
            cx.frame
                .temporal
                .as_ref()
                .and_then(|frame| self.temporal_pose_group(frame))
                .unwrap_or_else(|| self.instances.pose_group(0)),
            &[],
        );
        if !gpu_mode {
            pass.set_vertex_buffer(1, self.instances.slots().slice(..));
        }
        let debug = debug_records(cx);
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
            let bound = if prepass {
                self.bind_prepass(pass, &mut binds, opaque_kind, batch)
            } else {
                self.bind_batch(pass, &mut binds, (opaque_kind, false), batch)
            };
            let Some((mesh, draw)) = bound else {
                continue;
            };
            if gpu_mode {
                for level in 0..lod_levels(mesh) {
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
