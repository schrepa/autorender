//! Impostors (decision 048): baking the frame's missing ones and preparing
//! their records and draw pipeline.

use super::*;

/// Impostors (decision 048): the plan batches' baked layers and placed
/// objects' (decision 060), the pipelines both share, this frame's record
/// per plan batch (zeroed without an impostor), and the draw groups a
/// budget or the GPU left without one.
pub(super) struct ImpostorDraws {
    pub(super) atlas: ImpostorAtlas,
    pub(super) placed: ImpostorAtlas,
    pub(super) pipelines: ImpostorPipelines,
    pub(super) records: Vec<ImpostorRecord>,
    pub(super) dropped: u32,
}

impl ImpostorDraws {
    /// Impostors baked in both atlases.
    pub(super) fn live(&self) -> u32 {
        self.atlas.live() + self.placed.live()
    }

    /// GPU bytes of both atlases.
    pub(super) fn capacity_bytes(&self) -> u64 {
        self.atlas.capacity_bytes() + self.placed.capacity_bytes()
    }
}

impl Renderer {
    /// Impostors (decision 048) of this frame: drops those of retired or
    /// replaced meshes and materials, bakes the missing ones of opaque draw
    /// groups whose meshes have enough triangles into `encoder` (after the
    /// materials' levels) on GPU-presented frames with them, and fills the
    /// record of every plan batch, zeroed without one.
    pub(super) fn prepare_impostors(
        &mut self,
        snapshot: &RenderSnapshot,
        gpu_mode: bool,
        encoder: &mut Option<wgpu::CommandEncoder>,
    ) {
        // Records follow the plan, mesh residency, material groups, the
        // settings and the atlas.
        let settings = snapshot.environment.level_of_detail;
        let key = |renderer: &Self| {
            (
                renderer.structure_key(),
                settings,
                gpu_mode,
                renderer.impostors.atlas.generation(),
            )
        };
        if self.structure_keys.impostors == Some(key(self)) {
            return;
        }
        self.prepare_impostor_records(snapshot, gpu_mode, encoder);
        self.structure_keys.impostors = Some(key(self));
    }

    fn prepare_impostor_records(
        &mut self,
        snapshot: &RenderSnapshot,
        gpu_mode: bool,
        encoder: &mut Option<wgpu::CommandEncoder>,
    ) {
        self.impostors.records.clear();
        self.impostors.dropped = 0;
        let (meshes, materials, plan) = (&self.meshes, &self.materials, &self.plan);
        self.impostors.atlas.retain(
            |(mesh, material)| {
                meshes.contains_key(mesh)
                    && material.is_none_or(|handle| materials.group(handle).is_some())
            },
            |handle| plan.material(snapshot, handle).copied(),
        );
        let settings = snapshot.environment.level_of_detail;
        if !(gpu_mode && settings.enabled && settings.impostors) {
            return;
        }
        self.impostors.atlas.enforce_budget(settings.max_impostors);
        let device = &self.device;
        for batch in &self.plan.batches {
            let mesh = &self.meshes[&batch.mesh];
            let source = &snapshot.meshes[batch.source_index].mesh;
            let (group, variant) = self.materials.binding(batch.material);
            // Virtual geometry (decision 044) bakes its fallback cut: its
            // camera clusters select only instances not drawn as impostors.
            let eligible = !batch.transparent
                && !mesh.skinned
                && !variant.transmission()
                && source.indices().len() / 3 >= IMPOSTOR_MIN_TRIANGLES;
            let mut record = ImpostorRecord::default();
            if eligible {
                let material = batch
                    .material
                    .and_then(|handle| self.plan.material(snapshot, handle))
                    .copied();
                let key = (batch.mesh, batch.material);
                if let Some(baked) = self.impostors.atlas.record(key, material.as_ref()) {
                    record = baked;
                } else {
                    let encoder = encoder.get_or_insert_with(|| {
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("engine/texture-mips"),
                        })
                    });
                    let bounds = source.bounds();
                    let source = BakeSource {
                        draw: mesh.draw(&self.mesh_arena),
                        extended: mesh.extended,
                        group,
                        variant,
                        bounds: (bounds.min(), bounds.max()),
                        material,
                    };
                    if self.impostors.atlas.bake(
                        device,
                        encoder,
                        self.materials.mips(),
                        &self.mesh_shader,
                        &mut self.impostors.pipelines,
                        key,
                        &source,
                        settings.max_impostors,
                    ) {
                        record = self
                            .impostors
                            .atlas
                            .record(key, material.as_ref())
                            .unwrap_or_default();
                    } else {
                        self.impostors.dropped += 1;
                    }
                }
            }
            self.impostors.records.push(record);
        }
    }

    /// Uploads this frame's impostor records (decision 048), the plan
    /// batches' and placed draw groups' (decision 060) to their atlases, and
    /// creates the pipeline drawing them in the opaque pass of `kind`;
    /// returns its key, `None` when impostors cannot draw.
    pub(super) fn prepare_impostor_draws(&mut self, kind: PassKind) -> Option<(Targets, bool)> {
        let targets = kind.targets()?;
        let lit = kind.lit;
        let impostors = &mut self.impostors;
        let batches = impostors.atlas.prepare(
            &self.device,
            &self.queue,
            &impostors.pipelines,
            &impostors.records,
        );
        let placed = impostors.placed.prepare(
            &self.device,
            &self.queue,
            &impostors.pipelines,
            self.scatter_draws.impostors(),
        );
        if !(batches || placed) {
            return None;
        }
        if self.impostors.pipelines.pipeline(targets, lit).is_none() {
            let pose_layout = if targets == Targets::Temporal {
                self.temporal.as_ref()?.pose_layout()
            } else {
                &self.instances.pose_layout
            };
            let mut layouts = vec![
                Some(&self.camera.layout),
                Some(self.impostors.pipelines.draw_layout()),
                Some(pose_layout),
            ];
            if lit {
                layouts.push(Some(self.lights.as_ref()?.fragment_layout()));
            }
            let layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("engine/impostor-pipeline-layout"),
                    bind_group_layouts: &layouts,
                    immediate_size: 0,
                });
            let pipeline = crate::impostors::draw_pipeline(
                &self.device,
                &self.mesh_shader,
                &layout,
                targets,
                lit,
                self.display.format(),
                DEPTH_FORMAT,
            );
            self.impostors
                .pipelines
                .insert_pipeline(targets, lit, pipeline);
        }
        Some((targets, lit))
    }
}
