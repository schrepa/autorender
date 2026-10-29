use super::*;
use crate::pipeline_layouts::{LIT_IMPOSTOR_PIPELINES, LIT_TEMPORAL_IMPOSTOR_PIPELINES};

pub struct ImpostorDraws {
    pub atlas: ImpostorAtlas,
    pub pipelines: ImpostorPipelines,
    pub contents: Contents,
    pub records: Vec<ImpostorRecord>,
    pub dropped: u32,
}

impl ImpostorDraws {
    pub fn live(&self) -> u32 {
        self.atlas.live()
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.atlas.capacity_bytes()
    }
}

impl Core {
    pub(crate) fn prepare_impostors(
        &mut self,
        snapshot: &RenderSnapshot,
        gpu_mode: bool,
        encoder: &mut Option<wgpu::CommandEncoder>,
    ) {
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
        self.impostors.contents.retain(
            |mesh| meshes.contains_key(&mesh),
            |texture| materials.is_texture_resident(texture),
        );
        let settings = snapshot.environment.level_of_detail;
        let Some(entries) = self.mesh_entries else {
            return;
        };
        if !(gpu_mode && settings.enabled && settings.impostors) {
            return;
        }
        self.impostors.atlas.enforce_budget(settings.max_impostors);
        for batch in &self.plan.batches {
            let mesh = &self.meshes[&batch.mesh];
            let source = &snapshot.meshes[batch.source_index].mesh;
            let (group, variant) = self.materials.binding(batch.material);
            let eligible = source.indices().len() / 3 >= IMPOSTOR_MIN_TRIANGLES;
            let mut record = ImpostorRecord::default();
            if eligible {
                let material = batch
                    .material
                    .and_then(|handle| self.plan.material(snapshot, handle))
                    .copied();
                let key = (batch.mesh, batch.material);
                record = match self.impostors.atlas.record(key, material.as_ref()) {
                    Some(resident) => resident,
                    None => {
                        let contents = &mut self.impostors.contents;
                        let content = (
                            contents.mesh(batch.mesh, source),
                            contents.material(material.as_ref(), |handle| {
                                snapshot
                                    .textures
                                    .iter()
                                    .find(|texture| texture.handle == handle)
                                    .map(|texture| &*texture.texture)
                            }),
                        );
                        let bounds = source.bounds();
                        let bake = BakeSource {
                            draw: mesh.draw(&self.mesh_arena),
                            extended: mesh.extended,
                            group,
                            variant,
                            bounds: (bounds.min(), bounds.max()),
                            material,
                            parts: Vec::new(),
                        };
                        let gpu = BakeGpu {
                            device: &self.device,
                            queue: &self.queue,
                            encoder: &mut *encoder,
                            mips: self.materials.mips(),
                            pipelines: &mut self.impostors.pipelines,
                            entries,
                        };
                        let supplied = self.impostors.atlas.supply(
                            gpu,
                            key,
                            &bake,
                            content,
                            settings.max_impostors,
                        );
                        self.impostors.dropped += u32::from(matches!(supplied, Supplied::Refused));
                        supplied.record()
                    }
                };
            }
            self.impostors.records.push(record);
        }
    }

    pub(crate) fn prepare_impostor_draws(&mut self, kind: PassKind) -> Option<Targets> {
        let targets = kind.targets()?;
        let impostors = &mut self.impostors;
        let batches = impostors.atlas.prepare(
            &self.device,
            &self.queue,
            &impostors.pipelines,
            &impostors.records,
        );
        if !batches {
            return None;
        }
        self.ensure_card_pipeline(targets)?;
        Some(targets)
    }

    pub fn ensure_card_pipeline(&mut self, targets: Targets) -> Option<&wgpu::RenderPipeline> {
        let entries = self.mesh_entries?;
        let shading = self.mesh_pipelines.shading();
        if self
            .impostors
            .pipelines
            .pipeline(targets, shading)
            .is_none()
        {
            let declared = if targets == Targets::Temporal {
                &LIT_TEMPORAL_IMPOSTOR_PIPELINES
            } else {
                &LIT_IMPOSTOR_PIPELINES
            };
            let layout = crate::pipelines::layout_of(&self.device, declared);
            let pipeline = crate::impostors::draw_pipeline(
                &self.device,
                &layout,
                (targets, entries),
                self.mesh_pipelines.constants(),
                self.display.format(),
                DEPTH_FORMAT,
            );
            self.impostors
                .pipelines
                .insert_pipeline(targets, shading, pipeline);
        }
        self.impostors.pipelines.pipeline(targets, shading)
    }

    pub fn card_pipeline(&self, targets: Targets) -> Option<&wgpu::RenderPipeline> {
        self.impostors
            .pipelines
            .pipeline(targets, self.mesh_pipelines.shading())
    }
}
