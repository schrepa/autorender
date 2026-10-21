//! Impostors: supplying the frame's missing ones, from their records or
//! baked, and preparing their records and draw pipeline; and the records
//! hosts exchange with the renderer.

use super::*;

/// Impostors: the plan batches' layers and placed objects', the pipelines
/// both share, the contents their records are keyed by, this frame's
/// record per plan batch (zeroed without an impostor), and the draw groups
/// a budget or the GPU left without one.
pub(super) struct ImpostorDraws {
    pub(super) atlas: ImpostorAtlas,
    pub(super) placed: ImpostorAtlas,
    pub(super) pipelines: ImpostorPipelines,
    pub(super) contents: Contents,
    pub(super) records: Vec<ImpostorRecord>,
    pub(super) dropped: u32,
}

impl ImpostorDraws {
    /// Impostors resident in both atlases.
    pub(super) fn live(&self) -> u32 {
        self.atlas.live() + self.placed.live()
    }

    /// GPU bytes of both atlases.
    pub(super) fn capacity_bytes(&self) -> u64 {
        self.atlas.capacity_bytes() + self.placed.capacity_bytes()
    }
}

impl Renderer {
    /// Records the renderer asks hosts for since the last call: the layers
    /// of impostors it would otherwise bake. Hosts answer each
    /// ([`Self::give_record`]).
    pub fn take_record_wants(&mut self) -> Vec<RecordKey> {
        let mut wants = Vec::new();
        self.impostors.atlas.take_wants(&mut wants);
        self.impostors.placed.take_wants(&mut wants);
        wants
    }

    /// Hosts' answer for record `record`, which the renderer asked for
    /// ([`Self::take_record_wants`]): its bytes, or `None` where no pack or
    /// store holds it, which bakes what it holds.
    pub fn give_record(&mut self, record: RecordKey, bytes: Option<Vec<u8>>) {
        let impostors = &mut self.impostors;
        if impostors.atlas.asks(record) {
            impostors.atlas.answer(record, bytes);
        } else if impostors.placed.asks(record) {
            impostors.placed.answer(record, bytes);
        }
    }

    /// Records the renderer made since the last call, for hosts to keep:
    /// the layers of impostors it baked.
    pub fn take_made_records(&mut self) -> Vec<(RecordKey, Vec<u8>)> {
        let mut made = Vec::new();
        self.impostors.atlas.take_made(&mut made);
        self.impostors.placed.take_made(&mut made);
        made
    }

    /// Records the renderer still has to come: asked of hosts, or reading
    /// back.
    pub fn records_left(&self) -> u64 {
        self.impostors.atlas.left() + self.impostors.placed.left()
    }

    /// Impostors of this frame: drops those of retired or replaced meshes
    /// and materials, supplies the missing ones of opaque draw groups
    /// whose meshes have enough triangles on GPU-presented frames with
    /// them (baking into `encoder` after the materials' levels those no
    /// host holds), and fills the record of every plan batch, zeroed
    /// without one.
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
        self.impostors.contents.retain(
            |mesh| meshes.contains_key(&mesh),
            |texture| materials.is_texture_resident(texture),
        );
        let settings = snapshot.environment.level_of_detail;
        if !(gpu_mode && settings.enabled && settings.impostors) {
            return;
        }
        self.impostors.atlas.enforce_budget(settings.max_impostors);
        for batch in &self.plan.batches {
            let mesh = &self.meshes[&batch.mesh];
            let source = &snapshot.meshes[batch.source_index].mesh;
            let (group, variant) = self.materials.binding(batch.material);
            // Virtual geometry bakes its fallback cut: its
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
                        };
                        let gpu = BakeGpu {
                            device: &self.device,
                            queue: &self.queue,
                            encoder: &mut *encoder,
                            mips: self.materials.mips(),
                            shader: &self.mesh_shader,
                            pipelines: &mut self.impostors.pipelines,
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

    /// Uploads this frame's impostor records, the plan
    /// batches' and placed draw groups' to their atlases, and
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
            use crate::pipeline_layouts::{
                IMPOSTOR_PIPELINES, LIT_IMPOSTOR_PIPELINES, LIT_TEMPORAL_IMPOSTOR_PIPELINES,
                TEMPORAL_IMPOSTOR_PIPELINES,
            };
            let temporal = targets == Targets::Temporal;
            let pose_layout = if temporal {
                self.features.temporal.get()?.pose_layout()
            } else {
                &self.instances.pose_layout
            };
            let (camera, draw) = (&self.camera.layout, self.impostors.pipelines.draw_layout());
            let layout = if lit {
                let lights = self.features.lights.get()?.fragment_layout();
                let declared = if temporal {
                    &LIT_TEMPORAL_IMPOSTOR_PIPELINES
                } else {
                    &LIT_IMPOSTOR_PIPELINES
                };
                crate::pipelines::pipeline_layout(
                    &self.device,
                    declared,
                    &[camera, draw, pose_layout, lights],
                )
            } else {
                let declared = if temporal {
                    &TEMPORAL_IMPOSTOR_PIPELINES
                } else {
                    &IMPOSTOR_PIPELINES
                };
                crate::pipelines::pipeline_layout(
                    &self.device,
                    declared,
                    &[camera, draw, pose_layout],
                )
            };
            let pipeline = crate::impostors::draw_pipeline(
                &self.device,
                &self.mesh_shader,
                &layout,
                (targets, lit),
                self.mesh_pipelines.shading,
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
