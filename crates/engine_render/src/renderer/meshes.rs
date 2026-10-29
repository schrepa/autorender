use super::*;
use crate::bindings::EMPTY;
use crate::entries::Slots;
use crate::pipeline_layouts::{
    LIT_MESH_PIPELINES, LIT_TEMPORAL_MESH_PIPELINES, MESH_PIPELINES, TEMPORAL_MESH_PIPELINES,
};
use crate::pipelines::vertex_attributes;

pub(crate) struct MeshPipelines {
    pub(crate) variants: HashMap<(PassKind, Variant), wgpu::RenderPipeline>,
    pub(crate) empty_layout: crate::pipelines::GroupLayout,
    pub(crate) empty_group: wgpu::BindGroup,
    pub(crate) shading: SceneShading,
    pub(crate) cooking: bool,
}

impl MeshPipelines {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let empty_layout = crate::pipelines::layout(device, &EMPTY);
        let empty_group =
            crate::pipelines::group(device, "engine/empty-group", &EMPTY, &empty_layout, &[]);
        Self {
            variants: HashMap::default(),
            empty_layout,
            empty_group,
            shading: SceneShading::ALL,
            cooking: false,
        }
    }
}

pub(crate) fn relit(kind: PassKind) -> PassKind {
    PassKind {
        lit: !kind.lit,
        ..kind
    }
}

#[derive(Default)]
pub(crate) struct DrawOrder {
    pub(crate) order: Vec<u32>,
    pub(crate) masked: Vec<u32>,
    pub(crate) keys: Vec<(u8, Variant, Option<MaterialHandle>, u8, u32)>,
}

const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] = vertex_attributes!(Vertex {
    0 => position as Float32x3,
    1 => normal as Snorm16x2,
    7 => uv as Float32x2,
});
const SLOT_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![2 => Uint32];
const EXTRA_ATTRIBUTES: [wgpu::VertexAttribute; 2] = vertex_attributes!(ExtraVertex {
    9 => color as Unorm16x4,
    10 => uv1 as Float32x2,
});
pub(crate) fn extra_layout() -> Option<wgpu::VertexBufferLayout<'static>> {
    Some(wgpu::VertexBufferLayout {
        array_stride: size_of::<ExtraVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &EXTRA_ATTRIBUTES,
    })
}

pub(crate) fn vertex_layout() -> Option<wgpu::VertexBufferLayout<'static>> {
    Some(wgpu::VertexBufferLayout {
        array_stride: size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &VERTEX_ATTRIBUTES,
    })
}

pub(crate) fn slot_layouts() -> [Option<wgpu::VertexBufferLayout<'static>>; 2] {
    [
        vertex_layout(),
        Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<u32>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &SLOT_ATTRIBUTES,
        }),
    ]
}

#[derive(Default)]
pub(crate) struct BatchBinds {
    pub(crate) bindings: Bindings,
    variant: Option<Variant>,
}

impl Core {
    pub(crate) fn batch_binding(
        &self,
        batch: &DrawBatch,
        mesh: Option<&GpuMesh>,
    ) -> (&wgpu::BindGroup, Variant) {
        let (group, variant) = self.materials.binding(batch.material);
        let variant = variant.with_extended(mesh.is_some_and(|mesh| mesh.extended));
        (group, variant)
    }

    pub(crate) fn bind_batch(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        binds: &mut BatchBinds,
        (kind, uber): (PassKind, bool),
        batch: &DrawBatch,
    ) -> (&GpuMesh, MeshDraw<'_>) {
        let mesh = &self.meshes[&batch.mesh];
        let (group, variant) = self.batch_binding(batch, Some(mesh));
        let variant = if uber {
            variant.uber()
        } else if kind.targets().is_some() {
            variant.opaque()
        } else {
            variant
        };
        (
            mesh,
            self.bind_mesh(pass, binds, (kind, variant), (group, mesh)),
        )
    }

    pub(crate) fn bind_prepass(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        binds: &mut BatchBinds,
        kind: PassKind,
        batch: &DrawBatch,
    ) -> Option<(&GpuMesh, MeshDraw<'_>)> {
        let mesh = &self.meshes[&batch.mesh];
        let (group, variant) = self.batch_binding(batch, Some(mesh));
        let key = prepass_key(kind, variant)?;
        Some((mesh, self.bind_mesh(pass, binds, key, (group, mesh))))
    }

    fn bind_mesh<'m>(
        &'m self,
        pass: &mut wgpu::RenderPass<'_>,
        binds: &mut BatchBinds,
        key: (PassKind, Variant),
        (group, mesh): (&wgpu::BindGroup, &'m GpuMesh),
    ) -> MeshDraw<'m> {
        if binds.variant != Some(key.1) {
            pass.set_pipeline(&self.mesh_pipelines.variants[&key]);
            binds.variant = Some(key.1);
        }
        let bindings = &mut binds.bindings;
        bindings.group(pass, 1, group);
        let draw = mesh.draw(&self.mesh_arena);
        bindings.vertex(pass, 0, draw.vertices);
        if let Some(extras) = draw.extras.filter(|_| mesh.extended) {
            bindings.vertex(pass, 2, extras);
        }
        bindings.index(pass, draw.indices);
        draw
    }

    pub(crate) fn opaque_pipeline(
        &self,
        kind: PassKind,
        variant: Variant,
        prepass: bool,
    ) -> Option<&wgpu::RenderPipeline> {
        let key = if prepass {
            prepass_key(kind, variant)?
        } else {
            (kind, variant.opaque())
        };
        self.mesh_pipelines.variants.get(&key)
    }

    pub(crate) fn ensure_opaque_variant(&mut self, kind: PassKind, variant: Variant) {
        for key in opaque_keys(kind, variant).into_iter().flatten() {
            self.ensure_variant(key);
        }
    }

    fn ensure_variant(&mut self, (kind, variant): (PassKind, Variant)) {
        if self.mesh_pipelines.variants.contains_key(&(kind, variant)) {
            return;
        }
        let pipeline = self.create_variant(kind, variant);
        self.mesh_pipelines
            .variants
            .insert((kind, variant), pipeline);
    }

    pub(crate) fn ensure_variants(&mut self, opaque: PassKind, relights: bool) {
        if relights && opaque.targets() != Some(Targets::Debug) {
            self.create_variants(relit(opaque));
        }
        self.create_variants(opaque);
    }

    fn create_variants(&mut self, opaque: PassKind) {
        let key = (self.structure_key(), opaque);
        if self.structure_keys.variants != Some(key) {
            let mut missing = Vec::new();
            for batch in &self.plan.batches {
                let (_, variant) = self.batch_binding(batch, self.meshes.get(&batch.mesh));
                for pipeline in opaque_keys(opaque, variant).into_iter().flatten() {
                    if !self.mesh_pipelines.variants.contains_key(&pipeline)
                        && !missing.contains(&pipeline)
                    {
                        missing.push(pipeline);
                    }
                }
            }
            for pipeline in missing {
                self.ensure_variant(pipeline);
            }
            self.structure_keys.variants = Some(key);
        }
    }

    pub(crate) fn ensure_probe_variants(&mut self, kind: PassKind, relights: bool) {
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
            let pipeline = self.create_variant(kind, variant);
            self.mesh_pipelines
                .variants
                .insert((kind, variant), pipeline);
        }
    }

    fn mesh_layout(&self, kind: PassKind) -> wgpu::PipelineLayout {
        let declared = match (kind.slots(), kind.lit) {
            (Slots::Temporal, true) => &LIT_TEMPORAL_MESH_PIPELINES,
            (Slots::Temporal, false) => &TEMPORAL_MESH_PIPELINES,
            (Slots::Retained, true) => &LIT_MESH_PIPELINES,
            (Slots::Retained, false) => &MESH_PIPELINES,
        };
        crate::pipelines::layout_of(&self.device, declared)
    }

    pub(crate) fn create_variant(&self, kind: PassKind, variant: Variant) -> wgpu::RenderPipeline {
        let canvas = self.display.format();
        let layout = self.mesh_layout(kind);
        let fragment = match kind.pass {
            MeshPass::Opaque(targets) => MeshFragment {
                targets: if variant.depth_only() {
                    unwritten(targets.color_targets(canvas))
                } else {
                    targets.color_targets(canvas)
                },
                depth_compare: if variant.pretested() {
                    wgpu::CompareFunction::Equal
                } else {
                    wgpu::CompareFunction::Greater
                },
                depth_write: !variant.pretested(),
                mirrored: false,
            },
            MeshPass::Probe => MeshFragment {
                targets: vec![Some(ENVIRONMENT_FORMAT.into())],
                depth_compare: wgpu::CompareFunction::Greater,
                depth_write: true,
                mirrored: true,
            },
        };
        mesh_pipeline(
            &self.device,
            &layout,
            kind,
            &fragment,
            (variant, self.mesh_pipelines.shading),
        )
    }

    pub(crate) fn order_draws(&mut self, snapshot: &engine_core::RenderSnapshot) {
        let key = self.structure_key();
        if self.structure_keys.order == Some(key) {
            return;
        }
        self.structure_keys.order = Some(key);
        let mut keys = std::mem::take(&mut self.draw_order.keys);
        keys.clear();
        keys.extend(self.plan.batches.iter().enumerate().map(|(index, batch)| {
            let mesh = &self.meshes[&batch.mesh];
            let (_, variant) = self.batch_binding(batch, Some(mesh));
            let buffers = match &mesh.placement {
                Ok(placement) => placement.streams as u8,
                Err(_) => u8::MAX,
            };
            let flat = flat_batch(snapshot, &self.plan, batch);
            (
                u8::from(flat),
                variant,
                batch.material,
                buffers,
                index as u32,
            )
        }));
        keys.sort_unstable();
        self.draw_order.keys = keys;
        self.draw_order.order.clear();
        self.draw_order
            .order
            .extend(self.draw_order.keys.iter().map(|&(.., index)| index));
        self.draw_order.masked.clear();
        self.draw_order.masked.extend(
            self.draw_order
                .keys
                .iter()
                .filter(|key| key.1.alpha_mask())
                .map(|&(.., index)| index),
        );
    }
}

fn flat_batch(
    snapshot: &engine_core::RenderSnapshot,
    plan: &crate::batches::BatchPlan,
    batch: &crate::batches::DrawBatch,
) -> bool {
    let Some(instance) = plan
        .ordered_instances
        .get(batch.instances.start as usize)
        .and_then(|&slot| snapshot.instances.get(slot))
    else {
        return false;
    };
    let bounds = snapshot.meshes[batch.source_index].mesh.bounds();
    let size = bounds.max() - bounds.min();
    let model = instance.model;
    let extent = |row: usize| {
        model.x_axis[row].abs() * size.x
            + model.y_axis[row].abs() * size.y
            + model.z_axis[row].abs() * size.z
    };
    extent(1) < 0.1 * engine_core::math::max(extent(0), extent(2))
}

struct MeshFragment {
    targets: Vec<Option<wgpu::ColorTargetState>>,
    depth_compare: wgpu::CompareFunction,
    depth_write: bool,
    mirrored: bool,
}

fn unwritten(targets: Vec<Option<wgpu::ColorTargetState>>) -> Vec<Option<wgpu::ColorTargetState>> {
    targets
        .into_iter()
        .map(|target| {
            target.map(|target| wgpu::ColorTargetState {
                blend: None,
                write_mask: wgpu::ColorWrites::empty(),
                ..target
            })
        })
        .collect()
}

fn opaque_keys(kind: PassKind, variant: Variant) -> [Option<(PassKind, Variant)>; 2] {
    [Some((kind, variant.opaque())), prepass_key(kind, variant)]
}

fn prepass_key(kind: PassKind, variant: Variant) -> Option<(PassKind, Variant)> {
    variant
        .alpha_mask()
        .then_some((PassKind { lit: false, ..kind }, variant.prepass()))
}

fn mesh_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    kind: PassKind,
    fragment: &MeshFragment,
    (variant, shading): (Variant, SceneShading),
) -> wgpu::RenderPipeline {
    let constants = shading.with(&variant.constants());
    let extended = variant.extended();
    let streams = if extended {
        Streams::Extended
    } else {
        Streams::Basic
    };
    let vertex_entry = kind.vertex_entry(streams);
    let fragment_entry = if variant.depth_only() {
        kind.prepass_entry(extended)
    } else {
        kind.fragment_entry(extended)
    };
    let [mesh, slot_ids] = slot_layouts();
    let layouts = [mesh, slot_ids, extra_layout()];
    let buffers = match streams {
        Streams::Extended => 3,
        Streams::Basic => 2,
    };
    let buffers = if kind.targets() == Some(Targets::Debug) {
        crate::debug_view::with_record(&layouts[..buffers])
    } else {
        layouts[..buffers].to_vec()
    };
    let label = format!(
        "engine/mesh {} {} {:#06x}",
        vertex_entry.name,
        fragment_entry.name,
        variant.bits()
    );
    crate::pipelines::geometry(
        device,
        &crate::pipelines::Geometry {
            label: &label,
            layout,
            vertex_constants: &constants,
            vertex: vertex_entry,
            buffers: &buffers,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: if fragment.mirrored {
                    wgpu::FrontFace::Cw
                } else {
                    wgpu::FrontFace::Ccw
                },
                cull_mode: (!variant.double_sided()).then_some(wgpu::Face::Back),
                ..Default::default()
            },
            depth: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(fragment.depth_write),
                depth_compare: Some(fragment.depth_compare),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            fragment: Some(crate::pipelines::Fragment {
                entry: fragment_entry,
                constants: &constants,
                targets: &fragment.targets,
            }),
        },
    )
}
