use super::*;
use crate::bindings::EMPTY;
use crate::entries::{MeshEntries, Slots};
use crate::parts::Shading;
use crate::pipeline_layouts::{LIT_MESH_PIPELINES, LIT_TEMPORAL_MESH_PIPELINES};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ShadingSet(usize);

struct ShadingPipelines {
    constants: Vec<(&'static str, f64)>,
    variants: HashMap<(PassKind, Variant), wgpu::RenderPipeline>,
}

impl ShadingPipelines {
    fn new(constants: Vec<(&'static str, f64)>) -> Self {
        Self {
            constants,
            variants: HashMap::default(),
        }
    }
}

pub struct MeshPipelines {
    sets: Vec<ShadingPipelines>,
    current: ShadingSet,
    warm: Vec<ShadingSet>,
    view: Option<(wgpu::TextureFormat, bool)>,
    pub empty_group: wgpu::BindGroup,
    pub cooking: bool,
}

impl MeshPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let empty_layout = crate::pipelines::layout(device, &EMPTY);
        let empty_group =
            crate::pipelines::group(device, "engine/empty-group", &EMPTY, &empty_layout, &[]);
        Self {
            sets: vec![ShadingPipelines::new(Vec::new())],
            current: ShadingSet::default(),
            warm: Vec::new(),
            view: None,
            empty_group,
            cooking: false,
        }
    }

    pub fn specialize(&mut self, shading: &Shading) -> bool {
        let current = self.set_of(shading.values.clone());
        let mut warm = Vec::new();
        for &(name, value) in &shading.warm {
            let mut values = shading.values.clone();
            values.retain(|&(known, _)| known != name);
            values.push((name, value));
            let set = self.set_of(values);
            if set != current && !warm.contains(&set) {
                warm.push(set);
            }
        }
        self.warm = warm;
        let changed = current != self.current;
        self.current = current;
        changed
    }

    fn set_of(&mut self, mut values: Vec<(&'static str, f64)>) -> ShadingSet {
        values.sort_by_key(|value| value.0);
        values.dedup_by_key(|value| value.0);
        let same = |known: &[(&'static str, f64)]| {
            known.len() == values.len()
                && known
                    .iter()
                    .zip(&values)
                    .all(|(a, b)| a.0 == b.0 && a.1.to_bits() == b.1.to_bits())
        };
        if let Some(found) = self.sets.iter().position(|set| same(&set.constants)) {
            return ShadingSet(found);
        }
        self.sets.push(ShadingPipelines::new(values));
        ShadingSet(self.sets.len() - 1)
    }

    pub fn shading(&self) -> ShadingSet {
        self.current
    }

    pub fn constants(&self) -> &[(&'static str, f64)] {
        &self.sets[self.current.0].constants
    }

    fn warming(&self) -> Vec<ShadingSet> {
        if self.cooking {
            self.warm.clone()
        } else {
            Vec::new()
        }
    }

    fn pipeline(&self, key: &(PassKind, Variant)) -> Option<&wgpu::RenderPipeline> {
        self.sets[self.current.0].variants.get(key)
    }

    fn has(&self, set: ShadingSet, key: &(PassKind, Variant)) -> bool {
        self.sets[set.0].variants.contains_key(key)
    }

    pub fn forget(&mut self, targets: Option<Targets>) {
        for set in &mut self.sets {
            set.variants
                .retain(|(kind, _), _| kind.targets() != targets);
        }
    }

    pub(crate) fn release_view(&mut self) {
        if self.view.take().is_some() {
            self.forget(None);
        }
    }
}

#[derive(Default)]
pub struct DrawOrder {
    pub order: Vec<u32>,
    pub masked: Vec<u32>,
    pub keys: Vec<(u8, Variant, Option<MaterialHandle>, u8, u32)>,
}

const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] = crate::vertex_attributes!(Vertex {
    0 => position as Float32x3,
    1 => normal as Snorm16x2,
    7 => uv as Float32x2,
});
const SLOT_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![2 => Uint32];
const EXTRA_ATTRIBUTES: [wgpu::VertexAttribute; 2] = crate::vertex_attributes!(ExtraVertex {
    9 => color as Unorm16x4,
    10 => uv1 as Float32x2,
});
pub fn extra_layout() -> Option<wgpu::VertexBufferLayout<'static>> {
    Some(wgpu::VertexBufferLayout {
        array_stride: size_of::<ExtraVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &EXTRA_ATTRIBUTES,
    })
}

pub fn vertex_layout() -> Option<wgpu::VertexBufferLayout<'static>> {
    Some(wgpu::VertexBufferLayout {
        array_stride: size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &VERTEX_ATTRIBUTES,
    })
}

pub fn slot_layouts() -> [Option<wgpu::VertexBufferLayout<'static>>; 2] {
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
pub struct BatchBinds {
    pub bindings: Bindings,
    variant: Option<Variant>,
}

impl Core {
    pub(crate) fn set_shading(&mut self, shading: &Shading) {
        if self.mesh_pipelines.specialize(shading) {
            self.structure_keys.variants = None;
            self.structure_keys.probes = None;
        }
    }

    pub fn batch_binding(
        &self,
        batch: &DrawBatch,
        mesh: Option<&GpuMesh>,
    ) -> (&wgpu::BindGroup, Variant) {
        let (group, variant) = self.materials.binding(batch.material);
        let variant = variant.with_extended(mesh.is_some_and(|mesh| mesh.extended));
        (group, variant)
    }

    pub fn bind_batch(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        binds: &mut BatchBinds,
        (kind, uber): (PassKind, bool),
        batch: &DrawBatch,
    ) -> Option<(&GpuMesh, MeshDraw<'_>)> {
        let mesh = &self.meshes[&batch.mesh];
        let (group, variant) = self.batch_binding(batch, Some(mesh));
        let variant = if uber {
            variant.uber()
        } else if kind.targets().is_some() {
            variant.opaque()
        } else {
            variant
        };
        let draw = self.bind_mesh(pass, binds, (kind, variant), (group, mesh))?;
        Some((mesh, draw))
    }

    pub fn bind_prepass(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        binds: &mut BatchBinds,
        kind: PassKind,
        batch: &DrawBatch,
    ) -> Option<(&GpuMesh, MeshDraw<'_>)> {
        let mesh = &self.meshes[&batch.mesh];
        let (group, variant) = self.batch_binding(batch, Some(mesh));
        let key = prepass_key(kind, variant)?;
        let draw = self.bind_mesh(pass, binds, key, (group, mesh))?;
        Some((mesh, draw))
    }

    fn bind_mesh<'m>(
        &'m self,
        pass: &mut wgpu::RenderPass<'_>,
        binds: &mut BatchBinds,
        key: (PassKind, Variant),
        (group, mesh): (&wgpu::BindGroup, &'m GpuMesh),
    ) -> Option<MeshDraw<'m>> {
        if binds.variant != Some(key.1) {
            pass.set_pipeline(self.mesh_pipelines.pipeline(&key)?);
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
        Some(draw)
    }

    pub fn opaque_pipeline(
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
        self.mesh_pipelines.pipeline(&key)
    }

    pub fn ensure_opaque_variant(&mut self, kind: PassKind, variant: Variant) {
        let set = self.mesh_pipelines.shading();
        for key in opaque_keys(kind, variant).into_iter().flatten() {
            self.ensure_variant(key, set);
        }
    }

    fn ensure_variant(&mut self, (kind, variant): (PassKind, Variant), set: ShadingSet) {
        if self.mesh_pipelines.has(set, &(kind, variant)) {
            return;
        }
        let Some(pipeline) = self.create_variant(kind, variant, set) else {
            return;
        };
        self.mesh_pipelines.sets[set.0]
            .variants
            .insert((kind, variant), pipeline);
    }

    pub(crate) fn ensure_variants(&mut self, opaque: PassKind) {
        if opaque.targets() != Some(Targets::Debug) {
            for set in self.mesh_pipelines.warming() {
                self.create_variants(opaque, set);
            }
        }
        let key = (self.structure_key(), opaque);
        if self.structure_keys.variants != Some(key) {
            self.create_variants(opaque, self.mesh_pipelines.shading());
            self.structure_keys.variants = Some(key);
        }
    }

    fn create_variants(&mut self, opaque: PassKind, set: ShadingSet) {
        let mut missing = Vec::new();
        for batch in &self.plan.batches {
            let (_, variant) = self.batch_binding(batch, self.meshes.get(&batch.mesh));
            for pipeline in opaque_keys(opaque, variant).into_iter().flatten() {
                if !self.mesh_pipelines.has(set, &pipeline) && !missing.contains(&pipeline) {
                    missing.push(pipeline);
                }
            }
        }
        for pipeline in missing {
            self.ensure_variant(pipeline, set);
        }
    }

    pub(crate) fn ensure_view_variants(&mut self, format: wgpu::TextureFormat, mirrored: bool) {
        if self.mesh_pipelines.view != Some((format, mirrored)) {
            self.mesh_pipelines.forget(None);
            self.mesh_pipelines.view = Some((format, mirrored));
            self.structure_keys.probes = None;
        }
        let kind = PassKind::view();
        for set in self.mesh_pipelines.warming() {
            self.create_view_variants(kind, set);
        }
        let key = (self.structure_key(), kind);
        if self.structure_keys.probes == Some(key) {
            return;
        }
        self.structure_keys.probes = Some(key);
        self.create_view_variants(kind, self.mesh_pipelines.shading());
    }

    fn create_view_variants(&mut self, kind: PassKind, set: ShadingSet) {
        let mut missing = Vec::new();
        for batch in &self.plan.batches {
            let Some(mesh) = self.meshes.get(&batch.mesh) else {
                continue;
            };
            let variant = self.batch_binding(batch, Some(mesh)).1.uber();
            if !self.mesh_pipelines.has(set, &(kind, variant)) && !missing.contains(&variant) {
                missing.push(variant);
            }
        }
        for variant in missing {
            self.ensure_variant((kind, variant), set);
        }
    }

    fn mesh_layout(&self, kind: PassKind) -> wgpu::PipelineLayout {
        let declared = match kind.slots() {
            Slots::Temporal => &LIT_TEMPORAL_MESH_PIPELINES,
            Slots::Retained => &LIT_MESH_PIPELINES,
        };
        crate::pipelines::layout_of(&self.device, declared)
    }

    fn create_variant(
        &self,
        kind: PassKind,
        variant: Variant,
        set: ShadingSet,
    ) -> Option<wgpu::RenderPipeline> {
        let entries = self.mesh_entries?;
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
            MeshPass::View => {
                let (format, mirrored) = self.mesh_pipelines.view?;
                MeshFragment {
                    targets: vec![Some(format.into())],
                    depth_compare: wgpu::CompareFunction::Greater,
                    depth_write: true,
                    mirrored,
                }
            }
        };
        Some(mesh_pipeline(
            &self.device,
            &layout,
            (kind, entries),
            &fragment,
            (variant, &self.mesh_pipelines.sets[set.0].constants),
        ))
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
    variant.alpha_mask().then_some((kind, variant.prepass()))
}

fn mesh_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    (kind, entries): (PassKind, &MeshEntries),
    fragment: &MeshFragment,
    (variant, shading): (Variant, &[(&'static str, f64)]),
) -> wgpu::RenderPipeline {
    let mut constants = variant.constants().to_vec();
    constants.extend_from_slice(shading);
    let extended = variant.extended();
    let streams = if extended {
        Streams::Extended
    } else {
        Streams::Basic
    };
    let vertex_entry = kind.vertex_entry(streams);
    let fragment_entry = if variant.depth_only() {
        (entries.prepass)(kind, extended)
    } else {
        (entries.fragment)(kind, extended)
    };
    let [mesh, slot_ids] = slot_layouts();
    let layouts = [mesh, slot_ids, extra_layout()];
    let buffers = match streams {
        Streams::Extended => 3,
        Streams::Basic => 2,
    };
    let buffers = if kind.targets() == Some(Targets::Debug) {
        crate::debug_targets::with_record(&layouts[..buffers])
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
