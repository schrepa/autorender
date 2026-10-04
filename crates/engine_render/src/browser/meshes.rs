//! Mesh pipelines: the vertex streams of meshes and instances, each pass
//! kind's material variants (decision 031) over their layouts, the sky,
//! and the opaque drawing order (decision 046).

use super::*;
use crate::bindings::EMPTY;
use crate::entries::{self, Slots};

/// Mesh pipelines (decision 049): their layouts by `mesh_layout_index`,
/// created with the first variant needing each (temporal frames' come with
/// the temporal pass); every pass kind's material variants, whose BASIC
/// entries are the base pipelines; the sky by its targets, built when a pass
/// first draws it; and the empty group lit public-instance pipelines bind as
/// group 2.
pub(super) struct MeshPipelines {
    pub(super) layouts: [Option<wgpu::PipelineLayout>; 4],
    pub(super) variants: HashMap<(PassKind, Variant), wgpu::RenderPipeline>,
    sky: [Option<wgpu::RenderPipeline>; 4],
    pub(super) empty_layout: wgpu::BindGroupLayout,
    pub(super) empty_group: wgpu::BindGroup,
}

/// The targets a sky draws into: an opaque pass's, or a reflection probe
/// face's (decision 043).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SkyTargets {
    Frame(Targets),
    Probe,
}

impl SkyTargets {
    fn index(self) -> usize {
        match self {
            Self::Frame(Targets::Canvas) => 0,
            Self::Frame(Targets::Post) => 1,
            Self::Frame(Targets::Temporal) => 2,
            Self::Probe => 3,
        }
    }
}

impl MeshPipelines {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let empty_layout = crate::pipelines::layout(device, &EMPTY);
        let empty_group =
            crate::pipelines::group(device, "engine/empty-group", &EMPTY, &empty_layout, &[]);
        Self {
            layouts: Default::default(),
            variants: HashMap::new(),
            sky: Default::default(),
            empty_layout,
            empty_group,
        }
    }

    /// The sky into `targets`, once [`Renderer::ensure_sky`] built it.
    pub(super) fn sky(&self, targets: SkyTargets) -> &wgpu::RenderPipeline {
        self.sky[targets.index()]
            .as_ref()
            .expect("prepared with the frame's pipelines")
    }

    /// Forgets the sky into `targets`.
    pub(super) fn forget_sky(&mut self, targets: SkyTargets) {
        self.sky[targets.index()] = None;
    }
}

/// The plan's opaque batches in drawing order (decision 046), and the sort
/// keys ordering them.
#[derive(Default)]
pub(super) struct DrawOrder {
    pub(super) order: Vec<u32>,
    pub(super) keys: Vec<(Variant, Option<MaterialHandle>, u8, u32)>,
}

// Position, octahedral normal and texture coordinates.
const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Snorm16x2, 7 => Float32x2];
// Columns, packed color and packed surface of public instance records.
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Uint32x2, 8 => Uint32];
const SLOT_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![2 => Uint32];
// Vertex color and second texture coordinates (decision 034).
const EXTRA_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![9 => Unorm16x4, 10 => Float32x2];
// Joints and weights of meshes with joints (decision 036).
const SKIN_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![11 => Uint16x4, 12 => Unorm16x4];

/// The skin stream of skinned draws, after the second stream.
pub(crate) fn skin_layout() -> Option<wgpu::VertexBufferLayout<'static>> {
    Some(wgpu::VertexBufferLayout {
        array_stride: size_of::<SkinVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &SKIN_ATTRIBUTES,
    })
}

/// The second vertex stream of extended meshes.
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

/// Mesh vertices and the slot ID stream of retained frames.
pub(super) fn affine_layouts() -> [Option<wgpu::VertexBufferLayout<'static>>; 2] {
    [
        vertex_layout(),
        Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<u32>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &SLOT_ATTRIBUTES,
        }),
    ]
}

impl Renderer {
    /// Creates the pipelines of the plan's material variants for this frame's
    /// opaque and transparent passes, and the opaque pass's sky.
    pub(super) fn ensure_variants(&mut self, opaque: PassKind, transparent: PassKind) {
        let mut missing = Vec::new();
        for batch in &self.plan.batches {
            let (_, variant) = self.materials.binding(batch.material);
            let kind = if batch.transparent {
                transparent
            } else {
                opaque
            };
            // Only retained slots skin (decision 036).
            let mesh = self.meshes.get(&batch.mesh);
            let variant = variant
                .with_extended(mesh.is_some_and(|mesh| mesh.extended))
                .with_skinned(kind.affine && mesh.is_some_and(|mesh| mesh.skinned));
            if !self.mesh_pipelines.variants.contains_key(&(kind, variant))
                && !missing.contains(&(kind, variant))
            {
                missing.push((kind, variant));
            }
        }
        for (kind, variant) in missing {
            self.ensure_mesh_layout(kind);
            let pipeline = self.create_variant(kind, variant);
            self.mesh_pipelines
                .variants
                .insert((kind, variant), pipeline);
        }
        // The opaque pass ends with the sky of its targets.
        if let Some(targets) = opaque.targets() {
            self.ensure_sky(SkyTargets::Frame(targets));
        }
    }

    /// The sky into `targets`, built the first time a pass draws it.
    pub(super) fn ensure_sky(&mut self, targets: SkyTargets) {
        let index = targets.index();
        if self.mesh_pipelines.sky[index].is_some() {
            return;
        }
        let canvas = self.display.format();
        let (fragment, color_targets) = match targets {
            SkyTargets::Frame(targets) => (
                entries::sky_fragment(Some(targets)),
                targets.color_targets(canvas),
            ),
            SkyTargets::Probe => (
                entries::sky_fragment(None),
                vec![Some(ENVIRONMENT_FORMAT.into())],
            ),
        };
        self.mesh_pipelines.sky[index] = Some(sky_pipeline(
            &self.device,
            &self.mesh_shader,
            &self.camera.layout,
            fragment,
            &color_targets,
        ));
    }

    /// The pipeline layout of `kind`'s mesh variants unless created: the
    /// camera and material groups, retained slots' poses for affine passes
    /// (an empty group otherwise where lights follow), and the lights' group
    /// for lit ones (decision 023). Temporal frames have their own.
    pub(super) fn ensure_mesh_layout(&mut self, kind: PassKind) {
        if kind.targets() == Some(Targets::Temporal) {
            return;
        }
        let (lit, affine) = (kind.lit, kind.affine);
        let index = mesh_layout_index(lit, affine);
        if self.mesh_pipelines.layouts[index].is_some() {
            return;
        }
        let lights = lit.then(|| {
            self.lights
                .as_ref()
                .expect("lit frames create the light pass")
                .fragment_layout()
        });
        let groups = [
            Some(&self.camera.layout),
            Some(self.materials.layout()),
            Some(if affine {
                &self.instances.pose_layout
            } else {
                &self.mesh_pipelines.empty_layout
            }),
            lights,
        ];
        let count = if lit { 4 } else { 2 + usize::from(affine) };
        self.mesh_pipelines.layouts[index] = Some(self.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("engine/mesh-pipeline-layout"),
                bind_group_layouts: &groups[..count],
                immediate_size: 0,
            },
        ));
    }

    /// The pipeline of `kind` shading materials of `variant`.
    pub(super) fn create_variant(&self, kind: PassKind, variant: Variant) -> wgpu::RenderPipeline {
        let canvas = self.display.format();
        let layout = if kind.slots() == Slots::Temporal {
            let temporal = self.temporal.as_ref().expect("temporal frames create it");
            temporal.mesh_layout(kind.lit)
        } else {
            self.mesh_pipelines.layouts[mesh_layout_index(kind.lit, kind.affine)]
                .as_ref()
                .expect("created with the variant")
        };
        let fragment = match kind.pass {
            MeshPass::Opaque(targets) => MeshFragment {
                targets: targets.color_targets(canvas),
                depth_write: true,
                mirrored: false,
            },
            MeshPass::Transparent => MeshFragment {
                targets: accumulation_targets().to_vec(),
                depth_write: false,
                mirrored: false,
            },
            MeshPass::Probe => MeshFragment {
                targets: vec![Some(ENVIRONMENT_FORMAT.into())],
                depth_write: true,
                mirrored: true,
            },
        };
        mesh_pipeline(
            &self.device,
            &self.mesh_shader,
            layout,
            kind,
            &fragment,
            variant,
        )
    }

    /// Orders the plan's opaque batches for drawing: by pipeline variant,
    /// then material and vertex buffers. Meshes share the arena's buffers,
    /// so passes then switch pipelines and bind groups once per run rather
    /// than per batch.
    pub(super) fn order_draws(&mut self) {
        let (plan, meshes, materials) = (&self.plan, &self.meshes, &self.materials);
        self.draw_order.keys.clear();
        self.draw_order
            .keys
            .extend(
                plan.batches
                    .iter()
                    .enumerate()
                    .filter_map(|(index, batch)| {
                        if batch.transparent {
                            return None;
                        }
                        let mesh = &meshes[&batch.mesh];
                        let (_, variant) = materials.binding(batch.material);
                        let variant = variant
                            .with_extended(mesh.extended)
                            .with_skinned(mesh.skinned);
                        let buffers = match &mesh.placement {
                            Ok(placement) => placement.streams as u8,
                            Err(_) => u8::MAX,
                        };
                        Some((variant, batch.material, buffers, index as u32))
                    }),
            );
        self.draw_order.keys.sort_unstable();
        self.draw_order.order.clear();
        self.draw_order
            .order
            .extend(self.draw_order.keys.iter().map(|&(.., index)| index));
    }
}

/// The sky pass of `fragment` into `targets`: one triangle at the far plane,
/// tested but not written against depth, over the camera bindings.
fn sky_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    camera_layout: &wgpu::BindGroupLayout,
    fragment: &str,
    targets: &[Option<wgpu::ColorTargetState>],
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("engine/sky-pipeline-layout"),
        bind_group_layouts: &[Some(camera_layout)],
        immediate_size: 0,
    });
    crate::pipelines::fullscreen(
        device,
        &crate::pipelines::Fullscreen {
            label: "engine/sky",
            layout: &layout,
            module: shader,
            vertex: entries::SKY_VERTEX,
            fragment,
            constants: &[],
            targets,
            // On the far plane: it shows where nothing was drawn.
            depth: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
        },
    )
}

/// Fragment stage of a mesh pipeline: color targets and pipeline
/// constants, and whether it writes depth (transparent surfaces do not).
struct MeshFragment {
    targets: Vec<Option<wgpu::ColorTargetState>>,
    depth_write: bool,
    // Drawn through a mirroring view (reflection probe faces): triangles
    // wind the other way.
    mirrored: bool,
}

/// A mesh pipeline of `kind` over `layout`, shading materials of `variant`
/// (decision 031).
fn mesh_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    kind: PassKind,
    fragment: &MeshFragment,
    variant: Variant,
) -> wgpu::RenderPipeline {
    let constants = variant.constants();
    // Extended meshes (decision 034) read a third vertex buffer through
    // their own entry points; skinned slots' meshes (decision 036) a fourth.
    let affine = kind.slots() != Slots::Instances;
    let extended = variant.extended();
    let skinned = affine && variant.skinned();
    let streams = if skinned {
        Streams::Skinned
    } else if extended {
        Streams::Extended
    } else {
        Streams::Basic
    };
    let vertex_entry = kind.vertex_entry(streams);
    let fragment_entry = kind.fragment_entry(extended);
    let full_layouts = [
        vertex_layout(),
        Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<InstanceData>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE_ATTRIBUTES,
        }),
        extra_layout(),
    ];
    let [mesh, slot_ids] = affine_layouts();
    let affine_layouts = [mesh, slot_ids, extra_layout(), skin_layout()];
    let buffers = match streams {
        Streams::Skinned => 4,
        Streams::Extended => 3,
        Streams::Basic => 2,
    };
    // Entry points and variant bits name the pipeline in captures.
    let label = format!(
        "engine/mesh {vertex_entry} {fragment_entry} {:#06x}",
        variant.bits()
    );
    crate::pipelines::render(
        device,
        &wgpu::RenderPipelineDescriptor {
            label: Some(&label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some(vertex_entry),
                compilation_options: Default::default(),
                buffers: if affine {
                    &affine_layouts[..buffers]
                } else {
                    &full_layouts[..buffers]
                },
            },
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(fragment.depth_write),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(fragment_entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                targets: &fragment.targets,
            }),
            multiview_mask: None,
            cache: None,
        },
    )
}

/// The index in `mesh_layouts` of the mesh pipeline layout of lit and
/// affine passes.
fn mesh_layout_index(lit: bool, affine: bool) -> usize {
    usize::from(lit) * 2 + usize::from(affine)
}
