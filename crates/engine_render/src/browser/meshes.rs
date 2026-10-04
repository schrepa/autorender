//! Mesh pipelines: the vertex streams of meshes and slots, each pass kind's
//! material variants (decision 031) over their layouts, the sky, and the
//! opaque drawing order (decision 046).

use super::*;
use crate::bindings::EMPTY;
use crate::entries::{self, Slots};

/// Mesh pipelines (decision 049): their layouts, unlit and lit, created with
/// the first variant needing each (temporal frames' come with the temporal
/// pass); every pass kind's material variants, whose BASIC entries are the
/// base pipelines; the sky by its targets, built when a pass first draws it;
/// and the empty group pipelines skipping a group bind (the visibility pass,
/// impostor bakes).
pub(super) struct MeshPipelines {
    pub(super) layouts: [Option<wgpu::PipelineLayout>; 2],
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

/// Mesh vertices and the slot ID stream of CPU-prepared frames.
pub(super) fn slot_layouts() -> [Option<wgpu::VertexBufferLayout<'static>>; 2] {
    [
        vertex_layout(),
        Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<u32>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &SLOT_ATTRIBUTES,
        }),
    ]
}

/// What a pass drawing batches bound last: buffers and groups, and the
/// pipeline variant.
#[derive(Default)]
pub(super) struct BatchBinds {
    pub(super) bindings: Bindings,
    variant: Option<Variant>,
}

impl Renderer {
    /// The material group and pipeline variant a draw of `batch` uses: its
    /// material's, with the extra streams (decisions 034 and 036) of its
    /// `mesh`, plain while that is not resident.
    pub(super) fn batch_binding(
        &self,
        batch: &DrawBatch,
        mesh: Option<&GpuMesh>,
    ) -> (&wgpu::BindGroup, Variant) {
        let (group, variant) = self.materials.binding(batch.material);
        let variant = variant
            .with_extended(mesh.is_some_and(|mesh| mesh.extended))
            .with_skinned(mesh.is_some_and(|mesh| mesh.skinned));
        (group, variant)
    }

    /// Binds what `batch` draws in a pass of `kind`: its variant's pipeline
    /// (the uber variant with `uber`, decision 049), switched only when the
    /// variant changes, its material group and its mesh's streams. Returns
    /// the mesh and its draw.
    pub(super) fn bind_batch(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        binds: &mut BatchBinds,
        (kind, uber): (PassKind, bool),
        batch: &DrawBatch,
    ) -> (&GpuMesh, MeshDraw<'_>) {
        // Every planned source was validated and uploaded above.
        let mesh = &self.meshes[&batch.mesh];
        let (group, variant) = self.batch_binding(batch, Some(mesh));
        let variant = if uber { variant.uber() } else { variant };
        if binds.variant != Some(variant) {
            pass.set_pipeline(&self.mesh_pipelines.variants[&(kind, variant)]);
            binds.variant = Some(variant);
        }
        let bindings = &mut binds.bindings;
        bindings.group(pass, 1, group);
        let draw = mesh.draw(&self.mesh_arena);
        bindings.vertex(pass, 0, draw.vertices);
        if let Some(extras) = draw.extras.filter(|_| mesh.extended) {
            bindings.vertex(pass, 2, extras);
        }
        if let Some(skin) = draw.skin.filter(|_| mesh.skinned) {
            bindings.vertex(pass, 3, skin);
        }
        bindings.index(pass, draw.indices);
        (mesh, draw)
    }

    /// Creates the pipelines of the plan's material variants for this frame's
    /// opaque and transparent passes, and the opaque pass's sky.
    pub(super) fn ensure_variants(&mut self, opaque: PassKind, transparent: PassKind) {
        let key = (self.structure_key(), opaque, transparent);
        if self.structure_keys.variants != Some(key) {
            let mut missing = Vec::new();
            for batch in &self.plan.batches {
                let kind = if batch.transparent {
                    transparent
                } else {
                    opaque
                };
                let (_, variant) = self.batch_binding(batch, self.meshes.get(&batch.mesh));
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
            self.structure_keys.variants = Some(key);
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
    /// camera and material groups, the slots' poses, and the lights' group
    /// for lit ones (decision 023). Temporal frames have their own.
    pub(super) fn ensure_mesh_layout(&mut self, kind: PassKind) {
        if kind.targets() == Some(Targets::Temporal) {
            return;
        }
        let lit = kind.lit;
        let index = usize::from(lit);
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
            Some(&self.instances.pose_layout),
            lights,
        ];
        let count = if lit { 4 } else { 3 };
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
            self.mesh_pipelines.layouts[usize::from(kind.lit)]
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
        let key = self.structure_key();
        if self.structure_keys.order == Some(key) {
            return;
        }
        self.structure_keys.order = Some(key);
        let mut keys = std::mem::take(&mut self.draw_order.keys);
        keys.clear();
        keys.extend(
            self.plan
                .batches
                .iter()
                .enumerate()
                .filter(|(_, batch)| !batch.transparent)
                .map(|(index, batch)| {
                    let mesh = &self.meshes[&batch.mesh];
                    let (_, variant) = self.batch_binding(batch, Some(mesh));
                    let buffers = match &mesh.placement {
                        Ok(placement) => placement.streams as u8,
                        Err(_) => u8::MAX,
                    };
                    (variant, batch.material, buffers, index as u32)
                }),
        );
        keys.sort_unstable();
        self.draw_order.keys = keys;
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
    // their own entry points; skinned meshes (decision 036) a fourth.
    let extended = variant.extended();
    let skinned = variant.skinned();
    let streams = if skinned {
        Streams::Skinned
    } else if extended {
        Streams::Extended
    } else {
        Streams::Basic
    };
    let vertex_entry = kind.vertex_entry(streams);
    let fragment_entry = kind.fragment_entry(extended);
    let [mesh, slot_ids] = slot_layouts();
    let layouts = [mesh, slot_ids, extra_layout(), skin_layout()];
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
    crate::pipelines::geometry(
        device,
        &crate::pipelines::Geometry {
            label: &label,
            layout,
            module: shader,
            vertex: vertex_entry,
            buffers: &layouts[..buffers],
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
                depth_compare: Some(wgpu::CompareFunction::Greater),
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
