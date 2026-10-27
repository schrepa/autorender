//! Mesh pipelines: the vertex streams of meshes and slots, each pass kind's
//! material variants over their layouts (masked surfaces' depth pre-pass
//! and their colors after it among the opaque passes'), the sky, and the
//! opaque drawing order.

#![expect(
    clippy::expect_used,
    reason = "pipelines exist in the frames that draw with them"
)]

use super::*;
use crate::bindings::EMPTY;
use crate::entries::{self, Slots};
use crate::pipelines::vertex_attributes;

/// Mesh pipelines: their layouts, unlit and lit, created with
/// the first variant needing each (temporal frames' come with the temporal
/// pass); every pass kind's material variants, whose BASIC entries are the
/// base pipelines, masked surfaces' depth pre-pass among the opaque passes'
/// (keyed by their unlit kind, as it reads no lights); the sky by its
/// targets, built when a pass first draws it;
/// the empty group pipelines skipping a group bind (the visibility pass,
/// impostor bakes); the scene features every pipeline shading the scene
/// is specialized for; and whether the session cooks its pipeline list,
/// so frames create their variants lit and unlit alike.
pub(super) struct MeshPipelines {
    pub(super) layouts: [Option<wgpu::PipelineLayout>; 2],
    pub(super) variants: HashMap<(PassKind, Variant), wgpu::RenderPipeline>,
    sky: [Option<wgpu::RenderPipeline>; 5],
    pub(super) empty_layout: crate::pipelines::GroupLayout,
    pub(super) empty_group: wgpu::BindGroup,
    pub(super) shading: SceneShading,
    pub(super) cooking: bool,
}

/// The targets a sky draws into: an opaque pass's, or a reflection probe
/// face's.
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
            Self::Frame(Targets::Debug) => 3,
            Self::Probe => 4,
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
            variants: HashMap::default(),
            sky: Default::default(),
            empty_layout,
            empty_group,
            shading: SceneShading::ALL,
            cooking: false,
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

/// `kind` lit where it is not, and not where it is.
pub(super) fn relit(kind: PassKind) -> PassKind {
    PassKind {
        lit: !kind.lit,
        ..kind
    }
}

/// The plan's opaque batches in drawing order, the masked ones among them
/// in the same order (which masked surfaces' depth pre-pass draws), and
/// the sort keys ordering them.
#[derive(Default)]
pub(super) struct DrawOrder {
    pub(super) order: Vec<u32>,
    pub(super) masked: Vec<u32>,
    pub(super) keys: Vec<(u8, Variant, Option<MaterialHandle>, u8, u32)>,
}

// Position, octahedral normal and texture coordinates.
const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] = vertex_attributes!(Vertex {
    0 => position as Float32x3,
    1 => normal as Snorm16x2,
    7 => uv as Float32x2,
});
const SLOT_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![2 => Uint32];
// Vertex color and second texture coordinates.
const EXTRA_ATTRIBUTES: [wgpu::VertexAttribute; 2] = vertex_attributes!(ExtraVertex {
    9 => color as Unorm16x4,
    10 => uv1 as Float32x2,
});
// Joints and weights of meshes with joints.
const SKIN_ATTRIBUTES: [wgpu::VertexAttribute; 2] = vertex_attributes!(SkinVertex {
    11 => joints as Uint16x4,
    12 => weights as Unorm16x4,
});

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
    /// material's, with the extra streams of its
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
    /// (the uber variant with `uber`; in opaque passes the one drawing its
    /// colors, [`Variant::opaque`]), switched only when the variant changes,
    /// its material group and its mesh's streams. Returns the mesh and its
    /// draw.
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

    /// Binds what `batch` draws in masked surfaces' depth pre-pass of an
    /// opaque pass of `kind`, as [`Self::bind_batch`] binds its colors:
    /// `None`, binding nothing, unless its material is masked.
    pub(super) fn bind_prepass(
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

    /// Binds the pipeline of `key`, switched only when its variant changes,
    /// material group `group` and `mesh`'s streams; returns the mesh's draw.
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
        if let Some(skin) = draw.skin.filter(|_| mesh.skinned) {
            bindings.vertex(pass, 3, skin);
        }
        bindings.index(pass, draw.indices);
        draw
    }

    /// The pipeline the opaque passes of `kind` draw surfaces of `variant`
    /// with, once created: their colors, or with `prepass` masked ones'
    /// depth pre-pass (`None` for others).
    pub(super) fn opaque_pipeline(
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

    /// Creates the pipelines the opaque passes of `kind` draw surfaces of
    /// `variant` with, unless created: their colors, and masked ones' depth
    /// pre-pass.
    pub(super) fn ensure_opaque_variant(&mut self, kind: PassKind, variant: Variant) {
        for key in opaque_keys(kind, variant).into_iter().flatten() {
            self.ensure_variant(key);
        }
    }

    /// Creates the pipeline of `kind` shading materials of `variant` unless
    /// created.
    fn ensure_variant(&mut self, (kind, variant): (PassKind, Variant)) {
        if self.mesh_pipelines.variants.contains_key(&(kind, variant)) {
            return;
        }
        self.ensure_mesh_layout(kind);
        let pipeline = self.create_variant(kind, variant);
        self.mesh_pipelines
            .variants
            .insert((kind, variant), pipeline);
    }

    /// Creates the pipelines of the plan's material variants for this frame's
    /// opaque and transparent passes, and the opaque pass's sky.
    pub(super) fn ensure_variants(&mut self, opaque: PassKind, transparent: PassKind) {
        // Debug views shade no lights.
        if self.relights() && opaque.targets() != Some(Targets::Debug) {
            self.create_variants(relit(opaque), relit(transparent));
        }
        self.create_variants(opaque, transparent);
        // The opaque pass ends with the sky of its targets.
        if let Some(targets) = opaque.targets() {
            self.ensure_sky(SkyTargets::Frame(targets));
        }
    }

    /// Whether frames also create the pipelines of frames lit otherwise:
    /// while the session cooks its pipeline list, with local lights in the
    /// scene, which light a frame where some reach its view.
    pub(super) fn relights(&self) -> bool {
        self.mesh_pipelines.cooking && self.features.lights.get().is_some()
    }

    /// Creates the pipelines of the plan's material variants for `opaque`
    /// and `transparent` passes unless created: the opaque passes' masked
    /// surfaces' depth pre-pass too.
    fn create_variants(&mut self, opaque: PassKind, transparent: PassKind) {
        let key = (self.structure_key(), opaque, transparent);
        if self.structure_keys.variants != Some(key) {
            let mut missing = Vec::new();
            // Debug views draw no transparent surfaces.
            let debug = opaque.targets() == Some(Targets::Debug);
            for batch in &self.plan.batches {
                if batch.transparent && debug {
                    continue;
                }
                let (_, variant) = self.batch_binding(batch, self.meshes.get(&batch.mesh));
                let drawn = if batch.transparent {
                    [Some((transparent, variant)), None]
                } else {
                    opaque_keys(opaque, variant)
                };
                for pipeline in drawn.into_iter().flatten() {
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
            (&self.camera.layout, self.features.sky.get().pass_layout()),
            fragment,
            &color_targets,
        ));
    }

    /// The pipeline layout of `kind`'s mesh variants unless created: the
    /// camera and material groups, the slots' poses, and the lights' group
    /// for lit ones. Temporal frames have their own.
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
            self.features
                .lights
                .get()
                .expect("lit frames create the light pass")
                .fragment_layout()
        });
        let groups = (
            &self.camera.layout,
            self.materials.layout(),
            &self.instances.pose_layout,
        );
        self.mesh_pipelines.layouts[index] = Some(match lights {
            Some(lights) => crate::pipelines::pipeline_layout(
                &self.device,
                &crate::pipeline_layouts::LIT_MESH_PIPELINES,
                &[groups.0, groups.1, groups.2, lights],
            ),
            None => crate::pipelines::pipeline_layout(
                &self.device,
                &crate::pipeline_layouts::MESH_PIPELINES,
                &[groups.0, groups.1, groups.2],
            ),
        });
    }

    /// The pipeline of `kind` shading materials of `variant`; in opaque
    /// passes, for [`Variant::depth_only`] variants, masked surfaces' depth
    /// pre-pass, which writes depth and no color, and for
    /// [`Variant::pretested`] ones their colors after it, drawn where its
    /// depths equal theirs.
    pub(super) fn create_variant(&self, kind: PassKind, variant: Variant) -> wgpu::RenderPipeline {
        let canvas = self.display.format();
        let layout = if kind.slots() == Slots::Temporal {
            let temporal = self
                .features
                .temporal
                .get()
                .expect("temporal frames create it");
            temporal.mesh_layout(kind.lit)
        } else {
            self.mesh_pipelines.layouts[usize::from(kind.lit)]
                .as_ref()
                .expect("created with the variant")
        };
        let fragment = match kind.pass {
            // The depth pre-pass draws in its pass's targets, writing none.
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
            MeshPass::Transparent => MeshFragment {
                targets: accumulation_targets().to_vec(),
                depth_compare: wgpu::CompareFunction::Greater,
                depth_write: false,
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
            &self.mesh_shader,
            layout,
            kind,
            &fragment,
            (variant, self.mesh_pipelines.shading),
        )
    }

    /// Orders the plan's opaque batches for drawing: flat ground last, so
    /// depth testing leaves it only the pixels nothing standing on it covers;
    /// then by pipeline variant, material and vertex buffers. Meshes share
    /// the arena's buffers, so passes then switch pipelines and bind groups
    /// once per run rather than per batch. The masked ones keep that order
    /// in their depth pre-pass's list.
    pub(super) fn order_draws(&mut self, snapshot: &engine_core::RenderSnapshot) {
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
                    let flat = flat_batch(snapshot, &self.plan, batch);
                    (
                        u8::from(flat),
                        variant,
                        batch.material,
                        buffers,
                        index as u32,
                    )
                }),
        );
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

/// Whether `batch` draws flat ground: its first instance's world box is
/// under a tenth as tall as it is wide.
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

/// The sky pass of `fragment` into `targets`: one triangle at the far plane,
/// tested but not written against depth, over the camera bindings and the
/// sky's transmittance table.
fn sky_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    (camera_layout, transmittance_layout): (
        &crate::pipelines::GroupLayout,
        &crate::pipelines::GroupLayout,
    ),
    fragment: crate::shaders::Entry,
    targets: &[Option<wgpu::ColorTargetState>],
) -> wgpu::RenderPipeline {
    let layout = crate::pipelines::pipeline_layout(
        device,
        &crate::pipeline_layouts::SKY_PIPELINES,
        &[camera_layout, transmittance_layout],
    );
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

/// Fragment stage of a mesh pipeline: color targets, how it tests depth
/// and whether it writes it (transparent surfaces and masked ones' colors
/// after their depth pre-pass do not).
struct MeshFragment {
    targets: Vec<Option<wgpu::ColorTargetState>>,
    depth_compare: wgpu::CompareFunction,
    depth_write: bool,
    // Drawn through a mirroring view (reflection probe faces): triangles
    // wind the other way.
    mirrored: bool,
}

/// `targets` with nothing written to them: masked surfaces' depth pre-pass
/// draws in its opaque pass's targets, and its fragments write no color.
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

/// The pipelines the opaque passes of `kind` draw surfaces of `variant`
/// with: their colors, and masked ones' depth pre-pass before them.
fn opaque_keys(kind: PassKind, variant: Variant) -> [Option<(PassKind, Variant)>; 2] {
    [Some((kind, variant.opaque())), prepass_key(kind, variant)]
}

/// The pipeline of masked surfaces of `variant`'s depth pre-pass in the
/// opaque passes of `kind`, which reads no lights, so lit and unlit frames
/// share it; `None` for surfaces that are not masked.
fn prepass_key(kind: PassKind, variant: Variant) -> Option<(PassKind, Variant)> {
    variant
        .alpha_mask()
        .then_some((PassKind { lit: false, ..kind }, variant.prepass()))
}

/// A mesh pipeline of `kind` over `layout`, shading materials of `variant`
/// in a scene of `shading`'s features.
fn mesh_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    kind: PassKind,
    fragment: &MeshFragment,
    (variant, shading): (Variant, SceneShading),
) -> wgpu::RenderPipeline {
    let constants = shading.with(&variant.constants());
    // Extended meshes read a third vertex buffer through
    // their own entry points; skinned meshes a fourth.
    let extended = variant.extended();
    let skinned = variant.skinned();
    let streams = if skinned {
        Streams::Skinned
    } else if extended {
        Streams::Extended
    } else {
        Streams::Basic
    };
    // Masked surfaces' depth pre-pass draws the vertex stages of their
    // colors, so their depths are the same, with a fragment stage of its
    // own testing their alpha alone.
    let vertex_entry = kind.vertex_entry(streams);
    let fragment_entry = if variant.depth_only() {
        kind.prepass_entry(extended)
    } else {
        kind.fragment_entry(extended)
    };
    let [mesh, slot_ids] = slot_layouts();
    let layouts = [mesh, slot_ids, extra_layout(), skin_layout()];
    let buffers = match streams {
        Streams::Skinned => 4,
        Streams::Extended => 3,
        Streams::Basic => 2,
    };
    // Debug views read the draw's record.
    let buffers = if kind.targets() == Some(Targets::Debug) {
        crate::debug_view::with_record(&layouts[..buffers])
    } else {
        layouts[..buffers].to_vec()
    };
    // Entry points and variant bits name the pipeline in captures.
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
            module: shader,
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
