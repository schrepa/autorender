//! Decals: one record per decal and the pass drawing their
//! boxes after the opaque meshes, a feature of the renderer
//! (`crate::render_features`) created with the first frame that has decals.

#![expect(
    clippy::expect_used,
    reason = "drawing reads the group this frame prepared"
)]

use std::ops::Range;

use engine_core::{DecalBlend, MaterialHandle, RenderDecal};

pub(crate) use crate::gpu_layouts::constants::NO_DECAL_POSE as NO_POSE;
pub(crate) use crate::gpu_layouts::decals::DecalData;

impl DecalData {
    /// A decal's record, reading its pose from the frame's joint palette
    /// where it holds one.
    pub(crate) fn new(decal: &RenderDecal) -> Self {
        let rows = decal.model.transpose();
        Self {
            model: [
                rows.x_axis.to_array(),
                rows.y_axis.to_array(),
                rows.z_axis.to_array(),
            ],
            color: decal.color,
            params: [
                decal.fade,
                decal.surface.roughness,
                decal.surface.metalness,
                decal.surface.emission,
            ],
            pose: [decal.joint.unwrap_or(NO_POSE), 0, 0, 0],
        }
    }
}

/// Consecutive decals drawing with one material and blend.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DecalRun {
    pub material: Option<MaterialHandle>,
    pub blend: DecalBlend,
    pub records: Range<u32>,
}

/// Records of `decals` in order, up to `room` (what the GPU's binding
/// holds), and their runs; returns the decals that did not fit.
pub(crate) fn pack_decals(
    decals: &[RenderDecal],
    room: usize,
    records: &mut Vec<DecalData>,
    runs: &mut Vec<DecalRun>,
) -> usize {
    records.clear();
    runs.clear();
    let fitting = &decals[..decals.len().min(room)];
    for decal in fitting {
        let index = records.len() as u32;
        records.push(DecalData::new(decal));
        match runs.last_mut() {
            Some(run) if run.material == decal.material && run.blend == decal.blend => {
                run.records.end = index + 1;
            }
            _ => runs.push(DecalRun {
                material: decal.material,
                blend: decal.blend,
                records: index..index + 1,
            }),
        }
    }
    decals.len() - fitting.len()
}

mod gpu {
    use engine_core::collections::HashMap;

    use engine_core::{Drawn, RenderSnapshot};

    use super::*;
    use crate::bindings::DECALS;
    use crate::buffers::Growable;
    use crate::cached::{Bound, GroupCache, Versioned};
    use crate::materials::MaterialStore;
    use crate::profiling::Span;
    use crate::render_features::{Encode, FeaturePass, RenderFeature};
    use crate::render_graph::{Access, FrameShape, Node, Pass, RenderGraph, Resource};
    use crate::targets::Targets;

    /// A frame's decals: those drawn and those beyond the GPU's binding.
    pub(crate) struct DecalFrame {
        pub drawn: u32,
        pub dropped: u32,
    }

    /// Pipelines by target (canvas or scene target), blend and whether they
    /// shade local lights.
    type PipelineKey = (bool, DecalBlend, bool);

    pub(crate) struct DecalPass {
        layout: crate::pipelines::GroupLayout,
        // Without and with the local lights' group.
        pipeline_layouts: [Option<wgpu::PipelineLayout>; 2],
        pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,
        surface_format: wgpu::TextureFormat,
        // As many records as its room holds.
        records: Growable,
        staging: Vec<DecalData>,
        uploaded: Vec<DecalData>,
        runs: Vec<DecalRun>,
        // Per pose buffer parity (temporal frames alternate them), and the
        // parity this frame binds.
        groups: [GroupCache<3>; 2],
        parity: usize,
    }

    impl DecalPass {
        pub(crate) fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
            // Group 2 beside the mesh poses' bindings: the joint palette
            // holds retained frames' decal poses.
            let layout = crate::pipelines::layout(device, &DECALS);
            Self {
                layout,
                pipeline_layouts: [None, None],
                pipelines: HashMap::default(),
                surface_format,
                records: Growable::new(
                    device,
                    "engine/decal-records",
                    size_of::<DecalData>() as u64,
                    wgpu::BufferUsages::empty(),
                ),
                staging: Vec::new(),
                uploaded: Vec::new(),
                runs: Vec::new(),
                groups: Default::default(),
                parity: 0,
            }
        }

        /// Whether `snapshot`'s frame draws decals.
        pub(crate) fn wanted(snapshot: &RenderSnapshot) -> bool {
            snapshot.environment.decals.enabled && !snapshot.decals.list.is_empty()
        }

        /// Uploads `snapshot`'s decals when they changed; `None` when the
        /// frame draws none. Decals read their displayed poses from
        /// `palette` (the frame's joint palette, bound, and its parity) and
        /// sample the depth buffer `depth`.
        pub(crate) fn prepare(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            snapshot: &RenderSnapshot,
            (palette, parity): (Bound<'_>, usize),
            depth: &Versioned<wgpu::TextureView>,
        ) -> Option<DecalFrame> {
            let room = (self.records.room() / size_of::<DecalData>() as u64) as usize;
            let dropped = pack_decals(
                &snapshot.decals.list,
                room,
                &mut self.staging,
                &mut self.runs,
            );
            if self.staging.is_empty() {
                return None;
            }
            let bytes = (self.staging.len() * size_of::<DecalData>()) as u64;
            if self.records.grow(device, bytes) {
                self.uploaded.clear();
            }
            if self.staging != self.uploaded {
                queue.write_buffer(&self.records, 0, bytemuck::cast_slice(&self.staging));
                self.uploaded.clone_from(&self.staging);
            }
            self.parity = parity;
            self.groups[parity].bind(
                device,
                "engine/decal-group",
                (&DECALS, &self.layout),
                [
                    ("palette", palette),
                    ("decal_records", self.records.bound()),
                    ("decal_depth", depth.bound()),
                ],
            );
            Some(DecalFrame {
                drawn: self.staging.len() as u32,
                dropped: dropped as u32,
            })
        }

        /// Creates the pipelines this frame's runs draw with: into the canvas
        /// or the scene target (`post`), shading local lights with `lights`'
        /// group layout.
        pub(crate) fn ensure_pipelines(
            &mut self,
            device: &wgpu::Device,
            shader: &wgpu::ShaderModule,
            groups: [&crate::pipelines::GroupLayout; 2],
            lights: Option<&crate::pipelines::GroupLayout>,
            post: bool,
        ) {
            let lit = lights.is_some();
            for index in 0..self.runs.len() {
                let blend = self.runs[index].blend;
                // Multiplied color keeps the surface's light.
                let shades_lights = lit && blend == DecalBlend::Alpha;
                let key = (!post, blend, shades_lights);
                if self.pipelines.contains_key(&key) {
                    continue;
                }
                let layout =
                    self.pipeline_layouts[usize::from(shades_lights)].get_or_insert_with(|| {
                        let [camera, material] = groups;
                        match lights.filter(|_| shades_lights) {
                            Some(lights) => crate::pipelines::pipeline_layout(
                                device,
                                &crate::pipeline_layouts::LIT_DECAL_PIPELINES,
                                &[camera, material, &self.layout, lights],
                            ),
                            None => crate::pipelines::pipeline_layout(
                                device,
                                &crate::pipeline_layouts::DECAL_PIPELINES,
                                &[camera, material, &self.layout],
                            ),
                        }
                    });
                let pipeline = pipeline(device, shader, layout, self.surface_format, key);
                self.pipelines.insert(key, pipeline);
            }
        }

        /// Draws the prepared decals over the opaque pass's color; call
        /// [`Self::ensure_pipelines`] first. `lights` is the local lights'
        /// group of lit frames.
        fn draw(
            &self,
            pass: &mut wgpu::RenderPass<'_>,
            camera: &wgpu::BindGroup,
            materials: &MaterialStore,
            lights: Option<&wgpu::BindGroup>,
            post: bool,
        ) {
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(
                2,
                self.groups[self.parity].get().expect("prepared this frame"),
                &[],
            );
            if let Some(lights) = lights {
                pass.set_bind_group(3, lights, &[]);
            }
            let mut bound = None;
            for run in &self.runs {
                let key = (
                    !post,
                    run.blend,
                    lights.is_some() && run.blend == DecalBlend::Alpha,
                );
                if bound != Some(key) {
                    pass.set_pipeline(&self.pipelines[&key]);
                    bound = Some(key);
                }
                pass.set_bind_group(1, materials.binding(run.material).0, &[]);
                pass.draw(0..36, run.records.clone());
            }
        }
    }

    impl RenderFeature for DecalPass {
        type Frame = DecalFrame;
        type Stats = Drawn;

        /// Decals draw over the opaque surfaces whose depth they sample,
        /// shading local lights in frames binning them.
        fn declare(_frame: &DecalFrame, shape: &FrameShape, graph: &mut RenderGraph) {
            graph
                .add(Pass::Feature(FeaturePass::Decals))
                .read(Resource::Depth)
                .read_if(shape.lights, Resource::Lights)
                .color(shape.scene(), Access::Load);
        }

        fn encode(
            &self,
            _frame: &DecalFrame,
            node: &Node,
            encode: &Encode<'_>,
            encoder: &mut wgpu::CommandEncoder,
        ) -> u32 {
            let mut pass = encode.render_pass(
                encoder,
                node,
                "engine/decals",
                encode.profiler.span("decals").map(Span::render),
            );
            self.draw(
                &mut pass,
                encode.camera,
                encode.materials,
                encode.lights,
                encode.targets != Targets::Canvas,
            );
            // Their draws are not the frame's mesh draws.
            0
        }

        fn capacity_bytes(&self) -> u64 {
            self.records.size()
        }

        /// Decals drawn, invalid ones, and those left out: the snapshot's
        /// and those beyond the GPU's binding.
        fn stats(
            _this: Option<&Self>,
            frame: Option<&DecalFrame>,
            snapshot: &RenderSnapshot,
        ) -> Drawn {
            Drawn {
                drawn: frame.map_or(0, |frame| frame.drawn),
                invalid: snapshot.decals.invalid.min(u32::MAX as usize) as u32,
                dropped: (snapshot.decals.dropped as u64
                    + frame.map_or(0, |frame| u64::from(frame.dropped)))
                .min(u64::from(u32::MAX)) as u32,
            }
        }
    }

    fn pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        surface_format: wgpu::TextureFormat,
        (canvas, blend, lit): PipelineKey,
    ) -> wgpu::RenderPipeline {
        let flag = |on: bool| if on { 1.0 } else { 0.0 };
        let constants = [
            ("DECAL_MULTIPLY", flag(blend == DecalBlend::Multiply)),
            ("DECAL_CANVAS", flag(canvas)),
        ];
        // Alpha keeps the scene target's ambient share.
        let keep = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let color = match blend {
            DecalBlend::Alpha => wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            DecalBlend::Multiply => wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Dst,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let format = if canvas {
            surface_format
        } else {
            crate::post::SCENE_FORMAT
        };
        crate::pipelines::geometry(
            device,
            &crate::pipelines::Geometry {
                label: "engine/decals",
                layout,
                module: shader,
                vertex_constants: &[],
                vertex: crate::entries::DECAL_VERTEX,
                buffers: &[],
                // The box's far faces cover its pixels once, from inside too.
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Front),
                    ..Default::default()
                },
                depth: None,
                fragment: Some(crate::pipelines::Fragment {
                    entry: crate::entries::decal_fragment(lit),
                    constants: &constants,
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState { color, alpha: keep }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
            },
        )
    }
}

pub(crate) use gpu::{DecalFrame, DecalPass};
