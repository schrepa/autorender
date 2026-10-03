//! Decals (decision 039): one record per decal and the pass drawing their
//! boxes after the opaque meshes. Everything here is created with the first
//! frame that has decals.

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use engine_core::{DecalBlend, MaterialHandle, PoseRows, RenderDecal};

/// `DecalRecord::pose` of decals drawn from their own matrix.
pub(crate) const NO_POSE: u32 = u32::MAX;

/// WGSL `DecalRecord` of `meshes/decals.wgsl`: 96 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct DecalData {
    /// The box's world matrix, unless `pose[0]` names its palette entry.
    pub model: PoseRows,
    pub color: [f32; 4],
    /// Fade cosine, roughness, metalness and emission.
    pub params: [f32; 4],
    pub pose: [u32; 4],
}

impl DecalData {
    /// `palette`: whether the frame's joint palette holds decals' poses.
    pub(crate) fn new(decal: &RenderDecal, palette: bool) -> Self {
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
            pose: [decal.joint.filter(|_| palette).unwrap_or(NO_POSE), 0, 0, 0],
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
    palette: bool,
    room: usize,
    records: &mut Vec<DecalData>,
    runs: &mut Vec<DecalRun>,
) -> usize {
    records.clear();
    runs.clear();
    let fitting = &decals[..decals.len().min(room)];
    for decal in fitting {
        let index = records.len() as u32;
        records.push(DecalData::new(decal, palette));
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

#[cfg(target_arch = "wasm32")]
mod gpu {
    use std::collections::HashMap;

    use engine_core::RenderSnapshot;

    use super::*;
    use crate::cached::Cached;
    use crate::materials::MaterialStore;

    /// A frame's decals: those drawn and those beyond the GPU's binding.
    pub(crate) struct DecalFrame {
        pub drawn: u32,
        pub dropped: u32,
    }

    /// Pipelines by target (canvas or scene target), blend and whether they
    /// shade local lights.
    type PipelineKey = (bool, DecalBlend, bool);

    pub(crate) struct DecalPass {
        layout: wgpu::BindGroupLayout,
        // Without and with the local lights' group.
        pipeline_layouts: [Option<wgpu::PipelineLayout>; 2],
        pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,
        surface_format: wgpu::TextureFormat,
        records: wgpu::Buffer,
        staging: Vec<DecalData>,
        uploaded: Vec<DecalData>,
        runs: Vec<DecalRun>,
        // Keyed by the depth buffer, record buffer and palette they bind.
        group: Cached<(u64, u64, u64, usize), wgpu::BindGroup>,
        buffer_generation: u64,
    }

    fn storage(device: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/decal-records"),
            size: bytes.max(16).next_multiple_of(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    impl DecalPass {
        pub(crate) fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
            let read = |binding, visibility| wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
            // Group 2 beside the mesh poses' bindings: the joint palette
            // (decision 036) holds retained frames' decal poses.
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("engine/decal-layout"),
                entries: &[
                    read(3, wgpu::ShaderStages::VERTEX),
                    read(8, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
            Self {
                layout,
                pipeline_layouts: [None, None],
                pipelines: HashMap::new(),
                surface_format,
                records: storage(device, size_of::<DecalData>() as u64),
                staging: Vec::new(),
                uploaded: Vec::new(),
                runs: Vec::new(),
                group: Cached::default(),
                buffer_generation: 0,
            }
        }

        pub(crate) fn capacity_bytes(&self) -> u64 {
            self.records.size()
        }

        /// Uploads `snapshot`'s decals when they changed; `None` when the
        /// frame draws none. With `palette` (the joint palette of retained
        /// frames and its pose generation and parity), decals read their
        /// displayed poses there. `depth` is the depth buffer of
        /// `depth_generation`.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn prepare(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            snapshot: &RenderSnapshot,
            palette: (&wgpu::Buffer, Option<(u64, usize)>),
            depth: &wgpu::TextureView,
            depth_generation: u64,
        ) -> Option<DecalFrame> {
            if !snapshot.decal_settings.enabled || snapshot.decals.is_empty() {
                return None;
            }
            let limits = device.limits();
            let binding = limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size);
            let room = (binding / size_of::<DecalData>() as u64) as usize;
            let (palette, pose_key) = palette;
            let dropped = pack_decals(
                &snapshot.decals,
                pose_key.is_some(),
                room,
                &mut self.staging,
                &mut self.runs,
            );
            if self.staging.is_empty() {
                return None;
            }
            let bytes = (self.staging.len() * size_of::<DecalData>()) as u64;
            if bytes > self.records.size() {
                self.records = storage(device, bytes.next_power_of_two());
                self.uploaded.clear();
                self.buffer_generation += 1;
            }
            if self.staging != self.uploaded {
                queue.write_buffer(&self.records, 0, bytemuck::cast_slice(&self.staging));
                self.uploaded.clone_from(&self.staging);
            }
            let (generation, parity) = pose_key.unwrap_or((u64::MAX, 0));
            let key = (depth_generation, self.buffer_generation, generation, parity);
            if self.group.stale(&key) {
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("engine/decal-group"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: palette.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: self.records.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: wgpu::BindingResource::TextureView(depth),
                        },
                    ],
                });
                self.group.set(key, group);
            }
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
            groups: [&wgpu::BindGroupLayout; 2],
            lights: Option<&wgpu::BindGroupLayout>,
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
                        let mut layouts = vec![Some(camera), Some(material), Some(&self.layout)];
                        if shades_lights {
                            layouts.push(lights);
                        }
                        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("engine/decal-pipeline-layout"),
                            bind_group_layouts: &layouts,
                            immediate_size: 0,
                        })
                    });
                let pipeline = pipeline(device, shader, layout, self.surface_format, key);
                self.pipelines.insert(key, pipeline);
            }
        }

        /// Draws the prepared decals over the opaque pass's color; call
        /// [`Self::ensure_pipelines`] first. `lights` is the local lights'
        /// group of lit frames.
        pub(crate) fn draw(
            &self,
            pass: &mut wgpu::RenderPass<'_>,
            camera: &wgpu::BindGroup,
            materials: &MaterialStore,
            lights: Option<&wgpu::BindGroup>,
            post: bool,
        ) {
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(2, self.group.get().expect("prepared this frame"), &[]);
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
            ("SURFACE_IS_SRGB", flag(surface_format.is_srgb())),
        ];
        // Alpha keeps the scene target's ambient share (decision 021).
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
        crate::pipelines::render(
            device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("engine/decals"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_decal"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                // The box's far faces cover its pixels once, from inside too.
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Front),
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some(if lit { "fs_decal_lit" } else { "fs_decal" }),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState { color, alpha: keep }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            },
        )
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use gpu::{DecalFrame, DecalPass};

#[cfg(test)]
mod tests {
    use engine_core::{MaterialHandle, Surface, glam::Mat4};

    use super::*;

    fn decal(material: Option<MaterialHandle>, blend: DecalBlend, joint: u32) -> RenderDecal {
        RenderDecal {
            model: Mat4::from_translation(engine_core::glam::Vec3::new(1.0, 2.0, 3.0)),
            joint: Some(joint),
            material,
            color: [1.0, 0.5, 0.25, 0.75],
            surface: Surface::default(),
            blend,
            fade: 0.5,
        }
    }

    #[test]
    fn records_carry_rows_color_and_palette_entries() {
        let record = DecalData::new(&decal(None, DecalBlend::Alpha, 7), true);
        assert_eq!(record.model[0], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(record.model[1], [0.0, 1.0, 0.0, 2.0]);
        assert_eq!(record.model[2], [0.0, 0.0, 1.0, 3.0]);
        assert_eq!(record.color, [1.0, 0.5, 0.25, 0.75]);
        assert_eq!(record.params[0], 0.5);
        assert_eq!(record.pose[0], 7);
        // Owned snapshots draw their own matrices.
        let owned = DecalData::new(&decal(None, DecalBlend::Alpha, 7), false);
        assert_eq!(owned.pose[0], NO_POSE);
    }

    #[test]
    fn runs_group_consecutive_decals_and_room_drops_the_rest() {
        let mut library = engine_core::AssetLibrary::default();
        let first = Some(
            library
                .insert_material(engine_core::Material::new([1.0; 4]).unwrap())
                .unwrap(),
        );
        let decals = [
            decal(first, DecalBlend::Alpha, 0),
            decal(first, DecalBlend::Alpha, 1),
            decal(first, DecalBlend::Multiply, 2),
            decal(None, DecalBlend::Multiply, 3),
            decal(None, DecalBlend::Multiply, 4),
        ];
        let (mut records, mut runs) = (Vec::new(), Vec::new());
        assert_eq!(pack_decals(&decals, true, 16, &mut records, &mut runs), 0);
        assert_eq!(records.len(), 5);
        let ranges: Vec<_> = runs.iter().map(|run| run.records.clone()).collect();
        assert_eq!(ranges, [0..2, 2..3, 3..5]);
        assert_eq!(pack_decals(&decals, true, 2, &mut records, &mut runs), 3);
        assert_eq!(records.len(), 2);
        assert_eq!(runs.len(), 1);
    }
}
