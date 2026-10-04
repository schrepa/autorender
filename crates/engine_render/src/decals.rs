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

#[cfg(target_arch = "wasm32")]
mod gpu {
    use std::collections::HashMap;

    use engine_core::RenderSnapshot;

    use super::*;
    use crate::bindings::DECALS;
    use crate::buffers::{ensure_storage, storage};
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

    impl DecalPass {
        pub(crate) fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
            // Group 2 beside the mesh poses' bindings: the joint palette
            // (decision 036) holds retained frames' decal poses.
            let layout = crate::pipelines::layout(device, &DECALS);
            Self {
                layout,
                pipeline_layouts: [None, None],
                pipelines: HashMap::new(),
                surface_format,
                records: storage(
                    device,
                    "engine/decal-records",
                    size_of::<DecalData>() as u64,
                    wgpu::BufferUsages::empty(),
                ),
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
        /// frame draws none. Decals read their displayed poses from
        /// `palette` (the frame's joint palette and its pose generation and
        /// parity). `depth` is the depth buffer of `depth_generation`.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn prepare(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            snapshot: &RenderSnapshot,
            palette: (&wgpu::Buffer, (u64, usize)),
            depth: &wgpu::TextureView,
            depth_generation: u64,
        ) -> Option<DecalFrame> {
            if !snapshot.environment.decals.enabled || snapshot.decals.is_empty() {
                return None;
            }
            let limits = device.limits();
            let binding = limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size);
            let room = (binding / size_of::<DecalData>() as u64) as usize;
            let (palette, (generation, parity)) = palette;
            let dropped = pack_decals(&snapshot.decals, room, &mut self.staging, &mut self.runs);
            if self.staging.is_empty() {
                return None;
            }
            let bytes = (self.staging.len() * size_of::<DecalData>()) as u64;
            if ensure_storage(
                device,
                &mut self.records,
                "engine/decal-records",
                bytes,
                wgpu::BufferUsages::empty(),
            ) {
                self.uploaded.clear();
                self.buffer_generation += 1;
            }
            if self.staging != self.uploaded {
                queue.write_buffer(&self.records, 0, bytemuck::cast_slice(&self.staging));
                self.uploaded.clone_from(&self.staging);
            }
            let key = (depth_generation, self.buffer_generation, generation, parity);
            if self.group.stale(&key) {
                let group = crate::pipelines::group(
                    device,
                    "engine/decal-group",
                    &DECALS,
                    &self.layout,
                    &[
                        ("palette", palette.as_entire_binding()),
                        ("decal_records", self.records.as_entire_binding()),
                        ("decal_depth", crate::pipelines::view(depth)),
                    ],
                );
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
        crate::pipelines::geometry(
            device,
            &crate::pipelines::Geometry {
                label: "engine/decals",
                layout,
                module: shader,
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

#[cfg(target_arch = "wasm32")]
pub(crate) use gpu::{DecalFrame, DecalPass};
