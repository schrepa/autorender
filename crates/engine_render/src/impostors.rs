//! Octahedral impostors (decision 048): the first frame an opaque draw group
//! whose mesh has enough triangles draws, its mesh is baked with its material
//! from `IMPOSTOR_FRAMES` x `IMPOSTOR_FRAMES` directions over the sphere into
//! one layer of four texture arrays, and GPU-presented frames draw instances
//! too small for the layer's texels to show as cards blending the three
//! frames around the view (`meshes.wgsl`). The GPU side is created with the
//! first bake.

use bytemuck::{Pod, Zeroable};
use engine_core::glam::Vec3;

/// Frames per side of the octahedral grid, texels across a frame of the
/// first level, and levels, as `meshes.wgsl` declares them.
pub(crate) const IMPOSTOR_FRAMES: u32 = 8;
pub(crate) const IMPOSTOR_TEXELS: u32 = 16;
pub(crate) const IMPOSTOR_LEVELS: u32 = 4;

/// WGSL `ImpostorRecord` of `meshes.wgsl`: 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub(crate) struct ImpostorRecord {
    /// Mesh-space center and radius of the sphere every frame frames.
    pub sphere: [f32; 4],
    /// Mesh-space half extents of the mesh's bounds.
    pub extent: [f32; 3],
    pub layer: u32,
    /// The material's emissive color (glTF), linear; w unused.
    pub emissive: [f32; 4],
}

impl ImpostorRecord {
    /// The record of a mesh with bounds `min` to `max` baked into `layer`,
    /// whose material emits `emissive`.
    pub(crate) fn new(min: Vec3, max: Vec3, layer: u32, emissive: [f32; 3]) -> Self {
        let center = (min + max) * 0.5;
        let half = (max - min) * 0.5;
        // A hair over the half diagonal keeps every vertex inside the frames,
        // and flat bounds still get a radius.
        let radius = engine_core::math::max(half.length() * 1.0001, 1.0e-6);
        let [r, g, b] = emissive;
        Self {
            sphere: center.extend(radius).to_array(),
            extent: half.to_array(),
            layer,
            emissive: [r, g, b, 0.0],
        }
    }

    /// The impostor's error in mesh units: a texel of its first level, which
    /// the card resolves no finer; 0 for the zeroed records of draw groups
    /// without an impostor.
    pub(crate) fn error(&self) -> f32 {
        2.0 * self.sphere[3] / IMPOSTOR_TEXELS as f32
    }
}

/// WGSL `ImpostorBake` uniform of `meshes.wgsl`: 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct ImpostorBakeUniform {
    pub sphere: [f32; 4],
}

/// Layers of the texture arrays: free ones are reused lowest first, and the
/// arrays grow by doubling up to `limit` layers.
#[derive(Debug, Default)]
pub(crate) struct Layers {
    free: Vec<u32>,
    used: u32,
}

impl Layers {
    /// A layer for a new impostor, or `None` when `limit` layers are in use.
    pub(crate) fn allocate(&mut self, limit: u32) -> Option<u32> {
        if let Some(index) = self
            .free
            .iter()
            .enumerate()
            .min_by_key(|(_, layer)| **layer)
            .map(|(index, _)| index)
        {
            return Some(self.free.swap_remove(index));
        }
        (self.used < limit).then(|| {
            self.used += 1;
            self.used - 1
        })
    }

    pub(crate) fn release(&mut self, layer: u32) {
        self.free.push(layer);
    }

    /// Layers the arrays must hold: every layer ever handed out.
    pub(crate) fn extent(&self) -> u32 {
        self.used
    }

    /// Layers holding impostors.
    pub(crate) fn live(&self) -> u32 {
        self.used - self.free.len() as u32
    }
}

/// The capacity arrays grow to for `needed` layers: powers of two from 16,
/// never beyond `limit`.
pub(crate) fn grown_layers(needed: u32, limit: u32) -> u32 {
    needed.max(16).next_power_of_two().min(limit).max(needed)
}

#[cfg(target_arch = "wasm32")]
mod gpu {
    use std::collections::HashMap;

    use engine_core::{Material, MaterialHandle, MeshHandle};

    use super::*;
    use crate::gpu_data::Variant;
    use crate::mesh_arena::MeshDraw;
    use crate::mipmaps::MipGenerator;

    /// Texels across a layer's first level.
    const IMPOSTOR_SIZE: u32 = IMPOSTOR_FRAMES * IMPOSTOR_TEXELS;
    /// Fewest triangles a mesh needs for an impostor: a card's pixels cost
    /// about what fewer triangles would.
    pub(crate) const IMPOSTOR_MIN_TRIANGLES: usize = 64;

    /// What an impostor depicts: a mesh drawn with a material.
    pub(crate) type ImpostorKey = (MeshHandle, Option<MaterialHandle>);

    /// Albedo and coverage, emissive color and mask (both decode sRGB), then
    /// normal and depth, and roughness, metalness and occlusion.
    const FORMATS: [wgpu::TextureFormat; 4] = [
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba8Unorm,
    ];
    /// Bakes rasterize four samples a texel; their resolve averages
    /// fragments with the cleared zeros, premultiplying coverage.
    const SAMPLES: u32 = 4;
    const BAKE_DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    struct Arrays {
        textures: [wgpu::Texture; 4],
        views: [wgpu::TextureView; 4],
        layers: u32,
    }

    /// Multisampled targets every bake renders into before resolving to its
    /// layer.
    struct BakeTargets {
        colors: [wgpu::TextureView; 4],
        depth: wgpu::TextureView,
        _textures: [wgpu::Texture; 5],
    }

    struct Baked {
        record: ImpostorRecord,
        // The material baked, to rebake one replaced in place (decision 033).
        material: Option<Material>,
    }

    /// What a bake draws: the mesh as the arena holds it, with its
    /// material's group and pipeline variant.
    pub(crate) struct BakeSource<'a> {
        pub draw: MeshDraw<'a>,
        pub extended: bool,
        pub group: &'a wgpu::BindGroup,
        pub variant: Variant,
        pub bounds: (Vec3, Vec3),
        pub material: Option<Material>,
    }

    /// The impostor draw pipelines' targets: the canvas or the screen
    /// effects' targets, temporal targets, and whether they shade local
    /// lights.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub(crate) enum ImpostorTargets {
        Canvas,
        Post,
        Temporal,
    }

    pub(crate) struct ImpostorAtlas {
        arrays: Option<Arrays>,
        targets: Option<BakeTargets>,
        layers: Layers,
        baked: HashMap<ImpostorKey, Baked>,
        bake_layout: wgpu::BindGroupLayout,
        bake_pipeline_layout: wgpu::PipelineLayout,
        bake_pipelines: HashMap<Variant, wgpu::RenderPipeline>,
        draw_layout: wgpu::BindGroupLayout,
        draw_pipelines: HashMap<(ImpostorTargets, bool), wgpu::RenderPipeline>,
        sampler: wgpu::Sampler,
        // One record per plan batch as last uploaded, and the group binding
        // them with the arrays, keyed by the generations of both.
        records: wgpu::Buffer,
        uploaded: Vec<ImpostorRecord>,
        group: Option<((u64, u64), wgpu::BindGroup)>,
        records_generation: u64,
        arrays_generation: u64,
        // Advances whenever an impostor is baked or dropped: batch records
        // carrying impostor errors are rebuilt then.
        generation: u64,
    }

    fn record_buffer(device: &wgpu::Device, records: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/impostor-records"),
            size: (records.max(1) * size_of::<ImpostorRecord>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    impl ImpostorAtlas {
        /// `material_layout` binds a material's maps (group 1 of bakes).
        pub(crate) fn new(
            device: &wgpu::Device,
            material_layout: &wgpu::BindGroupLayout,
            empty_layout: &wgpu::BindGroupLayout,
        ) -> Self {
            let bake_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("engine/impostor-bake-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 20,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            size_of::<ImpostorBakeUniform>() as u64
                        ),
                    },
                    count: None,
                }],
            });
            let bake_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("engine/impostor-bake-pipeline-layout"),
                    bind_group_layouts: &[
                        Some(empty_layout),
                        Some(material_layout),
                        Some(&bake_layout),
                    ],
                    immediate_size: 0,
                });
            let array = |binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            };
            let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("engine/impostor-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                size_of::<ImpostorRecord>() as u64
                            ),
                        },
                        count: None,
                    },
                    array(12),
                    array(13),
                    array(14),
                    array(15),
                    wgpu::BindGroupLayoutEntry {
                        binding: 16,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/impostor-sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            });
            Self {
                arrays: None,
                targets: None,
                layers: Layers::default(),
                baked: HashMap::new(),
                bake_layout,
                bake_pipeline_layout,
                bake_pipelines: HashMap::new(),
                draw_layout,
                draw_pipelines: HashMap::new(),
                sampler,
                records: record_buffer(device, 1),
                uploaded: Vec::new(),
                group: None,
                records_generation: 0,
                arrays_generation: 0,
                generation: 0,
            }
        }

        /// The layout of the impostor draws' group 1.
        pub(crate) fn draw_layout(&self) -> &wgpu::BindGroupLayout {
            &self.draw_layout
        }

        /// Advances whenever an impostor is baked or dropped.
        pub(crate) fn generation(&self) -> u64 {
            self.generation
        }

        /// Impostors baked.
        pub(crate) fn live(&self) -> u32 {
            self.layers.live()
        }

        /// GPU bytes of the arrays and records.
        pub(crate) fn capacity_bytes(&self) -> u64 {
            let layers = self.arrays.as_ref().map_or(0, |arrays| arrays.layers);
            // Four bytes a texel in four arrays, a third more for the levels.
            let texels = u64::from(IMPOSTOR_SIZE * IMPOSTOR_SIZE) * 4 * 4 / 3;
            u64::from(layers) * texels * 4 + self.records.size()
        }

        /// The impostor of `key`, when baked for `material`.
        pub(crate) fn record(
            &self,
            key: ImpostorKey,
            material: Option<&Material>,
        ) -> Option<ImpostorRecord> {
            self.baked
                .get(&key)
                .filter(|baked| baked.material.as_ref() == material)
                .map(|baked| baked.record)
        }

        /// Drops impostors `keep` refuses and those whose material changed
        /// from `material` (decision 033), freeing their layers.
        pub(crate) fn retain(
            &mut self,
            mut keep: impl FnMut(&ImpostorKey) -> bool,
            mut material: impl FnMut(MaterialHandle) -> Option<Material>,
        ) {
            let layers = &mut self.layers;
            let before = self.baked.len();
            self.baked.retain(|key, baked| {
                let current = key.1.and_then(&mut material);
                let kept = keep(key) && baked.material == current;
                if !kept {
                    layers.release(baked.record.layer);
                }
                kept
            });
            if self.baked.len() != before {
                self.generation += 1;
            }
        }

        /// Drops every impostor when more than `budget` are baked: the draw
        /// groups first in the plan bake again up to it.
        pub(crate) fn enforce_budget(&mut self, budget: Option<u32>) {
            if budget.is_some_and(|budget| self.layers.live() > budget) {
                self.baked.clear();
                self.layers = Layers::default();
                self.generation += 1;
            }
        }

        /// Records the bake of `key` from `source` into `encoder`, and the
        /// levels below its first. Returns whether it was baked: not when
        /// `budget` impostors (at most the device's array layers) already
        /// are.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn bake(
            &mut self,
            device: &wgpu::Device,
            encoder: &mut wgpu::CommandEncoder,
            mips: &MipGenerator,
            shader: &wgpu::ShaderModule,
            key: ImpostorKey,
            source: &BakeSource<'_>,
            budget: Option<u32>,
        ) -> bool {
            if self.baked.contains_key(&key) {
                return true;
            }
            let limit = device
                .limits()
                .max_texture_array_layers
                .min(budget.unwrap_or(u32::MAX));
            if self.layers.live() >= limit {
                return false;
            }
            let Some(layer) = self
                .layers
                .allocate(device.limits().max_texture_array_layers)
            else {
                return false;
            };
            self.ensure_layers(device, encoder, self.layers.extent());
            let targets = self.targets.get_or_insert_with(|| bake_targets(device));
            // Bakes are rare: uber variants (decision 049).
            let variant = source.variant.with_extended(source.extended).uber();
            let pipeline = self.bake_pipelines.entry(variant).or_insert_with(|| {
                bake_pipeline(device, shader, &self.bake_pipeline_layout, variant)
            });
            let emissive = source
                .material
                .as_ref()
                .map_or([0.0; 3], Material::emissive);
            let record = ImpostorRecord::new(source.bounds.0, source.bounds.1, layer, emissive);
            let uniform = wgpu::util::DeviceExt::create_buffer_init(
                device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("engine/impostor-bake"),
                    contents: bytemuck::bytes_of(&ImpostorBakeUniform {
                        sphere: record.sphere,
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                },
            );
            let uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine/impostor-bake-group"),
                layout: &self.bake_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 20,
                    resource: uniform.as_entire_binding(),
                }],
            });
            let arrays = self.arrays.as_ref().expect("grown above");
            let resolves: Vec<wgpu::TextureView> = arrays
                .textures
                .iter()
                .map(|texture| layer_view(texture, layer, 0))
                .collect();
            {
                let attachments: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> = targets
                    .colors
                    .iter()
                    .zip(&resolves)
                    .map(|(view, resolve)| {
                        Some(wgpu::RenderPassColorAttachment {
                            view,
                            depth_slice: None,
                            resolve_target: Some(resolve),
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Discard,
                            },
                        })
                    })
                    .collect();
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("engine/impostor-bake"),
                    color_attachments: &attachments,
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &targets.depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(1, source.group, &[]);
                pass.set_bind_group(2, &uniform_group, &[]);
                pass.set_vertex_buffer(0, source.draw.vertices.slice(..));
                if source.extended
                    && let Some(extras) = source.draw.extras
                {
                    pass.set_vertex_buffer(1, extras.slice(..));
                }
                pass.set_index_buffer(source.draw.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(
                    source.draw.first_index..source.draw.first_index + source.draw.index_count,
                    source.draw.base_vertex,
                    0..IMPOSTOR_FRAMES * IMPOSTOR_FRAMES,
                );
            }
            // Premultiplied texels average into the levels below: a frame's
            // cell halves with each level and never mixes with another's.
            for texture in &arrays.textures {
                mips.record_layer(device, encoder, texture, layer);
            }
            self.baked.insert(
                key,
                Baked {
                    record,
                    material: source.material,
                },
            );
            self.generation += 1;
            true
        }

        /// Grows the arrays to hold `layers`, copying the baked ones.
        fn ensure_layers(
            &mut self,
            device: &wgpu::Device,
            encoder: &mut wgpu::CommandEncoder,
            layers: u32,
        ) {
            if self
                .arrays
                .as_ref()
                .is_some_and(|arrays| arrays.layers >= layers)
            {
                return;
            }
            let capacity = grown_layers(layers, device.limits().max_texture_array_layers);
            let textures = FORMATS.map(|format| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("engine/impostor-array"),
                    size: wgpu::Extent3d {
                        width: IMPOSTOR_SIZE,
                        height: IMPOSTOR_SIZE,
                        depth_or_array_layers: capacity,
                    },
                    mip_level_count: IMPOSTOR_LEVELS,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                })
            });
            if let Some(old) = &self.arrays {
                for (from, to) in old.textures.iter().zip(&textures) {
                    for level in 0..IMPOSTOR_LEVELS {
                        let size = IMPOSTOR_SIZE >> level;
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo {
                                texture: from,
                                mip_level: level,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::TexelCopyTextureInfo {
                                texture: to,
                                mip_level: level,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::Extent3d {
                                width: size,
                                height: size,
                                depth_or_array_layers: old.layers,
                            },
                        );
                    }
                }
            }
            let views = textures.each_ref().map(|texture| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("engine/impostor-array-view"),
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    ..Default::default()
                })
            });
            self.arrays = Some(Arrays {
                textures,
                views,
                layers: capacity,
            });
            self.arrays_generation += 1;
        }

        /// Uploads `records`, one per plan batch (zeroed for batches without
        /// an impostor), when they changed, and builds the draws' group;
        /// returns whether there is one, not before any bake.
        pub(crate) fn prepare(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            records: &[ImpostorRecord],
        ) -> bool {
            let Some(arrays) = self.arrays.as_ref() else {
                return false;
            };
            if records != self.uploaded.as_slice() {
                let bytes = size_of_val(records) as u64;
                if bytes > self.records.size() {
                    self.records = record_buffer(device, records.len().next_power_of_two());
                    self.records_generation += 1;
                }
                if !records.is_empty() {
                    queue.write_buffer(&self.records, 0, bytemuck::cast_slice(records));
                }
                self.uploaded.clear();
                self.uploaded.extend_from_slice(records);
            }
            let key = (self.records_generation, self.arrays_generation);
            if self.group.as_ref().is_none_or(|(built, _)| *built != key) {
                let view = |binding, view| wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::TextureView(view),
                };
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("engine/impostor-group"),
                    layout: &self.draw_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 11,
                            resource: self.records.as_entire_binding(),
                        },
                        view(12, &arrays.views[0]),
                        view(13, &arrays.views[1]),
                        view(14, &arrays.views[2]),
                        view(15, &arrays.views[3]),
                        wgpu::BindGroupEntry {
                            binding: 16,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });
                self.group = Some((key, group));
            }
            true
        }

        /// The draws' group, after [`Self::prepare`] built it.
        pub(crate) fn group(&self) -> Option<&wgpu::BindGroup> {
            self.group.as_ref().map(|(_, group)| group)
        }

        /// The draw pipeline into `targets`, shading local lights when `lit`,
        /// once inserted.
        pub(crate) fn pipeline(
            &self,
            targets: ImpostorTargets,
            lit: bool,
        ) -> Option<&wgpu::RenderPipeline> {
            self.draw_pipelines.get(&(targets, lit))
        }

        pub(crate) fn insert_pipeline(
            &mut self,
            targets: ImpostorTargets,
            lit: bool,
            pipeline: wgpu::RenderPipeline,
        ) {
            self.draw_pipelines.insert((targets, lit), pipeline);
        }
    }

    fn layer_view(texture: &wgpu::Texture, layer: u32, level: u32) -> wgpu::TextureView {
        texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("engine/impostor-layer"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_mip_level: level,
            mip_level_count: Some(1),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        })
    }

    fn bake_targets(device: &wgpu::Device) -> BakeTargets {
        let target = |format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("engine/impostor-bake-target"),
                size: wgpu::Extent3d {
                    width: IMPOSTOR_SIZE,
                    height: IMPOSTOR_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: SAMPLES,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        };
        let textures = [
            target(FORMATS[0]),
            target(FORMATS[1]),
            target(FORMATS[2]),
            target(FORMATS[3]),
            target(BAKE_DEPTH),
        ];
        let view = |texture: &wgpu::Texture| texture.create_view(&Default::default());
        BakeTargets {
            colors: [
                view(&textures[0]),
                view(&textures[1]),
                view(&textures[2]),
                view(&textures[3]),
            ],
            depth: view(&textures[4]),
            _textures: textures,
        }
    }

    fn bake_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        variant: Variant,
    ) -> wgpu::RenderPipeline {
        let constants = variant.constants();
        let extended = variant.extended();
        let suffix = if extended { "_extended" } else { "" };
        let vertex = format!("vs_impostor_bake{suffix}");
        let fragment = format!("fs_impostor_bake{suffix}");
        let mut buffers = vec![crate::browser::vertex_layout()];
        if extended {
            buffers.push(crate::browser::extra_layout());
        }
        let targets = FORMATS.map(|format| Some(wgpu::ColorTargetState::from(format)));
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("engine/impostor-bake"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some(&vertex),
                compilation_options: Default::default(),
                buffers: &buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: (!variant.double_sided()).then_some(wgpu::Face::Back),
                ..Default::default()
            },
            // Nearer the frame's viewer is greater, as with reversed depth.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: BAKE_DEPTH,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: SAMPLES,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(&fragment),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                targets: &targets,
            }),
            multiview_mask: None,
            cache: None,
        })
    }

    /// An impostor draw pipeline: `layout` binds the camera, the impostors'
    /// group, the poses (with previous poses on temporal frames) and, when
    /// `lit`, the local lights. `canvas` is the surface's format.
    pub(crate) fn draw_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        targets: ImpostorTargets,
        lit: bool,
        canvas: wgpu::TextureFormat,
        depth: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let (vertex, fragment) = match (targets, lit) {
            (ImpostorTargets::Canvas, false) => ("vs_impostor", "fs_impostor"),
            (ImpostorTargets::Canvas, true) => ("vs_impostor", "fs_impostor_lit"),
            (ImpostorTargets::Post, false) => ("vs_impostor", "fs_impostor_post"),
            (ImpostorTargets::Post, true) => ("vs_impostor", "fs_impostor_post_lit"),
            (ImpostorTargets::Temporal, false) => ("vs_impostor_temporal", "fs_impostor_temporal"),
            (ImpostorTargets::Temporal, true) => {
                ("vs_impostor_temporal", "fs_impostor_temporal_lit")
            }
        };
        let color_targets: Vec<Option<wgpu::ColorTargetState>> = match targets {
            ImpostorTargets::Canvas => vec![Some(canvas.into())],
            ImpostorTargets::Post => vec![
                Some(crate::post::SCENE_FORMAT.into()),
                Some(crate::post::GLOW_FORMAT.into()),
                Some(crate::post::SURFACE_FORMAT.into()),
            ],
            ImpostorTargets::Temporal => vec![
                Some(crate::post::SCENE_FORMAT.into()),
                Some(crate::temporal::MOTION_FORMAT.into()),
                Some(crate::post::GLOW_FORMAT.into()),
                Some(crate::post::SURFACE_FORMAT.into()),
            ],
        };
        let constants = [("SURFACE_IS_SRGB", if canvas.is_srgb() { 1.0 } else { 0.0 })];
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("engine/impostors"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some(vertex),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![2 => Uint32, 3 => Uint32],
                })],
            },
            // Mirrored poses turn cards around.
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(fragment),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: if targets == ImpostorTargets::Canvas {
                        &constants
                    } else {
                        &[]
                    },
                    ..Default::default()
                },
                targets: &color_targets,
            }),
            multiview_mask: None,
            cache: None,
        })
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use gpu::{
    BakeSource, IMPOSTOR_MIN_TRIANGLES, ImpostorAtlas, ImpostorTargets, draw_pipeline,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_frame_the_bounds_sphere_and_carry_texel_errors() {
        let record = ImpostorRecord::new(
            Vec3::new(-1.0, 0.0, -2.0),
            Vec3::new(3.0, 2.0, 2.0),
            5,
            [0.5, 0.25, 1.0],
        );
        assert_eq!(&record.sphere[..3], &[1.0, 1.0, 0.0]);
        assert_eq!(record.extent, [2.0, 1.0, 2.0]);
        let radius = record.sphere[3];
        assert!((3.0..3.001).contains(&radius), "{radius}");
        assert_eq!(record.layer, 5);
        assert_eq!(record.emissive, [0.5, 0.25, 1.0, 0.0]);
        assert_eq!(record.error(), 2.0 * radius / IMPOSTOR_TEXELS as f32);
        // A point mesh still frames a sphere; groups without one have no
        // error.
        assert!(ImpostorRecord::new(Vec3::ONE, Vec3::ONE, 0, [0.0; 3]).sphere[3] > 0.0);
        assert_eq!(ImpostorRecord::default().error(), 0.0);
    }

    #[test]
    fn layers_reuse_the_lowest_free_one_and_stop_at_the_limit() {
        let mut layers = Layers::default();
        assert_eq!(layers.allocate(3), Some(0));
        assert_eq!(layers.allocate(3), Some(1));
        assert_eq!(layers.allocate(3), Some(2));
        assert_eq!(layers.allocate(3), None);
        assert_eq!((layers.extent(), layers.live()), (3, 3));
        layers.release(2);
        layers.release(0);
        assert_eq!(layers.live(), 1);
        assert_eq!(layers.allocate(3), Some(0));
        assert_eq!(layers.allocate(3), Some(2));
        assert_eq!(layers.extent(), 3);
    }

    #[test]
    fn arrays_grow_by_powers_of_two_within_the_limit() {
        assert_eq!(grown_layers(1, 256), 16);
        assert_eq!(grown_layers(17, 256), 32);
        assert_eq!(grown_layers(200, 256), 256);
        assert_eq!(grown_layers(20, 24), 24);
    }
}
