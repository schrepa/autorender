//! Octahedral impostors: the first frame an opaque draw group
//! whose mesh has enough triangles draws, its mesh is baked with its material
//! from `IMPOSTOR_FRAMES` x `IMPOSTOR_FRAMES` directions into one layer of
//! four texture arrays, and GPU-presented frames draw instances too small
//! for the layer's texels to show as cards blending the three frames around
//! the view (`meshes/impostors.wgsl`). Plan batches' frames cover the
//! sphere; placed objects', which stand on the ground, cover
//! the upper hemisphere at four times the texels in an atlas of their own.
//! Bakes also record how much of the sky each texel's
//! surface sees past its own mesh, which shades the sun on cards as the
//! mesh's own shadows would. The GPU side is created with
//! the first bake.

#![expect(
    clippy::expect_used,
    reason = "the atlas arrays grow before they are read"
)]

pub(crate) use crate::gpu_layouts::constants::IMPOSTOR_FRAMES;
pub(crate) use crate::gpu_layouts::constants::IMPOSTOR_HEMISPHERE as HEMISPHERE;
use engine_core::glam::Vec3;

/// How an atlas lays out its impostors: texels across a frame of the first
/// level, and whether its frames cover only the upper hemisphere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AtlasShape {
    pub texels: u32,
    pub hemisphere: bool,
}

impl AtlasShape {
    /// Plan batches' impostors: 16 texels a frame over the
    /// sphere.
    pub(crate) const SPHERE: Self = Self {
        texels: 16,
        hemisphere: false,
    };
    /// Placed objects': 64 texels a frame over the upper
    /// hemisphere, which views from below see from the horizon.
    pub(crate) const PLACED: Self = Self {
        texels: 64,
        hemisphere: true,
    };

    /// Texels across a layer's first level.
    pub(crate) fn size(self) -> u32 {
        IMPOSTOR_FRAMES * self.texels
    }

    /// Levels down to two texels a frame.
    pub(crate) fn levels(self) -> u32 {
        self.texels.trailing_zeros()
    }

    /// The flags of records of this shape.
    fn flags(self) -> u32 {
        self.texels | if self.hemisphere { HEMISPHERE } else { 0 }
    }
}

pub(crate) use crate::gpu_layouts::impostors::ImpostorRecord;

impl ImpostorRecord {
    /// The record of a mesh with bounds `min` to `max` baked into `layer` of
    /// an atlas of `shape`, whose material emits `emissive`.
    pub(crate) fn new(
        min: Vec3,
        max: Vec3,
        layer: u32,
        emissive: [f32; 3],
        shape: AtlasShape,
    ) -> Self {
        let center = (min + max) * 0.5;
        let half = (max - min) * 0.5;
        // A hair over the half diagonal keeps every vertex inside the frames,
        // and flat bounds still get a radius.
        let radius = engine_core::math::max(half.length() * 1.0001, 1.0e-6);
        Self {
            sphere: center.extend(radius).to_array(),
            extent: half.to_array(),
            layer,
            emissive,
            flags: shape.flags(),
        }
    }

    /// The impostor's error in mesh units: a texel of its first level, which
    /// the card resolves no finer; 0 for the zeroed records of draw groups
    /// without an impostor.
    pub(crate) fn error(&self) -> f32 {
        let texels = self.flags & (HEMISPHERE - 1);
        if texels == 0 {
            return 0.0;
        }
        2.0 * self.sphere[3] / texels as f32
    }
}

pub(crate) use crate::gpu_layouts::impostors::ImpostorBakeUniform;

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

mod gpu {
    use engine_core::collections::HashMap;

    use engine_core::{Material, MaterialHandle, MeshHandle};

    use super::*;
    use crate::bindings::{IMPOSTOR_BAKE, IMPOSTOR_DRAW};
    use crate::cached::Cached;
    use crate::gpu_data::Variant;
    use crate::mesh_arena::MeshDraw;
    use crate::mipmaps::MipGenerator;
    use crate::targets::Targets;

    /// Fewest triangles a mesh needs for an impostor: a card's pixels cost
    /// about what fewer triangles would.
    pub(crate) const IMPOSTOR_MIN_TRIANGLES: usize = 64;

    /// What an impostor depicts: a mesh drawn with a material.
    pub(crate) type ImpostorKey = (MeshHandle, Option<MaterialHandle>);

    /// Albedo and coverage, emissive color and mask (both decode sRGB), then
    /// normal and depth, and roughness, metalness, occlusion and the sky the
    /// surface sees past its mesh.
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
    /// The depth of every frame's nearest surface, which bakes look up to
    /// find the directions a surface sees past its mesh, has this many
    /// times the layer's texels across.
    const SHADE_SCALE: u32 = 4;

    struct Arrays {
        textures: [wgpu::Texture; 4],
        views: [wgpu::TextureView; 4],
        layers: u32,
    }

    /// Targets every bake renders into: the frames' nearest depths, then
    /// the multisampled colors resolving to its layer.
    struct BakeTargets {
        shade: wgpu::TextureView,
        colors: [wgpu::TextureView; 4],
        depth: wgpu::TextureView,
        _textures: [wgpu::Texture; 6],
    }

    struct Baked {
        record: ImpostorRecord,
        // The material baked, to rebake one replaced in place.
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

    /// What every atlas shares: the bakes' and draws' layouts and
    /// pipelines, and the draws' sampler.
    pub(crate) struct ImpostorPipelines {
        bake_layout: crate::pipelines::GroupLayout,
        bake_pipeline_layout: wgpu::PipelineLayout,
        bake_pipelines: HashMap<Variant, wgpu::RenderPipeline>,
        shade_pipelines: HashMap<Variant, wgpu::RenderPipeline>,
        draw_layout: crate::pipelines::GroupLayout,
        draw_pipelines: HashMap<(Targets, bool), wgpu::RenderPipeline>,
        sampler: wgpu::Sampler,
        // Stands in for the frames' depths while bakes draw them.
        no_shade: Option<wgpu::TextureView>,
    }

    impl ImpostorPipelines {
        pub(crate) fn invalidate_scene(&mut self) {
            self.draw_pipelines.clear();
        }

        /// `material_layout` binds a material's maps (group 1 of bakes).
        pub(crate) fn new(
            device: &wgpu::Device,
            material_layout: &crate::pipelines::GroupLayout,
            empty_layout: &crate::pipelines::GroupLayout,
        ) -> Self {
            let bake_layout = crate::pipelines::layout(device, &IMPOSTOR_BAKE);
            let bake_pipeline_layout = crate::pipelines::pipeline_layout(
                device,
                &crate::pipeline_layouts::IMPOSTOR_BAKE_PIPELINES,
                &[empty_layout, material_layout, &bake_layout],
            );
            let draw_layout = crate::pipelines::layout(device, &IMPOSTOR_DRAW);
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
                bake_layout,
                bake_pipeline_layout,
                bake_pipelines: HashMap::default(),
                shade_pipelines: HashMap::default(),
                draw_layout,
                draw_pipelines: HashMap::default(),
                sampler,
                no_shade: None,
            }
        }

        /// The layout of the impostor draws' group 1.
        pub(crate) fn draw_layout(&self) -> &crate::pipelines::GroupLayout {
            &self.draw_layout
        }

        /// The draw pipeline into `targets`, shading local lights when `lit`,
        /// once inserted.
        pub(crate) fn pipeline(
            &self,
            targets: Targets,
            lit: bool,
        ) -> Option<&wgpu::RenderPipeline> {
            self.draw_pipelines.get(&(targets, lit))
        }

        pub(crate) fn insert_pipeline(
            &mut self,
            targets: Targets,
            lit: bool,
            pipeline: wgpu::RenderPipeline,
        ) {
            self.draw_pipelines.insert((targets, lit), pipeline);
        }

        /// Drops the draw pipelines of `targets`, built over layouts of a
        /// pass that is gone.
        pub(crate) fn forget_pipelines(&mut self, targets: Targets) {
            self.draw_pipelines.retain(|(kind, _), _| *kind != targets);
        }
    }

    pub(crate) struct ImpostorAtlas {
        shape: AtlasShape,
        arrays: Option<Arrays>,
        targets: Option<BakeTargets>,
        layers: Layers,
        baked: HashMap<ImpostorKey, Baked>,
        // The records last uploaded, and the group binding them with the
        // arrays, keyed by the generations of both.
        records: wgpu::Buffer,
        uploaded: Vec<ImpostorRecord>,
        group: Cached<(u64, u64), wgpu::BindGroup>,
        records_generation: u64,
        arrays_generation: u64,
        // Advances whenever an impostor is baked or dropped: records carrying
        // impostor errors are rebuilt then.
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
        pub(crate) fn new(device: &wgpu::Device, shape: AtlasShape) -> Self {
            Self {
                shape,
                arrays: None,
                targets: None,
                layers: Layers::default(),
                baked: HashMap::default(),
                records: record_buffer(device, 1),
                uploaded: Vec::new(),
                group: Cached::default(),
                records_generation: 0,
                arrays_generation: 0,
                generation: 0,
            }
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
            let size = u64::from(self.shape.size());
            u64::from(layers) * size * size * 4 * 4 * 4 / 3 + self.records.size()
        }

        /// Whether the impostor of `key` is baked.
        pub(crate) fn contains(&self, key: ImpostorKey) -> bool {
            self.baked.contains_key(&key)
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
        /// from `material`, freeing their layers.
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
        /// groups first in order bake again up to it.
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
        #[expect(clippy::too_many_arguments, reason = "a pass's disjoint borrows")]
        #[expect(
            clippy::too_many_lines,
            reason = "longer than the line ceiling; it may only shrink"
        )]
        pub(crate) fn bake(
            &mut self,
            device: &wgpu::Device,
            encoder: &mut wgpu::CommandEncoder,
            mips: &MipGenerator,
            shader: &wgpu::ShaderModule,
            pipelines: &mut ImpostorPipelines,
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
            let shape = self.shape;
            let targets = self
                .targets
                .get_or_insert_with(|| bake_targets(device, shape));
            // Bakes are rare: uber variants.
            let variant = source.variant.with_extended(source.extended).uber();
            let layout = &pipelines.bake_pipeline_layout;
            let pipeline = pipelines
                .bake_pipelines
                .entry(variant)
                .or_insert_with(|| bake_pipeline(device, shader, layout, variant, false));
            let shade_pipeline = pipelines
                .shade_pipelines
                .entry(variant)
                .or_insert_with(|| bake_pipeline(device, shader, layout, variant, true));
            let emissive = source
                .material
                .as_ref()
                .map_or([0.0; 3], Material::emissive);
            let record =
                ImpostorRecord::new(source.bounds.0, source.bounds.1, layer, emissive, shape);
            let uniform = wgpu::util::DeviceExt::create_buffer_init(
                device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("engine/impostor-bake"),
                    contents: bytemuck::bytes_of(&ImpostorBakeUniform {
                        sphere: record.sphere,
                        flags: [record.flags, 0, 0, 0],
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                },
            );
            let no_shade = pipelines.no_shade.get_or_insert_with(|| {
                bake_texture(
                    device,
                    1,
                    1,
                    BAKE_DEPTH,
                    wgpu::TextureUsages::TEXTURE_BINDING,
                )
                .create_view(&Default::default())
            });
            let group = |shade: &wgpu::TextureView| {
                crate::pipelines::group(
                    device,
                    "engine/impostor-bake-group",
                    &IMPOSTOR_BAKE,
                    &pipelines.bake_layout,
                    &[
                        ("impostor_bake", uniform.as_entire_binding()),
                        ("impostor_shade", crate::pipelines::view(shade)),
                    ],
                )
            };
            let (shade_group, color_group) = (group(no_shade), group(&targets.shade));
            let draw = |pass: &mut wgpu::RenderPass<'_>| {
                pass.set_bind_group(1, source.group, &[]);
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
            };
            // Every frame's nearest depths first, which the colors' bake
            // looks up.
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("engine/impostor-shade"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &targets.shade,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(shade_pipeline);
                pass.set_bind_group(2, &shade_group, &[]);
                draw(&mut pass);
            }
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
                pass.set_bind_group(2, &color_group, &[]);
                draw(&mut pass);
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

        /// Frees the targets bakes render into until the next bake: a large
        /// atlas's would otherwise stay resident.
        pub(crate) fn release_targets(&mut self) {
            self.targets = None;
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
            let (size, levels) = (self.shape.size(), self.shape.levels());
            let textures = FORMATS.map(|format| {
                crate::targets::texture_2d(
                    device,
                    "engine/impostor-array",
                    (size, size),
                    capacity,
                    levels,
                    format,
                    wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                )
            });
            if let Some(old) = &self.arrays {
                for (from, to) in old.textures.iter().zip(&textures) {
                    for level in 0..levels {
                        let size = size >> level;
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

        /// Uploads `records` (zeroed for draw groups without an impostor)
        /// when they changed, and builds the draws' group; returns whether
        /// there is one, not before any bake.
        pub(crate) fn prepare(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            pipelines: &ImpostorPipelines,
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
            if self.group.stale(&key) {
                let group = crate::pipelines::group(
                    device,
                    "engine/impostor-group",
                    &IMPOSTOR_DRAW,
                    &pipelines.draw_layout,
                    &[
                        ("impostor_records", self.records.as_entire_binding()),
                        ("impostor_albedo", crate::pipelines::view(&arrays.views[0])),
                        (
                            "impostor_emissive",
                            crate::pipelines::view(&arrays.views[1]),
                        ),
                        (
                            "impostor_geometry",
                            crate::pipelines::view(&arrays.views[2]),
                        ),
                        ("impostor_surface", crate::pipelines::view(&arrays.views[3])),
                        (
                            "impostor_sampler",
                            crate::pipelines::sampler(&pipelines.sampler),
                        ),
                    ],
                );
                self.group.set(key, group);
            }
            true
        }

        /// The draws' group, after [`Self::prepare`] built it.
        pub(crate) fn group(&self) -> Option<&wgpu::BindGroup> {
            self.group.get()
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

    fn bake_texture(
        device: &wgpu::Device,
        size: u32,
        samples: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engine/impostor-bake-target"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    fn bake_targets(device: &wgpu::Device, shape: AtlasShape) -> BakeTargets {
        let size = shape.size();
        let target = |format| {
            bake_texture(
                device,
                size,
                SAMPLES,
                format,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            )
        };
        let textures = [
            bake_texture(
                device,
                size * SHADE_SCALE,
                1,
                BAKE_DEPTH,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            target(FORMATS[0]),
            target(FORMATS[1]),
            target(FORMATS[2]),
            target(FORMATS[3]),
            target(BAKE_DEPTH),
        ];
        let view = |texture: &wgpu::Texture| texture.create_view(&Default::default());
        BakeTargets {
            shade: view(&textures[0]),
            colors: [
                view(&textures[1]),
                view(&textures[2]),
                view(&textures[3]),
                view(&textures[4]),
            ],
            depth: view(&textures[5]),
            _textures: textures,
        }
    }

    /// A bake pipeline of `variant`: the colors', or with `shade` the
    /// frames' nearest depths alone.
    fn bake_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        variant: Variant,
        shade: bool,
    ) -> wgpu::RenderPipeline {
        let constants = variant.constants();
        let extended = variant.extended();
        let (vertex, fragment) = crate::entries::impostor_bake(extended, shade);
        let mut buffers = vec![crate::renderer::vertex_layout()];
        if extended {
            buffers.push(crate::renderer::extra_layout());
        }
        let targets = if shade {
            Vec::new()
        } else {
            FORMATS
                .map(|format| Some(wgpu::ColorTargetState::from(format)))
                .to_vec()
        };
        crate::pipelines::render(
            device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("engine/impostor-bake"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some(vertex.name),
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
                    count: if shade { 1 } else { SAMPLES },
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some(fragment.name),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            },
        )
    }

    /// An impostor draw pipeline: `layout` binds the camera, the impostors'
    /// group, the poses (with previous poses on temporal frames) and, when
    /// `lit`, the local lights. `canvas` is the surface's format.
    pub(crate) fn draw_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        targets: Targets,
        lit: bool,
        canvas: wgpu::TextureFormat,
        depth: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let vertex = crate::entries::impostor_vertex(targets);
        let fragment = crate::entries::impostor_fragment(targets, lit);
        let color_targets = targets.color_targets(canvas);
        crate::pipelines::geometry(
            device,
            &crate::pipelines::Geometry {
                label: "engine/impostors",
                layout,
                module: shader,
                vertex_constants: &[],
                vertex,
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![2 => Uint32, 3 => Uint32],
                })],
                // Mirrored poses turn cards around.
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth: Some(wgpu::DepthStencilState {
                    format: depth,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                fragment: Some(crate::pipelines::Fragment {
                    entry: fragment,
                    constants: &[],
                    targets: &color_targets,
                }),
            },
        )
    }
}

pub(crate) use gpu::{
    BakeSource, IMPOSTOR_MIN_TRIANGLES, ImpostorAtlas, ImpostorPipelines, draw_pipeline,
};
