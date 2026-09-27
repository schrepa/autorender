//! Screen effects (decision 021): ambient occlusion and bloom.
//!
//! Frames with an effect draw the scene into a color target whose alpha is the
//! ambient share of each pixel's light, and a glow target of each pixel's
//! exposed brightness above the bloom threshold. Ambient occlusion samples the
//! depth buffer at half resolution; bloom filters the glowing light down a
//! chain of half-resolution levels and back up. Temporally antialiased frames
//! (decision 017) apply both in their resolve, other frames in a composite
//! that writes the canvas.

use engine_core::{RenderSnapshot, glam::Mat4};

use crate::gpu_data::{POST_SHADER, PostUniform};

/// Linear color with 8 perceptual bits; alpha holds the ambient share.
pub(crate) const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
pub(crate) const GLOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
const OCCLUSION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
const BLOOM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Half-resolution bloom levels; each halves the one before.
const BLOOM_LEVELS: u32 = 6;

/// Which effects a frame applies.
#[derive(Clone, Copy)]
pub(crate) struct PostFrame {
    pub occlusion: bool,
    pub bloom: bool,
}

struct Targets {
    size: (u32, u32),
    // Scene, glow, occlusion and bloom, destroyed with the targets.
    textures: [wgpu::Texture; 4],
    scene: wgpu::TextureView,
    glow: wgpu::TextureView,
    occlusion: wgpu::TextureView,
    // One view per bloom level, and the level count.
    bloom: Vec<wgpu::TextureView>,
    bloom_first: wgpu::BindGroup,
    // Group `i` reads level `i` (down, writing `i + 1`) or level `i + 1`
    // (up, adding into `i`).
    bloom_down: Vec<wgpu::BindGroup>,
    bloom_up: Vec<wgpu::BindGroup>,
    composite: wgpu::BindGroup,
}

impl Drop for Targets {
    fn drop(&mut self) {
        for texture in &self.textures {
            texture.destroy();
        }
    }
}

pub(crate) struct PostPass {
    occlusion: wgpu::RenderPipeline,
    bloom_first: wgpu::RenderPipeline,
    bloom_down: wgpu::RenderPipeline,
    bloom_up: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    written: Option<PostUniform>,
    sampler: wgpu::Sampler,
    targets: Option<Targets>,
    // Occlusion bindings over the depth buffer of that generation.
    occlusion_group: Option<(u64, u64, wgpu::BindGroup)>,
    // Advances when targets are replaced: the temporal resolve rebinds.
    generation: u64,
}

impl PostPass {
    pub(crate) fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/post.wgsl"),
            source: wgpu::ShaderSource::Wgsl(POST_SHADER.into()),
        });
        let pipeline =
            |label, entry: &str, format: wgpu::TextureFormat, blend, srgb: Option<bool>| {
                let constants =
                    srgb.map(|srgb| [("SURFACE_IS_SRGB", if srgb { 1.0 } else { 0.0 })]);
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: None,
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_fullscreen"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: constants.as_ref().map_or(&[], |constants| &constants[..]),
                            ..Default::default()
                        },
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
            };
        let add = Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/post-uniform"),
            size: size_of::<PostUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/post-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            occlusion: pipeline(
                "engine/post-occlusion",
                "fs_occlusion",
                OCCLUSION_FORMAT,
                None,
                None,
            ),
            bloom_first: pipeline(
                "engine/post-bloom-first",
                "fs_bloom_first",
                BLOOM_FORMAT,
                None,
                None,
            ),
            bloom_down: pipeline(
                "engine/post-bloom-down",
                "fs_bloom_down",
                BLOOM_FORMAT,
                None,
                None,
            ),
            bloom_up: pipeline(
                "engine/post-bloom-up",
                "fs_bloom_up",
                BLOOM_FORMAT,
                add,
                None,
            ),
            composite: pipeline(
                "engine/post-composite",
                "fs_composite",
                surface_format,
                None,
                Some(surface_format.is_srgb()),
            ),
            uniform,
            written: None,
            sampler,
            targets: None,
            occlusion_group: None,
            generation: 0,
        }
    }

    /// Advances when targets are replaced.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// Releases the targets while no frame uses an effect; the next such
    /// frame recreates them.
    pub(crate) fn release_targets(&mut self) {
        self.targets = None;
        self.occlusion_group = None;
    }

    /// Target bytes: scene (4) and glow (1) per pixel, occlusion at half
    /// resolution and the bloom chain (8 bytes per half-resolution texel).
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.targets.as_ref().map_or(0, |targets| {
            let pixels = u64::from(targets.size.0) * u64::from(targets.size.1);
            pixels * 5 + pixels / 4 + pixels / 4 * 8 * 4 / 3
        })
    }

    /// Sizes the targets to `size` and writes this frame's settings: the
    /// effects of `snapshot` over the depth buffer of `depth_generation`.
    pub(crate) fn begin(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        snapshot: &RenderSnapshot,
        size: (u32, u32),
        depth: &wgpu::TextureView,
        depth_generation: u64,
    ) -> PostFrame {
        if self
            .targets
            .as_ref()
            .is_none_or(|targets| targets.size != size)
        {
            self.targets = None;
            self.targets = Some(self.create_targets(device, size));
            self.generation += 1;
        }
        let occlusion = snapshot.ambient_occlusion.sanitized();
        let bloom = snapshot.bloom.sanitized();
        let frame = PostFrame {
            occlusion: occlusion.enabled(),
            bloom: bloom.enabled(),
        };
        if frame.occlusion
            && self
                .occlusion_group
                .as_ref()
                .is_none_or(|(depth, targets, _)| {
                    *depth != depth_generation || *targets != self.generation
                })
        {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine/post-occlusion-group"),
                layout: &self.occlusion.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(depth),
                    },
                ],
            });
            self.occlusion_group = Some((depth_generation, self.generation, group));
        }
        let (width, height) = (size.0 as f32, size.1 as f32);
        let view_projection = snapshot.view_projection;
        // Row y of a perspective view-projection is the focal scale times a
        // unit axis of the view: its length converts world units at unit
        // distance into half the viewport height.
        let focal = view_projection.row(1).truncate().length();
        let half = ((size.0 / 2).max(1) as f32, (size.1 / 2).max(1) as f32);
        let uniform = PostUniform {
            inverse_view_projection: inverse_or_identity(view_projection).to_cols_array_2d(),
            eye: snapshot
                .camera_position
                .extend(focal * height * 0.5)
                .to_array(),
            size: [width, height, 1.0 / width, 1.0 / height],
            occlusion: [
                occlusion.radius,
                occlusion.intensity,
                0.0,
                if frame.occlusion { 1.0 } else { 0.0 },
            ],
            bloom: [
                bloom.intensity,
                if frame.bloom { 1.0 } else { 0.0 },
                1.0 / half.0,
                1.0 / half.1,
            ],
        };
        if self.written != Some(uniform) {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
            self.written = Some(uniform);
        }
        frame
    }

    /// The scene color and glow targets of the opaque pass.
    pub(crate) fn targets(&self) -> (&wgpu::TextureView, &wgpu::TextureView) {
        let targets = self.targets.as_ref().expect("sized in begin");
        (&targets.scene, &targets.glow)
    }

    /// Views the temporal resolve reads: scene, occlusion and bloom, and the
    /// settings uniform.
    pub(crate) fn resolve_inputs(
        &self,
    ) -> (
        &wgpu::TextureView,
        &wgpu::TextureView,
        &wgpu::TextureView,
        &wgpu::Buffer,
    ) {
        let targets = self.targets.as_ref().expect("sized in begin");
        (
            &targets.scene,
            &targets.occlusion,
            &targets.bloom[0],
            &self.uniform,
        )
    }

    /// Records ambient occlusion and bloom after the opaque pass. With
    /// `timestamps` (a query set and its first index), occlusion writes
    /// queries `first` and `first + 1`, bloom `first + 2` and `first + 3`.
    pub(crate) fn encode_effects(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame: PostFrame,
        timestamps: Option<(&wgpu::QuerySet, u32)>,
    ) {
        let targets = self.targets.as_ref().expect("sized in begin");
        if frame.occlusion
            && let Some((_, _, group)) = &self.occlusion_group
        {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-occlusion",
                &targets.occlusion,
                wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                timestamps.map(|(set, first)| (set, Some(first), Some(first + 1))),
            );
            pass.set_pipeline(&self.occlusion);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
        if frame.bloom {
            let levels = targets.bloom.len();
            let timed = |first: bool, last: bool| {
                timestamps.map(|(set, index)| {
                    (set, first.then_some(index + 2), last.then_some(index + 3))
                })
            };
            {
                let mut pass = fullscreen_pass(
                    encoder,
                    "engine/post-bloom",
                    &targets.bloom[0],
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    timed(true, levels == 1),
                );
                pass.set_pipeline(&self.bloom_first);
                pass.set_bind_group(0, &targets.bloom_first, &[]);
                pass.draw(0..3, 0..1);
            }
            for level in 1..levels {
                let mut pass = fullscreen_pass(
                    encoder,
                    "engine/post-bloom",
                    &targets.bloom[level],
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    None,
                );
                pass.set_pipeline(&self.bloom_down);
                pass.set_bind_group(0, &targets.bloom_down[level - 1], &[]);
                pass.draw(0..3, 0..1);
            }
            for level in (0..levels - 1).rev() {
                let mut pass = fullscreen_pass(
                    encoder,
                    "engine/post-bloom",
                    &targets.bloom[level],
                    wgpu::LoadOp::Load,
                    timed(false, level == 0),
                );
                pass.set_pipeline(&self.bloom_up);
                pass.set_bind_group(0, &targets.bloom_up[level], &[]);
                pass.draw(0..3, 0..1);
            }
        }
    }

    /// Records the composite of a frame without temporal antialiasing into
    /// `canvas`.
    pub(crate) fn encode_composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        canvas: &wgpu::TextureView,
        timestamps: Option<(&wgpu::QuerySet, u32)>,
    ) {
        let targets = self.targets.as_ref().expect("sized in begin");
        let mut pass = fullscreen_pass(
            encoder,
            "engine/post-composite",
            canvas,
            wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            timestamps.map(|(set, first)| (set, Some(first), Some(first + 1))),
        );
        pass.set_pipeline(&self.composite);
        pass.set_bind_group(0, &targets.composite, &[]);
        pass.draw(0..3, 0..1);
    }

    fn create_targets(&self, device: &wgpu::Device, (width, height): (u32, u32)) -> Targets {
        let half = ((width / 2).max(1), (height / 2).max(1));
        let texture = |label, (width, height): (u32, u32), format, levels| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let levels = BLOOM_LEVELS.min(crate::gpu_data::mip_levels(half.0, half.1));
        let textures = [
            texture("engine/post-scene", (width, height), SCENE_FORMAT, 1),
            texture("engine/post-glow", (width, height), GLOW_FORMAT, 1),
            texture("engine/post-occlusion", half, OCCLUSION_FORMAT, 1),
            texture("engine/post-bloom", half, BLOOM_FORMAT, levels),
        ];
        let [scene, glow, occlusion, _] = textures
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
        let bloom: Vec<wgpu::TextureView> = (0..levels)
            .map(|level| {
                textures[3].create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let uniform = wgpu::BindGroupEntry {
            binding: 0,
            resource: self.uniform.as_entire_binding(),
        };
        let sampler = wgpu::BindGroupEntry {
            binding: 7,
            resource: wgpu::BindingResource::Sampler(&self.sampler),
        };
        let group = |pipeline: &wgpu::RenderPipeline, entries: &[wgpu::BindGroupEntry<'_>]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine/post-group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries,
            })
        };
        let bloom_first = group(
            &self.bloom_first,
            &[
                uniform.clone(),
                texture_entry(2, &scene),
                texture_entry(3, &glow),
                sampler.clone(),
            ],
        );
        let bloom_down = (0..bloom.len().saturating_sub(1))
            .map(|level| {
                group(
                    &self.bloom_down,
                    &[texture_entry(4, &bloom[level]), sampler.clone()],
                )
            })
            .collect();
        let bloom_up = (0..bloom.len().saturating_sub(1))
            .map(|level| {
                group(
                    &self.bloom_up,
                    &[texture_entry(4, &bloom[level + 1]), sampler.clone()],
                )
            })
            .collect();
        let composite = group(
            &self.composite,
            &[
                uniform.clone(),
                texture_entry(2, &scene),
                texture_entry(5, &occlusion),
                texture_entry(6, &bloom[0]),
                sampler.clone(),
            ],
        );
        Targets {
            size: (width, height),
            textures,
            scene,
            glow,
            occlusion,
            bloom,
            bloom_first,
            bloom_down,
            bloom_up,
            composite,
        }
    }
}

fn texture_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn inverse_or_identity(matrix: Mat4) -> Mat4 {
    let inverse = matrix.inverse();
    if inverse.is_finite() {
        inverse
    } else {
        Mat4::IDENTITY
    }
}

fn fullscreen_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    label: &str,
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
    timestamps: Option<(&wgpu::QuerySet, Option<u32>, Option<u32>)>,
) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: timestamps.and_then(|(query_set, begin, end)| {
            (begin.is_some() || end.is_some()).then_some(wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: begin,
                end_of_pass_write_index: end,
            })
        }),
        occlusion_query_set: None,
        multiview_mask: None,
    })
}
