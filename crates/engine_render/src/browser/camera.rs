//! The main passes' first group: the camera and lighting uniforms, shadow
//! lookups (decisions 020 and 028), what glass refracts (decision 038) and
//! what surfaces reflect (decision 043), rebuilt when the shadow maps,
//! environment maps or reflection probe grid it binds are replaced.

use super::*;

pub(super) struct CameraBindings {
    pub(super) layout: wgpu::BindGroupLayout,
    pub(super) buffer: wgpu::Buffer,
    lighting_buffer: wgpu::Buffer,
    // The scene's atmosphere (decision 018), written when it changes.
    lighting: Option<LightingUniform>,
    // The group, for the shadow maps, environment maps and probe grid of
    // these generations.
    pub(super) group: wgpu::BindGroup,
    pub(super) key: CameraGroupsKey,
    // Refraction through glass (decision 038): the white texel bound where
    // no opaque color is, its sampler, and the group of transparent passes
    // over the opaque color of post target generation `.0`.
    refraction_fallback: wgpu::TextureView,
    refraction_sampler: wgpu::Sampler,
    refraction_group: Cached<(u64, CameraGroupsKey), wgpu::BindGroup>,
    // Each reflection probe face's group (decision 043), built with the
    // main group's key.
    probe_groups: Vec<wgpu::BindGroup>,
}

impl CameraBindings {
    /// The layout, the camera uniform and the first group, over
    /// `lighting_buffer` and the maps of `shadows`, `environment` and
    /// `probes`.
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lighting_buffer: wgpu::Buffer,
        shadows: &ShadowPass,
        environment: &Environment,
        probes: &ProbePass,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/camera-uniform"),
            size: size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_entry = |binding, visibility, size: usize| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size as u64),
            },
            count: None,
        };
        let shadow_map_entry = |binding, view_dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/camera-layout"),
            entries: &[
                // Fragments read the eye, sky rays and the jitter.
                uniform_entry(
                    0,
                    wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    size_of::<CameraUniform>(),
                ),
                uniform_entry(
                    1,
                    wgpu::ShaderStages::FRAGMENT,
                    size_of::<LightingUniform>(),
                ),
                // Shadow lookups (decisions 020 and 028): the near cascades,
                // then the kept and moving maps.
                uniform_entry(2, wgpu::ShaderStages::FRAGMENT, size_of::<ShadowUniform>()),
                shadow_map_entry(3, wgpu::TextureViewDimension::D2Array),
                shadow_map_entry(4, wgpu::TextureViewDimension::D2),
                shadow_map_entry(5, wgpu::TextureViewDimension::D2),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                // Receiver-plane lookups read four texels (decision 028).
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // What glass refracts (decision 038) and its sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // The environment maps surfaces reflect (decision 043) and
                // their sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::CubeArray,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // The grid of reflection probes each cell's surfaces read.
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let refraction_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engine/refraction-fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            refraction_texture.as_image_copy(),
            &[255; 4],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let refraction_fallback =
            refraction_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let refraction_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/refraction-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let group = camera_group(
            device,
            &layout,
            (&buffer, &lighting_buffer),
            shadows,
            (&refraction_fallback, &refraction_sampler),
            (environment.binding(), probes.words().0),
        );
        Self {
            layout,
            buffer,
            lighting_buffer,
            lighting: None,
            group,
            key: (shadows.generation(), 0, 0),
            refraction_fallback,
            refraction_sampler,
            refraction_group: Cached::default(),
            probe_groups: Vec::new(),
        }
    }

    /// Writes `lighting` unless already written; returns the bytes written.
    pub(super) fn write_lighting(&mut self, queue: &wgpu::Queue, lighting: LightingUniform) -> u64 {
        if self.lighting == Some(lighting) {
            return 0;
        }
        queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&lighting));
        self.lighting = Some(lighting);
        size_of::<LightingUniform>() as u64
    }

    /// Rebuilds the group when the shadow maps, environment maps or probe
    /// grid it binds were replaced, and forgets the probe faces' groups.
    pub(super) fn refresh(
        &mut self,
        device: &wgpu::Device,
        shadows: &ShadowPass,
        environment: &Environment,
        probes: &ProbePass,
    ) {
        let key = (
            shadows.generation(),
            environment.generation(),
            probes.words().1,
        );
        if key == self.key {
            return;
        }
        self.key = key;
        self.group = camera_group(
            device,
            &self.layout,
            (&self.buffer, &self.lighting_buffer),
            shadows,
            (&self.refraction_fallback, &self.refraction_sampler),
            (environment.binding(), probes.words().0),
        );
        self.probe_groups.clear();
    }

    /// The group of transparent passes when glass refracts (decision 038):
    /// the main group over `scene`, the opaque color of post target
    /// generation `generation`.
    pub(super) fn prepare_refraction(
        &mut self,
        device: &wgpu::Device,
        (scene, generation): (&wgpu::TextureView, u64),
        shadows: &ShadowPass,
        environment: &Environment,
        probes: &ProbePass,
    ) {
        let key = (generation, self.key);
        if self.refraction_group.stale(&key) {
            let group = camera_group(
                device,
                &self.layout,
                (&self.buffer, &self.lighting_buffer),
                shadows,
                (scene, &self.refraction_sampler),
                (environment.binding(), probes.words().0),
            );
            self.refraction_group.set(key, group);
        }
    }

    /// The group glass refracting the opaque color binds, once prepared.
    pub(super) fn refraction_group(&self) -> Option<&wgpu::BindGroup> {
        self.refraction_group.get()
    }

    /// Each reflection probe face's group (decision 043): the main group
    /// over the face's camera uniform in `probes`.
    pub(super) fn prepare_probe_groups(
        &mut self,
        device: &wgpu::Device,
        shadows: &ShadowPass,
        environment: &Environment,
        probes: &ProbePass,
    ) {
        if self.probe_groups.is_empty() {
            self.probe_groups = (0..FACES)
                .map(|face| {
                    camera_group(
                        device,
                        &self.layout,
                        (probes.camera(face), &self.lighting_buffer),
                        shadows,
                        (&self.refraction_fallback, &self.refraction_sampler),
                        (environment.binding(), probes.words().0),
                    )
                })
                .collect();
        }
    }

    /// Reflection probe face `face`'s group, once prepared.
    pub(super) fn probe_group(&self, face: usize) -> &wgpu::BindGroup {
        &self.probe_groups[face]
    }
}

/// The main passes' first group over the `camera` and `lighting` uniforms:
/// shadow lookups, what glass refracts (decision 038) with its sampler, and
/// what surfaces reflect (decision 043): the environment maps with their
/// sampler and the reflection probes' grid.
fn camera_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    (camera, lighting): (&wgpu::Buffer, &wgpu::Buffer),
    shadows: &ShadowPass,
    (refraction, refraction_sampler): (&wgpu::TextureView, &wgpu::Sampler),
    ((environment, environment_sampler), probe_words): (
        (&wgpu::TextureView, &wgpu::Sampler),
        &wgpu::Buffer,
    ),
) -> wgpu::BindGroup {
    let [shadow, near, kept, moving, sampler, texel_sampler] = shadows.entries();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("engine/camera-bind-group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: lighting.as_entire_binding(),
            },
            shadow,
            near,
            kept,
            moving,
            sampler,
            texel_sampler,
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(refraction),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::Sampler(refraction_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::TextureView(environment),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: wgpu::BindingResource::Sampler(environment_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: probe_words.as_entire_binding(),
            },
        ],
    })
}
