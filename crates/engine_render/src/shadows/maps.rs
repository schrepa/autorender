//! The shadow maps (decisions 020 and 028) and what the main passes bind to
//! look them up: the near cascades as layers of one texture, the moving and
//! kept far maps, the lookup uniform and the samplers, with placeholders
//! while shadows are off.

use crate::gpu_data::ShadowUniform;

use crate::pipelines::{sampler, view};

pub(crate) const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth16Unorm;

/// One resolution's maps.
pub(super) struct MapTextures {
    pub(super) resolution: u32,
    cascades: u32,
    // The near cascades as layers of one texture, and the moving and kept
    // far maps.
    textures: [wgpu::Texture; 3],
    // The near cascades as one array (lookups) and each layer (drawing).
    near_array: wgpu::TextureView,
    pub(super) near_layers: Vec<wgpu::TextureView>,
    pub(super) moving: wgpu::TextureView,
    pub(super) kept: wgpu::TextureView,
}

impl Drop for MapTextures {
    fn drop(&mut self) {
        // Browsers otherwise keep the memory until garbage collection;
        // submitted work that still reads them completes first.
        for texture in &self.textures {
            texture.destroy();
        }
    }
}

impl MapTextures {
    /// Maps of `resolution` texels (the near cascades half as many) with
    /// `cascades` near layers.
    fn new(device: &wgpu::Device, resolution: u32, cascades: u32) -> Self {
        let texture = |label, size, layers| {
            crate::targets::texture_2d(
                device,
                label,
                (size, size),
                layers,
                1,
                SHADOW_FORMAT,
                crate::targets::RENDER_TARGET,
            )
        };
        let textures = [
            texture("engine/shadow-near", resolution / 2, cascades),
            texture("engine/shadow-moving", resolution, 1),
            texture("engine/shadow-kept", resolution, 1),
        ];
        let near_array = textures[0].create_view(&wgpu::TextureViewDescriptor {
            label: Some("engine/shadow-near-cascades"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let near_layers = (0..cascades)
            .map(|layer| {
                textures[0].create_view(&wgpu::TextureViewDescriptor {
                    label: Some("engine/shadow-near-cascade"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let moving = textures[1].create_view(&Default::default());
        let kept = textures[2].create_view(&Default::default());
        Self {
            resolution,
            cascades,
            textures,
            near_array,
            near_layers,
            moving,
            kept,
        }
    }
}

pub(super) struct ShadowMaps {
    textures: Option<MapTextures>,
    // Advances when the views `entries` binds change: bindings of the main
    // passes follow.
    generation: u64,
    uniform: wgpu::Buffer,
    written: Option<ShadowUniform>,
    sampler: wgpu::Sampler,
    // Reads the four texels around a receiver-plane lookup (decision 028).
    texel_sampler: wgpu::Sampler,
    _placeholder: wgpu::Texture,
    placeholder_view: wgpu::TextureView,
    placeholder_array: wgpu::TextureView,
}

impl ShadowMaps {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let uniform = crate::buffers::uniform(
            device,
            "engine/shadow-uniform",
            size_of::<ShadowUniform>() as u64,
        );
        // Comparisons filter four texels.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/shadow-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let texel_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/shadow-texel-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let placeholder = crate::targets::texture_2d(
            device,
            "engine/shadow-placeholder",
            (1, 1),
            1,
            1,
            SHADOW_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let placeholder_view = placeholder.create_view(&Default::default());
        let placeholder_array = placeholder.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Self {
            textures: None,
            generation: 0,
            uniform,
            written: None,
            sampler,
            texel_sampler,
            _placeholder: placeholder,
            placeholder_view,
            placeholder_array,
        }
    }

    /// Advances when the map views bound by [`Self::entries`] change.
    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    /// The maps, once shadows ran.
    pub(super) fn textures(&self) -> Option<&MapTextures> {
        self.textures.as_ref()
    }

    /// Bindings of the main passes' first group (`bindings::CAMERA`): the
    /// lookup uniform, the near cascades, the kept and moving maps, the
    /// comparison sampler and the texel sampler of receiver planes.
    pub(super) fn entries(&self) -> [(&'static str, wgpu::BindingResource<'_>); 6] {
        let (near, kept, moving) = match &self.textures {
            Some(maps) => (&maps.near_array, &maps.kept, &maps.moving),
            None => (
                &self.placeholder_array,
                &self.placeholder_view,
                &self.placeholder_view,
            ),
        };
        [
            ("shadow", self.uniform.as_entire_binding()),
            ("near_shadows", view(near)),
            ("kept_shadows", view(kept)),
            ("moving_shadows", view(moving)),
            ("shadow_sampler", sampler(&self.sampler)),
            ("shadow_texels", sampler(&self.texel_sampler)),
        ]
    }

    /// Bytes of the maps.
    pub(super) fn capacity_bytes(&self) -> u64 {
        self.textures.as_ref().map_or(0, |maps| {
            let far = u64::from(maps.resolution) * u64::from(maps.resolution) * 2;
            2 * far + u64::from(maps.cascades) * far / 4
        })
    }

    /// Creates maps of `resolution` texels with `cascades` near layers
    /// unless they exist; returns whether it did (new maps hold nothing).
    pub(super) fn ensure(&mut self, device: &wgpu::Device, resolution: u32, cascades: u32) -> bool {
        if self
            .textures
            .as_ref()
            .is_some_and(|maps| maps.resolution == resolution && maps.cascades == cascades)
        {
            return false;
        }
        self.textures = Some(MapTextures::new(device, resolution, cascades));
        self.generation += 1;
        true
    }

    /// Releases the maps while shadows are off.
    pub(super) fn release(&mut self) {
        if self.textures.take().is_some() {
            self.generation += 1;
        }
    }

    /// Writes the lookup uniform when it changed.
    pub(super) fn write_uniform(&mut self, queue: &wgpu::Queue, uniform: ShadowUniform) {
        if self.written != Some(uniform) {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
            self.written = Some(uniform);
        }
    }
}

/// A depth pass into `view`, loading or clearing it as `load` says.
pub(super) fn map_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<f32>,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("engine/shadow-map"),
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}
