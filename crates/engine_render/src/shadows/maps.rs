//! The shadow maps and what the main passes
//! bind to look them up: the near cascades as layers of one texture, the
//! kept and moving far maps as layers of another, the kept levels as layers
//! of a third, the
//! lookup uniform and the samplers, with placeholders while shadows (or
//! levels) are off; and the near cascades' caches, layers of a third,
//! which their cascades are composed from.

use crate::cached::{Bound, Versioned};
use crate::gpu_data::ShadowUniform;
use crate::gpu_layouts::constants::{FAR_KEPT, FAR_MOVING};

use crate::pipelines::{sampler, view};

pub(crate) const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth16Unorm;

/// One resolution's maps.
pub(super) struct MapTextures {
    pub(super) resolution: u32,
    cascades: u32,
    levels: u32,
    // The near cascades as layers of one texture, the kept and moving far
    // maps as the two layers of another, and the kept levels as layers of a
    // third.
    textures: [wgpu::Texture; 2],
    level_texture: Option<wgpu::Texture>,
    // The near cascades as one array (lookups) and each layer (drawing).
    near_array: wgpu::TextureView,
    pub(super) near_layers: Vec<wgpu::TextureView>,
    // The far maps as one array (lookups) and each (drawing).
    far_array: wgpu::TextureView,
    pub(super) moving: wgpu::TextureView,
    pub(super) kept: wgpu::TextureView,
    // The kept levels as one array (lookups) and each layer (drawing).
    level_array: Option<wgpu::TextureView>,
    pub(super) level_layers: Vec<wgpu::TextureView>,
    // The near cascades' caches, while objects terrains place are
    // resident: one array (composing) and each layer (drawing).
    cache_texture: Option<wgpu::Texture>,
    cache_array: Option<wgpu::TextureView>,
    pub(super) cache_layers: Vec<wgpu::TextureView>,
}

impl Drop for MapTextures {
    fn drop(&mut self) {
        // Browsers otherwise keep the memory until garbage collection;
        // submitted work that still reads them completes first.
        for texture in self
            .textures
            .iter()
            .chain(&self.level_texture)
            .chain(&self.cache_texture)
        {
            texture.destroy();
        }
    }
}

/// Views of each layer of `texture`'s `layers`, for drawing.
fn layer_views(texture: &wgpu::Texture, layers: u32) -> Vec<wgpu::TextureView> {
    (0..layers)
        .map(|layer| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("engine/shadow-layer"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
        .collect()
}

/// A view of every layer of `texture`, for lookups.
fn array_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("engine/shadow-layers"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

impl MapTextures {
    /// Texels along a side of map `map` drawn every frame: a near
    /// cascade's layer, or the moving map.
    pub(super) fn drawn_texels(&self, map: usize) -> u32 {
        let texture = if map < super::MOVING_MAP {
            &self.textures[0]
        } else {
            &self.textures[1]
        };
        texture.width()
    }

    /// Maps of `resolution` texels (the near cascades, their caches and
    /// kept levels half as many) with `cascades` near layers, `levels` kept
    /// levels and, where `cached`, the near cascades' caches.
    fn new(
        device: &wgpu::Device,
        resolution: u32,
        (cascades, levels, cached): (u32, u32, bool),
    ) -> Self {
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
            texture("engine/shadow-far", resolution, 2),
        ];
        let level_texture =
            (levels > 0).then(|| texture("engine/shadow-levels", resolution / 2, levels));
        let cache_texture =
            cached.then(|| texture("engine/shadow-near-caches", resolution / 2, cascades));
        let far_layers = layer_views(&textures[1], 2);
        Self {
            resolution,
            cascades,
            levels,
            near_array: array_view(&textures[0]),
            near_layers: layer_views(&textures[0], cascades),
            far_array: array_view(&textures[1]),
            moving: far_layers[FAR_MOVING as usize].clone(),
            kept: far_layers[FAR_KEPT as usize].clone(),
            level_array: level_texture.as_ref().map(array_view),
            level_layers: level_texture
                .as_ref()
                .map_or_else(Vec::new, |texture| layer_views(texture, levels)),
            cache_array: cache_texture.as_ref().map(array_view),
            cache_layers: cache_texture
                .as_ref()
                .map_or_else(Vec::new, |texture| layer_views(texture, cascades)),
            textures,
            level_texture,
            cache_texture,
        }
    }

    /// Whether it holds the near cascades' caches.
    pub(super) fn caches(&self) -> bool {
        self.cache_array.is_some()
    }

    /// Texels along a side of the near cascades and their caches.
    pub(super) fn near_texels(&self) -> u32 {
        self.textures[0].width()
    }
}

pub(super) struct ShadowMaps {
    // The maps, replaced (or released) as a whole: bindings of the main
    // passes follow.
    textures: Versioned<Option<MapTextures>>,
    uniform: wgpu::Buffer,
    written: Option<ShadowUniform>,
    sampler: wgpu::Sampler,
    // Reads the four texels around a receiver-plane lookup.
    texel_sampler: wgpu::Sampler,
    _placeholder: wgpu::Texture,
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
        // Two layers, as the far maps' texture holds.
        let placeholder = crate::targets::texture_2d(
            device,
            "engine/shadow-placeholder",
            (1, 1),
            2,
            1,
            SHADOW_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let placeholder_array = placeholder.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Self {
            textures: Versioned::new(None),
            uniform,
            written: None,
            sampler,
            texel_sampler,
            _placeholder: placeholder,
            placeholder_array,
        }
    }

    /// The maps, once shadows ran.
    pub(super) fn textures(&self) -> Option<&MapTextures> {
        Option::as_ref(&self.textures)
    }

    /// Bindings of the main passes' first group (`bindings::CAMERA`): the
    /// lookup uniform, the near cascades, the far maps, the comparison
    /// sampler, the texel sampler of receiver planes and the kept levels.
    pub(super) fn entries(&self) -> [(&'static str, Bound<'_>); 6] {
        let (near, far, levels) = match &*self.textures {
            Some(maps) => (
                &maps.near_array,
                &maps.far_array,
                maps.level_array.as_ref().unwrap_or(&self.placeholder_array),
            ),
            None => (
                &self.placeholder_array,
                &self.placeholder_array,
                &self.placeholder_array,
            ),
        };
        let map = |texture| Bound::of(&self.textures, view(texture));
        [
            ("shadow", Bound::fixed(self.uniform.as_entire_binding())),
            ("near_shadows", map(near)),
            ("far_shadows", map(far)),
            ("shadow_sampler", Bound::fixed(sampler(&self.sampler))),
            ("shadow_texels", Bound::fixed(sampler(&self.texel_sampler))),
            ("kept_levels", map(levels)),
        ]
    }

    /// Bytes of the maps.
    pub(super) fn capacity_bytes(&self) -> u64 {
        self.textures().map_or(0, |maps| {
            let far = u64::from(maps.resolution) * u64::from(maps.resolution) * 2;
            let caches = if maps.caches() { maps.cascades } else { 0 };
            2 * far + u64::from(maps.cascades + maps.levels + caches) * far / 4
        })
    }

    /// Creates maps of `resolution` texels with `cascades` near layers,
    /// `levels` kept levels and, where `cached`, the near cascades' caches,
    /// unless they exist; returns whether it did (new maps hold nothing).
    pub(super) fn ensure(
        &mut self,
        device: &wgpu::Device,
        resolution: u32,
        (cascades, levels, cached): (u32, u32, bool),
    ) -> bool {
        if self.textures().is_some_and(|maps| {
            maps.resolution == resolution
                && maps.cascades == cascades
                && maps.levels == levels
                && maps.caches() == cached
        }) {
            return false;
        }
        self.textures.replace(Some(MapTextures::new(
            device,
            resolution,
            (cascades, levels, cached),
        )));
        true
    }

    /// The near cascades' caches as composing binds them, while the maps
    /// hold them.
    pub(super) fn caches_bound(&self) -> Option<Bound<'_>> {
        let caches = self.textures()?.cache_array.as_ref()?;
        Some(Bound::of(&self.textures, view(caches)))
    }

    /// What depth passes bind in place of the caches: they may draw into
    /// them.
    pub(super) fn caches_placeholder(&self) -> Bound<'_> {
        Bound::fixed(view(&self.placeholder_array))
    }

    /// Releases the maps while shadows are off.
    pub(super) fn release(&mut self) {
        if self.textures.is_some() {
            self.textures.replace(None);
        }
    }

    /// Writes the lookup uniform when it changed.
    pub(super) fn write_uniform(&mut self, queue: &wgpu::Queue, uniform: ShadowUniform) {
        if self.written != Some(uniform) {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
            self.written = Some(uniform);
        }
    }

    /// Sets the written uniform's count of terrain horizon records
    /// (`level_state.w`) where it changed.
    pub(super) fn write_horizon_records(&mut self, queue: &wgpu::Queue, count: f32) {
        if let Some(written) = &mut self.written
            && written.level_state[3] != count
        {
            written.level_state[3] = count;
            let offset = std::mem::offset_of!(ShadowUniform, level_state) + 12;
            queue.write_buffer(&self.uniform, offset as u64, bytemuck::bytes_of(&count));
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
