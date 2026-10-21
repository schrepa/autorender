//! Falling water's map (`meshes/water.wgsl`): one tiling texture of the
//! grain waterfalls' sheets and white water carry, with its levels, as the
//! scene's static tables hold it (`engine_core::table_data::falls` makes
//! and describes it). Water reads it [`FALL_MAP_STRETCH`] times longer
//! along the water than across it.
//!
//! [`FALL_MAP_STRETCH`]: crate::gpu_layouts::constants::FALL_MAP_STRETCH

use engine_core::table_data::falls::{LEVELS, TEXELS};

/// The map, sampled with water's repeating sampler.
pub(crate) struct FallMap {
    view: wgpu::TextureView,
}

impl FallMap {
    /// The map `table` holds, uploaded with its levels.
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        table: &engine_core::table_data::falls::FallMap,
    ) -> Self {
        let side = TEXELS as u32;
        let texture = crate::targets::texture_2d(
            device,
            "engine/water-fall-map",
            (side, side),
            1,
            LEVELS,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        for (level, texels) in table.levels.iter().enumerate() {
            let side = side >> level;
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                texels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(side * 4),
                    rows_per_image: Some(side),
                },
                wgpu::Extent3d {
                    width: side,
                    height: side,
                    depth_or_array_layers: 1,
                },
            );
        }
        Self {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        }
    }

    pub(crate) fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Bytes of the map with its levels.
    pub(crate) fn capacity_bytes() -> u64 {
        (0..LEVELS)
            .map(|level| ((TEXELS >> level) * (TEXELS >> level) * 4) as u64)
            .sum()
    }
}
