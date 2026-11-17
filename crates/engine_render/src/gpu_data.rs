use engine_core::{Material, glam::Mat4};

pub use crate::CameraUniform;

impl CameraUniform {
    pub fn exact(view_projection: Mat4, position: glam::Vec3, seconds: f32) -> Self {
        Self {
            view_projection: view_projection.to_cols_array_2d(),
            position: position.extend(seconds).to_array(),
            mip_bias: 0.0,
            pad_0: 0,
            pad_1: 0,
            pad_2: 0,
        }
    }
}

pub fn pack_surface([first, second]: [f32; 2], emission: f32) -> u32 {
    let unit = |value: f32, max: f32| {
        let value = if value.is_nan() { 0.0 } else { value };
        (engine_core::math::clamp(value / max, 0.0, 1.0) * 255.0).round() as u32
    };
    unit(first, 1.0) | unit(second, 1.0) << 8 | unit(emission, Material::MAX_EMISSION) << 16
}

pub fn pack_color(color: [f32; 4]) -> [u32; 2] {
    let unit = |value: f32, fallback: f32| {
        let value = if value.is_nan() { fallback } else { value };
        (engine_core::math::clamp(value, 0.0, 1.0) * 65535.0).round() as u32
    };
    [
        unit(color[0], 0.0) | unit(color[1], 0.0) << 16,
        unit(color[2], 0.0) | unit(color[3], 1.0) << 16,
    ]
}

pub fn mip_levels(width: u32, height: u32) -> u32 {
    u32::BITS - width.max(height).max(1).leading_zeros()
}

pub(crate) fn level_copy(
    format: engine_core::TextureFormat,
    (width, height): (u32, u32),
    level: u32,
) -> (wgpu::TexelCopyBufferLayout, wgpu::Extent3d) {
    let (width, height) = ((width >> level).max(1), (height >> level).max(1));
    let (bytes_per_row, rows, extent) = if format.is_compressed() {
        let blocks = [width.div_ceil(4), height.div_ceil(4)];
        (blocks[0] * 16, blocks[1], (blocks[0] * 4, blocks[1] * 4))
    } else {
        (width * 4, height, (width, height))
    };
    (
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(rows),
        },
        wgpu::Extent3d {
            width: extent.0,
            height: extent.1,
            depth_or_array_layers: 1,
        },
    )
}

pub fn mip_chain_bytes(width: u32, height: u32) -> u64 {
    (1..mip_levels(width, height))
        .map(|level| u64::from((width >> level).max(1)) * u64::from((height >> level).max(1)) * 4)
        .sum()
}

pub fn clamped_extent(width: u32, height: u32, limit: u32) -> Option<(u32, u32)> {
    if width == 0 || height == 0 || limit == 0 {
        return None;
    }
    let largest = width.max(height);
    if largest <= limit {
        return Some((width, height));
    }
    let fit = |dimension: u32| {
        ((u64::from(dimension) * u64::from(limit)) / u64::from(largest)).max(1) as u32
    };
    Some((fit(width), fit(height)))
}
