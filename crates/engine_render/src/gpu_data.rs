use engine_core::glam::Mat4;

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

pub(crate) fn mip_levels(width: u32, height: u32) -> u32 {
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
