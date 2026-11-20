use crate::CameraUniform;

impl CameraUniform {
    pub(crate) fn of(facts: &engine_core::ViewFacts, seconds: f32) -> Self {
        Self {
            view_projection: facts.view_projection.to_cols_array_2d(),
            position: facts.eye.extend(seconds).to_array(),
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
    let (side, bytes) = format.block();
    let columns = (width >> level).max(1).div_ceil(side);
    let rows = (height >> level).max(1).div_ceil(side);
    (
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(columns * bytes),
            rows_per_image: Some(rows),
        },
        wgpu::Extent3d {
            width: columns * side,
            height: rows * side,
            depth_or_array_layers: 1,
        },
    )
}

pub(crate) fn clamped_extent(width: u32, height: u32, limit: u32) -> Option<(u32, u32)> {
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
