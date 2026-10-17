//! Checked sizing for the reflection queues. Rejection precedes allocation.

#[derive(Clone, Copy, Debug)]
pub(crate) struct Level {
    pub offset: u32,
    pub length: u32,
    pub grid: [u32; 2],
}

pub(crate) struct Plan {
    pub size: [u32; 4],
    pub pixels: u32,
    pub row_log2: u32,
    pub levels: Vec<Level>,
    pub stride: u32,
    // Ranks, count/prefix levels, coordinates, raw f32, metadata, indirect, uniforms.
    pub bytes: [u64; 7],
}

/// Flattened groups use a power-of-two row; every padded lane fits u32 too.
pub(crate) fn grid(count: u32, maximum: u32) -> Option<[u32; 2]> {
    let row = 1u32.checked_shl(maximum.checked_ilog2()?)?;
    let groups = count.div_ceil(64);
    let width = groups.min(row);
    let height = groups.div_ceil(row).max(1);
    width.checked_mul(height)?.checked_mul(64)?;
    (height <= maximum).then_some([width, height])
}

fn supported(limits: &wgpu::Limits) -> bool {
    limits.max_bind_groups >= 2
        && limits.max_bindings_per_bind_group >= 15
        && limits.max_storage_buffers_per_shader_stage >= 6
        && limits.max_uniform_buffers_per_shader_stage >= 2
        && limits.max_sampled_textures_per_shader_stage >= 6
        && limits.max_samplers_per_shader_stage >= 1
        && limits.max_dynamic_uniform_buffers_per_pipeline_layout >= 1
        && limits.max_uniform_buffer_binding_size >= 48
        && limits.max_compute_invocations_per_workgroup >= 64
        && limits.max_compute_workgroup_size_x >= 64
        && limits.max_compute_workgroup_size_y >= 8
        && limits.max_compute_workgroup_size_z >= 1
        && limits.max_compute_workgroup_storage_size >= 272
        && limits.min_uniform_buffer_offset_alignment.is_power_of_two()
}

impl Plan {
    pub(crate) fn new(width: u32, height: u32, limits: &wgpu::Limits) -> Option<Self> {
        if !supported(limits)
            || width == 0
            || height == 0
            || width.max(height) > limits.max_texture_dimension_2d
        {
            return None;
        }
        let (width, height) = ((width / 2).max(1), (height / 2).max(1));
        let pixels = width.checked_mul(height)?;
        let (tiles_x, tiles_y) = (width.div_ceil(8), height.div_ceil(8));
        let maximum = limits.max_compute_workgroups_per_dimension;
        if tiles_x.max(tiles_y) > maximum {
            return None;
        }
        grid(pixels, maximum)?;
        let tiles = tiles_x.checked_mul(tiles_y)?;
        let mut length = tiles;
        let (mut records, mut levels) = (0u32, 0u32);
        loop {
            grid(length, maximum)?;
            records = records.checked_add(length)?;
            levels += 1;
            if length <= 64 {
                break;
            }
            length = length.div_ceil(64);
        }
        let stride = 48u32
            .div_ceil(limits.min_uniform_buffer_offset_alignment)
            .checked_mul(limits.min_uniform_buffer_offset_alignment)?;
        let arena = stride.checked_mul(levels.checked_add(1)?)?;
        let bytes = [
            u64::from(pixels) * 4,
            u64::from(records) * 4,
            u64::from(pixels) * 8,
            u64::from(pixels) * 16,
            16,
            12,
            u64::from(arena),
        ];
        for (index, &size) in bytes.iter().enumerate() {
            if size > limits.max_buffer_size
                || (index < 6 && size > limits.max_storage_buffer_binding_size)
                || usize::try_from(size).is_err()
            {
                return None;
            }
        }
        // All limits and arithmetic passed. Only now allocate the level plan.
        let mut scan = Vec::with_capacity(levels as usize);
        let (mut offset, mut length) = (0, tiles);
        loop {
            scan.push(Level {
                offset,
                length,
                grid: grid(length, maximum)?,
            });
            offset += length;
            if length <= 64 {
                break;
            }
            length = length.div_ceil(64);
        }
        Some(Self {
            size: [width, height, tiles_x, tiles_y],
            pixels,
            row_log2: maximum.checked_ilog2()?,
            levels: scan,
            stride,
            bytes,
        })
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.bytes.iter().sum()
    }
}
