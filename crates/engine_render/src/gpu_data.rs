use engine_core::{MeshVertex, Surface, glam::Mat4};

use crate::targets::FrameRegion;

pub use crate::CameraUniform;

impl CameraUniform {
    pub fn exact(view_projection: Mat4, position: glam::Vec3, seconds: f32) -> Self {
        let columns = view_projection.to_cols_array_2d();
        Self {
            view_projection: columns,
            previous_view_projection: columns,
            inverse_view_projection: view_projection.inverse().to_cols_array_2d(),
            position: position.extend(seconds).to_array(),
            jitter: [0.0; 4],
            options: [0.0; 4],
            wind: [0.0, 0.0, 0.0, seconds],
            region: [0.0; 4],
            detail: position.extend(0.0).to_array(),
            view: [0; 4],
        }
    }

    pub fn set_region(&mut self, view_projection: Mat4, region: FrameRegion) {
        self.options[1] = pixel_angle(view_projection, region.display.height);
        self.options[3] = region.mip_bias();
        self.region = region.uniform();
    }

    pub fn set_detail(&mut self, eye: glam::Vec3) {
        self.detail = eye.extend(0.0).to_array();
    }

    pub fn temporal(
        view_projection: Mat4,
        position: glam::Vec3,
        seconds: f32,
        previous: Mat4,
        jitter: [f32; 2],
        (width, height): (u32, u32),
        frame: u64,
    ) -> Self {
        let turn = (frame % 1024) as f32 * 0.618_034;
        Self {
            view_projection: jittered(view_projection, jitter).to_cols_array_2d(),
            previous_view_projection: previous.to_cols_array_2d(),
            jitter: [
                jitter[0],
                jitter[1],
                2.0 / width.max(1) as f32,
                2.0 / height.max(1) as f32,
            ],
            options: [0.0, 0.0, turn - turn.floor(), 0.0],
            ..Self::exact(view_projection, position, seconds)
        }
    }
}

pub fn pixel_angle(view_projection: Mat4, height: u32) -> f32 {
    let row = view_projection.row(1).truncate().length() * height.max(1) as f32;
    2.0 / engine_core::math::max(row, 1.0e-12)
}

pub fn pack_surface(surface: Surface) -> u32 {
    let unit = |value: f32, fallback: f32, max: f32| {
        let value = if value.is_nan() { fallback } else { value };
        (engine_core::math::clamp(value / max, 0.0, 1.0) * 255.0).round() as u32
    };
    let matte = Surface::MATTE;
    unit(surface.roughness, matte.roughness, 1.0)
        | unit(surface.metalness, matte.metalness, 1.0) << 8
        | unit(surface.emission, matte.emission, Surface::MAX_EMISSION) << 16
        | unit(surface.ripples, matte.ripples, 1.0) << 24
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

pub fn ground_data(
    grounding: Option<engine_core::Grounding>,
    normal: [f32; 2],
    sink: f32,
    scale: f32,
) -> [u32; 4] {
    let Some(grounding) = grounding else {
        return [0; 4];
    };
    let snorm = |value: f32| {
        u32::from((engine_core::math::clamp(value, -1.0, 1.0) * 32767.0).round() as i16 as u16)
    };
    [
        snorm(normal[0]) | snorm(normal[1]) << 16,
        sink.to_bits(),
        (grounding.height * scale).to_bits(),
        0,
    ]
}

pub fn pyramid_extents((width, height): (u32, u32), count: u32) -> Vec<[u32; 4]> {
    let mut source = (width.max(1), height.max(1));
    (0..count)
        .map(|_| {
            let written = (source.0.div_ceil(2), source.1.div_ceil(2));
            let extent = [written.0, written.1, source.0 - 1, source.1 - 1];
            source = written;
            extent
        })
        .collect()
}

pub fn jittered(view_projection: Mat4, jitter: [f32; 2]) -> Mat4 {
    Mat4::from_cols(
        glam::Vec4::X,
        glam::Vec4::Y,
        glam::Vec4::Z,
        glam::Vec4::new(jitter[0], jitter[1], 0.0, 1.0),
    ) * view_projection
}

pub use crate::gpu_layouts::scene::Vertex;

impl From<&MeshVertex> for Vertex {
    fn from(vertex: &MeshVertex) -> Self {
        Self {
            position: vertex.position,
            normal: octahedral(vertex.normal),
            uv: vertex.uv,
        }
    }
}

pub use crate::gpu_layouts::scene::ExtraVertex;

impl ExtraVertex {
    pub fn stream(mesh: &engine_core::Mesh) -> Option<Vec<Self>> {
        if mesh.colors().is_none() && mesh.uv1().is_none() {
            return None;
        }
        Some(
            mesh.vertices()
                .iter()
                .enumerate()
                .map(|(index, vertex)| Self {
                    color: mesh.colors().map_or([u16::MAX; 4], |colors| {
                        colors[index].map(|value| {
                            (engine_core::math::clamp(value, 0.0, 1.0) * 65535.0).round() as u16
                        })
                    }),
                    uv1: mesh.uv1().map_or(vertex.uv, |uv1| uv1[index]),
                })
                .collect(),
        )
    }
}

pub fn octahedral(normal: [f32; 3]) -> [i16; 2] {
    let [x, y, z] = normal;
    let length = x.abs() + y.abs() + z.abs();
    let (u, v) = (x / length, y / length);
    let sign = |value: f32| if value >= 0.0 { 1.0 } else { -1.0 };
    let (u, v) = if z < 0.0 {
        ((1.0 - v.abs()) * sign(u), (1.0 - u.abs()) * sign(v))
    } else {
        (u, v)
    };
    [u, v].map(|value| (engine_core::math::clamp(value, -1.0, 1.0) * 32767.0).round() as i16)
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

pub fn planes(view_projection: Mat4) -> [[f32; 4]; 6] {
    let rows = [
        view_projection.row(0),
        view_projection.row(1),
        view_projection.row(2),
        view_projection.row(3),
    ];
    [
        rows[3] + rows[0],
        rows[3] - rows[0],
        rows[3] + rows[1],
        rows[3] - rows[1],
        rows[2],
        rows[3] - rows[2],
    ]
    .map(|plane: glam::Vec4| {
        (plane / engine_core::math::max(plane.truncate().length(), 1.0e-12)).to_array()
    })
}

pub fn pixels_per_meter(culling: Mat4, height: f32) -> f32 {
    culling.row(1).truncate().length() * height * 0.5
}

pub fn mip_chain_bytes(width: u32, height: u32) -> u64 {
    (1..mip_levels(width, height))
        .map(|level| u64::from((width >> level).max(1)) * u64::from((height >> level).max(1)) * 4)
        .sum()
}

pub fn jitter_clip(offset: [f32; 2], width: u32, height: u32) -> [f32; 2] {
    [
        2.0 * offset[0] / width.max(1) as f32,
        -2.0 * offset[1] / height.max(1) as f32,
    ]
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
