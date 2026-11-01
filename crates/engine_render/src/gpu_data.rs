use engine_core::{MeshVertex, glam::Mat4};

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
