pub use crate::gpu_layouts::scene::MAX_LEVELS;
use engine_core::{MeshVertex, glam::Mat4};

use crate::targets::FrameRegion;

pub use crate::gpu_layouts::constants::COMPOSED_BYTES;
pub use crate::gpu_layouts::constants::DRAW_ARGS_BYTES;
pub use crate::gpu_layouts::constants::FLAG_PHASE1;
pub use crate::gpu_layouts::constants::FLAG_PHASE2;
pub use crate::gpu_layouts::constants::FLAG_PROBE;
pub use crate::gpu_layouts::constants::FLAG_VISIBLE;
pub use crate::gpu_layouts::constants::LOD_LEVELS;
pub use crate::gpu_layouts::constants::LOD_STATE;
pub use crate::gpu_layouts::constants::MOVED_SLOT;
pub use crate::gpu_layouts::constants::STILL_FRAMES;
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

    pub fn probe(view_projection: Mat4, position: glam::Vec3, seconds: f32, size: u32) -> Self {
        Self {
            options: [1.0, pixel_angle(view_projection, size), 0.0, 0.0],
            region: [size as f32, size as f32, 1.0, 1.0],
            ..Self::exact(view_projection, position, seconds)
        }
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

pub use crate::gpu_layouts::scene::PoseData;

pub use crate::gpu_layouts::scene::GroundData;

pub fn ground_data(
    grounding: Option<engine_core::Grounding>,
    normal: [f32; 2],
    sink: f32,
    scale: f32,
) -> GroundData {
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

pub use crate::gpu_layouts::scene::PresentationFrame;

pub use crate::gpu_layouts::scene::CandidateData;

pub use crate::gpu_layouts::scene::BatchData;

pub const FAR_DEPTH: f32 = 0.0;

pub fn frustum_uniform(view_projection: glam::Mat4) -> ([[f32; 4]; 6], [[f32; 4]; 4]) {
    let rows = view_projection.transpose();
    let planes = [
        rows.w_axis + rows.x_axis,
        rows.w_axis - rows.x_axis,
        rows.w_axis + rows.y_axis,
        rows.w_axis - rows.y_axis,
        rows.z_axis,
        rows.w_axis - rows.z_axis,
    ]
    .map(|plane| plane.to_array());
    let absolute = glam::Mat4::from_cols(
        view_projection.x_axis.abs(),
        view_projection.y_axis.abs(),
        view_projection.z_axis.abs(),
        view_projection.w_axis.abs(),
    );
    (planes, absolute.to_cols_array_2d())
}

pub fn planes(view_projection: glam::Mat4) -> [[f32; 4]; 6] {
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

pub fn pixels_per_meter(culling: glam::Mat4, height: f32) -> f32 {
    culling.row(1).truncate().length() * height * 0.5
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
