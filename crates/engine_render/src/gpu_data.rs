//! Explicit schemas for the adjacent shader, separate from gameplay components.

use bytemuck::{Pod, Zeroable};
use engine_core::{MeshVertex, RenderInstance, glam::Mat4};

/// WGSL uniform layout: one column-major mat4x4<f32>, alignment 16, size 64.
/// The GPU buffer starts at offset zero. Rust alignment need not match the GPU
/// allocation: only these 64 initialized bytes are copied into that buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CameraUniform {
    pub view_projection: [[f32; 4]; 4],
}

/// Vertex attributes, NOT a uniform/storage struct: columns at offsets
/// 0/16/32/48 and color at 64, with explicit 80-byte instance stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct InstanceData {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
}

impl From<&RenderInstance> for InstanceData {
    fn from(instance: &RenderInstance) -> Self {
        Self {
            model: instance.model.to_cols_array_2d(),
            color: instance.color,
        }
    }
}

/// Three aligned affine rows at storage offsets 0/16/32, with a 48-byte stride.
/// Only engine-owned frames establish the implicit final row [0, 0, 0, 1].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct PoseData {
    pub rows: [[f32; 4]; 3],
}

// Retained frames upload core's `PoseRows` bytes directly as these records.
const _: () = assert!(size_of::<PoseData>() == size_of::<engine_core::PoseRows>());

impl From<Mat4> for PoseData {
    fn from(m: Mat4) -> Self {
        Self {
            rows: [
                [m.x_axis.x, m.y_axis.x, m.z_axis.x, m.w_axis.x],
                [m.x_axis.y, m.y_axis.y, m.z_axis.y, m.w_axis.y],
                [m.x_axis.z, m.y_axis.z, m.z_axis.z, m.w_axis.z],
            ],
        }
    }
}

impl From<&RenderInstance> for PoseData {
    fn from(instance: &RenderInstance) -> Self {
        Self::from(instance.model)
    }
}

/// Separate per-instance color attribute; unchanged colors need no pose upload.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ColorData {
    pub color: [f32; 4],
}

/// Vertex attributes at offsets 0/12/24 with an explicit 32-byte vertex stride.
/// The CPU asset representation has no GPU byte-layout contract.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl From<&MeshVertex> for Vertex {
    fn from(vertex: &MeshVertex) -> Self {
        Self {
            position: vertex.position,
            normal: vertex.normal,
            uv: vertex.uv,
        }
    }
}

pub(crate) const SHADER: &str = include_str!("meshes.wgsl");
/// Compute passes of GPU presentation (decision 014).
pub(crate) const PRESENTATION_SHADER: &str = include_str!("presentation.wgsl");

/// `Frame` uniform of `presentation.wgsl`: 192 bytes, WGSL uniform layout.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct PresentationFrame {
    pub alpha: f32,
    pub candidates: u32,
    pub budget: u32,
    pub culling: u32,
    pub groups: u32,
    pub batches: u32,
    /// Node slots per half of the pose buffer; previous poses follow.
    pub node_capacity: u32,
    /// Zero while previous poses are inconsistent (no blending).
    pub blend: u32,
    pub planes: [[f32; 4]; 6],
    pub absolute_view_projection: [[f32; 4]; 4],
}

/// One candidate slot, its hierarchy node and its draw group (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CandidateData {
    pub slot: u32,
    pub node: u32,
    pub batch: u32,
    pub pad: u32,
}

/// One draw group: its candidate range (also its compacted visible-ID region),
/// mesh index count and local bounds (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct BatchData {
    pub start: u32,
    pub count: u32,
    pub index_count: u32,
    pub pad: u32,
    pub bounds_min: [f32; 4],
    pub bounds_max: [f32; 4],
}

/// Indexed indirect draw arguments written by the GPU (20 bytes).
pub(crate) const DRAW_ARGS_BYTES: u64 = 20;
/// Composed active-parent record of `presentation.wgsl` (112 bytes).
pub(crate) const COMPOSED_BYTES: u64 = 112;

/// Inward clip planes (WebGPU depth range) of a view-projection matrix and its
/// absolute value, as the CPU culling test derives them.
pub(crate) fn frustum_uniform(view_projection: glam::Mat4) -> ([[f32; 4]; 6], [[f32; 4]; 4]) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{AssetLibrary, Mesh, valid_global_matrix};
    use glam::{DMat3, Mat3, Mat4, Quat, Vec3};

    #[test]
    fn resize_suspends_zero_and_preserves_aspect_at_device_limit() {
        assert_eq!(clamped_extent(0, 1080, 8192), None);
        assert_eq!(clamped_extent(1920, 0, 8192), None);
        assert_eq!(clamped_extent(1920, 1080, 8192), Some((1920, 1080)));
        assert_eq!(clamped_extent(16_384, 8192, 8192), Some((8192, 4096)));
        assert_eq!(clamped_extent(u32::MAX, 1, 8192), Some((8192, 1)));
    }

    #[test]
    fn gpu_records_have_documented_offsets_and_column_order() {
        assert_eq!(size_of::<CameraUniform>(), 64);
        assert_eq!(size_of::<InstanceData>(), 80);
        assert_eq!(std::mem::offset_of!(InstanceData, color), 64);
        assert_eq!(size_of::<PoseData>(), 48);
        assert_eq!(size_of::<ColorData>(), 16);
        assert_eq!(size_of::<Vertex>(), 32);
        assert_eq!(std::mem::offset_of!(Vertex, normal), 12);
        assert_eq!(std::mem::offset_of!(Vertex, uv), 24);
        let mut assets = AssetLibrary::default();
        let instance = RenderInstance {
            model: Mat4::from_translation(Vec3::new(2.0, 3.0, 4.0)),
            color: [0.1, 0.2, 0.3, 1.0],
            mesh: assets.insert_mesh(Mesh::cube()).unwrap(),
            texture: None,
        };
        let upload = InstanceData::from(&instance);
        assert_eq!(upload.model[3], [2.0, 3.0, 4.0, 1.0]);
        assert_eq!(upload.color, instance.color);
        let affine = Mat4::from_scale_rotation_translation(
            Vec3::new(-2.0, 0.5, 3.0),
            Quat::from_rotation_y(0.7),
            Vec3::new(2.0, 3.0, 4.0),
        ) * Mat4::from_quat(Quat::from_rotation_z(0.4));
        let pose = PoseData::from(&RenderInstance {
            model: affine,
            ..instance
        });
        // Reconstruct exactly as vs_affine does. This retains shear/reflection.
        let reconstructed = Mat4::from_cols(
            pose.rows[0].into(),
            pose.rows[1].into(),
            pose.rows[2].into(),
            glam::Vec4::W,
        )
        .transpose();
        assert_eq!(reconstructed, affine);
        // Public snapshots retain all four rows even for a projective matrix.
        let projective = Mat4::perspective_rh(0.8, 1.5, 0.1, 100.0);
        let full = InstanceData::from(&RenderInstance {
            model: projective,
            ..instance
        });
        assert_eq!(full.model, projective.to_cols_array_2d());
        let vertex = Vertex::from(&MeshVertex {
            position: [1.0, 2.0, 3.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.25, 0.75],
        });
        assert_eq!(
            bytemuck::cast_slice::<_, f32>(&[vertex]),
            &[1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.25, 0.75]
        );
    }

    #[test]
    fn shader_cofactors_preserve_normals_under_nonuniform_positive_scale() {
        let model = Mat3::from_quat(Quat::from_rotation_y(0.7))
            * Mat3::from_diagonal(Vec3::new(0.25, 2.0, 5.0));
        let normal = Vec3::new(1.0, 1.0, 1.0).normalize();
        let shader_normal = shader_normal(model, normal);
        let expected = (model.inverse().transpose() * normal).normalize();
        assert!(shader_normal.abs_diff_eq(expected, 1e-6));
        assert!((model * normal).normalize().distance(expected) > 0.5);
    }

    fn shader_normal(model: Mat3, normal: Vec3) -> Vec3 {
        let scale = model
            .x_axis
            .abs()
            .max(model.y_axis.abs())
            .max(model.z_axis.abs())
            .max_element();
        let model = Mat3::from_cols(
            model.x_axis / scale,
            model.y_axis / scale,
            model.z_axis / scale,
        );
        let cofactors = Mat3::from_cols(
            model.y_axis.cross(model.z_axis),
            model.z_axis.cross(model.x_axis),
            model.x_axis.cross(model.y_axis),
        );
        let sign = model.x_axis.dot(cofactors.x_axis).signum();
        let scale = cofactors
            .x_axis
            .abs()
            .max(cofactors.y_axis.abs())
            .max(cofactors.z_axis.abs())
            .max_element();
        let cofactors = Mat3::from_cols(
            cofactors.x_axis / scale,
            cofactors.y_axis / scale,
            cofactors.z_axis / scale,
        );
        let transformed = cofactors * normal;
        (transformed / transformed.abs().max_element()).normalize() * sign
    }

    fn reference_normal(model: Mat3, normal: Vec3) -> Vec3 {
        let matrix = DMat3::from_cols_array(&model.to_cols_array().map(f64::from));
        (matrix.inverse().transpose() * normal.as_dvec3())
            .normalize()
            .as_vec3()
    }

    #[test]
    fn admitted_extreme_world_scales_keep_finite_unit_shader_normals() {
        for scale in [1.0e-12, 1.0e12] {
            let model = Mat3::from_quat(Quat::from_rotation_y(0.7))
                * Mat3::from_diagonal(Vec3::splat(scale));
            assert!(valid_global_matrix(Mat4::from_mat3(model)));
            for normal in [
                Vec3::X,
                Vec3::Y,
                Vec3::Z,
                Vec3::new(1.0, -1.0, 1.0).normalize(),
            ] {
                let result = shader_normal(model, normal);
                assert!(result.is_finite());
                assert!((result.length() - 1.0).abs() < 1.0e-6);
                assert!(result.abs_diff_eq(reference_normal(model, normal), 1.0e-6));
            }
            let unscaled = model.y_axis.cross(model.z_axis).normalize();
            assert!(!unscaled.is_finite() || unscaled.length() < 0.1);
        }
    }

    #[test]
    fn conditioned_shear_and_reflection_match_double_precision_inverse_transpose() {
        let model = Mat3::from_quat(Quat::from_rotation_y(0.7))
            * Mat3::from_diagonal(Vec3::new(-2.0, 0.003, 3.0))
            * Mat3::from_quat(Quat::from_rotation_z(0.4));
        assert!(valid_global_matrix(Mat4::from_mat3(model)));
        for normal in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(0.5, -1.0, 0.7).normalize(),
        ] {
            let result = shader_normal(model, normal);
            assert!(result.is_finite());
            assert!((result.length() - 1.0).abs() < 1.0e-6);
            assert!(result.abs_diff_eq(reference_normal(model, normal), 1.0e-4));
        }
    }

    #[test]
    fn unsafe_normal_conditioning_is_rejected_before_gpu_submission() {
        for model in [
            Mat3::from_diagonal(Vec3::new(0.0001, 10_000.0, 1.0)),
            Mat3::from_cols(Vec3::X, Vec3::new(1.0, 1.0e-8, 0.0), Vec3::Z),
        ] {
            assert!(model.inverse().is_finite());
            assert!(!valid_global_matrix(Mat4::from_mat3(model)));
        }
    }

    #[test]
    fn shader_is_valid_wgsl_with_no_optional_capabilities() {
        for source in [SHADER, PRESENTATION_SHADER] {
            let module = naga::front::wgsl::parse_str(source).expect("WGSL syntax");
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .expect("WGSL semantics");
        }
    }

    #[test]
    fn shader_float_literals_are_finite_f32_values() {
        // Browsers reject decimal literals beyond f32::MAX that naga rounds.
        for source in [SHADER, PRESENTATION_SHADER] {
            let bytes = source.as_bytes();
            let mut index = 0;
            while index < bytes.len() {
                let boundary = index == 0
                    || !(bytes[index - 1].is_ascii_alphanumeric() || bytes[index - 1] == b'_');
                if !(bytes[index].is_ascii_digit() && boundary) {
                    index += 1;
                    continue;
                }
                let start = index;
                while index < bytes.len()
                    && (bytes[index].is_ascii_alphanumeric()
                        || bytes[index] == b'.'
                        || (matches!(bytes[index], b'+' | b'-')
                            && matches!(bytes[index - 1], b'e' | b'E')))
                {
                    index += 1;
                }
                let token = &source[start..index];
                if token.starts_with("0x") || token.starts_with("0X") {
                    continue;
                }
                if let Ok(value) = token.trim_end_matches(['f', 'h', 'i', 'u']).parse::<f64>() {
                    assert!(value.abs() <= f64::from(f32::MAX), "{token}");
                }
            }
        }
    }

    #[test]
    fn presentation_bounds_match_cpu_culling() {
        let module = naga::front::wgsl::parse_str(PRESENTATION_SHADER).unwrap();
        let constant = |name: &str| {
            let (_, constant) = module
                .constants
                .iter()
                .find(|(_, constant)| constant.name.as_deref() == Some(name))
                .unwrap();
            match module.global_expressions[constant.init] {
                naga::Expression::Literal(naga::Literal::F32(value)) => value,
                ref other => panic!("{name} is {other:?}"),
            }
        };
        // engine_core::culling's support-test bounds, bit for bit.
        assert_eq!(constant("ROUNDING_MARGIN"), 32.0 * f32::EPSILON);
        assert_eq!(constant("MAX_COEFFICIENT"), 1.0e15);
        assert_eq!(constant("MAX_PRODUCT_BOUND"), f32::MAX / 8.0);
        assert_eq!(constant("F32_MAX"), f32::MAX);
        // hierarchy/validation.rs's common admission path. Its proof, and so
        // this GPU gate, requires a normal-conditioning limit of at least 6200
        // (checked at compile time below).
        assert_eq!(constant("MIN_LINEAR_SCALE"), 1.0 / 1024.0);
        assert_eq!(constant("COEFFICIENT_LIMIT"), 1_048_576.0);
        assert_eq!(constant("COMMON_DETERMINANT_MARGIN"), 1.0 / 512.0);
    }

    const _: () = assert!(engine_core::MAX_NORMAL_INVERSE_NORM >= 6200.0);

    #[test]
    fn presentation_layouts_match_wgsl() {
        assert_eq!(size_of::<PresentationFrame>(), 192);
        assert_eq!(std::mem::offset_of!(PresentationFrame, planes), 32);
        assert_eq!(
            std::mem::offset_of!(PresentationFrame, absolute_view_projection),
            128
        );
        assert_eq!(size_of::<CandidateData>(), 16);
        assert_eq!(size_of::<BatchData>(), 48);
        // Ten floats per pose, matching POSE_FLOATS in the shader.
        assert_eq!(size_of::<engine_core::NodePose>(), 40);
        assert_eq!(size_of::<engine_core::NodeInfo>(), 8);
        let module = naga::front::wgsl::parse_str(PRESENTATION_SHADER).unwrap();
        let size = |name: &str| {
            let (_, ty) = module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some(name))
                .unwrap();
            ty.inner.size(module.to_ctx())
        };
        assert_eq!(size("Frame"), 192);
        let (_, pose_floats) = module
            .constants
            .iter()
            .find(|(_, constant)| constant.name.as_deref() == Some("POSE_FLOATS"))
            .unwrap();
        assert!(matches!(
            module.global_expressions[pose_floats.init],
            naga::Expression::Literal(naga::Literal::U32(floats))
                if floats as usize * 4 == size_of::<engine_core::NodePose>()
        ));
        assert_eq!(size("Rows"), 48);
        assert_eq!(size("Composed") as u64, COMPOSED_BYTES);
        assert_eq!(size("Candidate"), 16);
        assert_eq!(size("Batch"), 48);
        assert_eq!(size("DrawArgs") as u64, DRAW_ARGS_BYTES);
    }

    #[test]
    fn frustum_uniform_matches_core_planes() {
        let camera = engine_core::Camera::default();
        let view_projection = camera.view_projection(1.5);
        let (planes, absolute) = frustum_uniform(view_projection);
        // A point in front of the camera is inside every plane.
        let inside = view_projection.inverse() * glam::Vec4::new(0.0, 0.0, 0.5, 1.0);
        let inside = inside / inside.w;
        for plane in planes {
            assert!(glam::Vec4::from(plane).dot(inside) > 0.0);
        }
        assert!(absolute.iter().flatten().all(|value| *value >= 0.0));
    }
}
