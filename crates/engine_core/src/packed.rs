//! Four local matrices from component-major lanes, bit-identical to
//! [`crate::Transform::matrix`] for the same component values.
//!
//! Every lane performs exactly the scalar builder's sequence of IEEE additions
//! and multiplications (no fused operations or reordering), including the
//! signed zero `0 * scale` in each basis column's w lane. Transposes restore
//! ordinary matrices for parent-first composition. Callers whose poses are
//! already in lanes avoid assembling each column from scalars.

use glam::{Mat4, Vec4};

#[inline(always)]
pub(crate) fn local_matrices(
    translation: [Vec4; 3],
    rotation: [Vec4; 4],
    scale: [Vec4; 3],
) -> [Mat4; 4] {
    let [x, y, z, w] = rotation;
    let (x2, y2, z2) = (x + x, y + y, z + z);
    let (xx, xy, xz) = (x * x2, x * y2, x * z2);
    let (yy, yz, zz) = (y * y2, y * z2, z * z2);
    let (wx, wy, wz) = (w * x2, w * y2, w * z2);
    let one = Vec4::ONE;
    let axis = |a: Vec4, b: Vec4, c: Vec4, scale: Vec4| {
        Mat4::from_cols(a * scale, b * scale, c * scale, Vec4::ZERO * scale).transpose()
    };
    let x = axis(one - (yy + zz), xy + wz, xz - wy, scale[0]);
    let y = axis(xy - wz, one - (xx + zz), yz + wx, scale[1]);
    let z = axis(xz + wy, yz - wx, one - (xx + yy), scale[2]);
    let t = Mat4::from_cols(translation[0], translation[1], translation[2], one).transpose();
    [
        Mat4::from_cols(x.x_axis, y.x_axis, z.x_axis, t.x_axis),
        Mat4::from_cols(x.y_axis, y.y_axis, z.y_axis, t.y_axis),
        Mat4::from_cols(x.z_axis, y.z_axis, z.z_axis, t.z_axis),
        Mat4::from_cols(x.w_axis, y.w_axis, z.w_axis, t.w_axis),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Transform;
    use glam::{Quat, Vec3};

    fn bits(matrix: Mat4) -> [u32; 16] {
        matrix.to_cols_array().map(f32::to_bits)
    }

    #[test]
    fn packed_local_matrices_match_scalar_bits() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut random = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
        };
        let special = [0.0, -0.0, 1.0e-30, -1.0e30, f32::MIN_POSITIVE, 3.0e38];
        for case in 0..20_000 {
            let poses: [Transform; 4] = std::array::from_fn(|lane| {
                let mut scale = Vec3::new(random(), random(), random()) * 8.0;
                let mut translation = Vec3::new(random(), random(), random()) * 1.0e4;
                if (case + lane) % 7 == 0 {
                    scale.y = special[(case + lane) % special.len()];
                    translation.z = special[(case / 7 + lane) % special.len()];
                }
                let rotation = match (case + lane) % 5 {
                    // Admitted rotations need not be exactly unit length.
                    0 => Quat::from_xyzw(random(), random(), random(), random()),
                    1 => Quat::IDENTITY,
                    2 => -Quat::from_rotation_y(random() * 3.0),
                    _ => Quat::from_euler(
                        glam::EulerRot::XYZ,
                        random() * 6.0,
                        random() * 6.0,
                        random() * 6.0,
                    ),
                };
                Transform {
                    translation,
                    rotation,
                    scale,
                }
            });
            let lanes = |component: fn(&Transform) -> [f32; 4], axis: usize| {
                Vec4::from_array(poses.each_ref().map(|pose| component(pose)[axis]))
            };
            let packed = local_matrices(
                std::array::from_fn(|axis| lanes(|pose| pose.translation.extend(0.0).into(), axis)),
                std::array::from_fn(|axis| lanes(|pose| pose.rotation.to_array(), axis)),
                std::array::from_fn(|axis| lanes(|pose| pose.scale.extend(0.0).into(), axis)),
            );
            for lane in 0..4 {
                assert_eq!(
                    bits(packed[lane]),
                    bits(poses[lane].matrix()),
                    "{:?}",
                    poses[lane]
                );
            }
        }
    }
}
