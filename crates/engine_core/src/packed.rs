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
