use glam::{Mat3, Mat4, Vec3A, Vec4};

use super::MAX_NORMAL_INVERSE_NORM;
use crate::presentation_bounds::{COEFFICIENT_LIMIT, COMMON_DETERMINANT_MARGIN, MIN_LINEAR_SCALE};
use crate::{Transform, math};

const DETERMINANT_MARGIN: f32 = 1.0 / 4096.0;
const PROVEN_NORMAL_INVERSE_NORM: f32 = 6200.0;

const MARGIN_UP: f32 = 1.0 + 1.0 / 4096.0;
const MARGIN_DOWN: f32 = 1.0 - 1.0 / 4096.0;
const STRETCH: f32 = 1.002_02;
const ROTATION_VOLUME: f32 = 0.9939;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Certificate {
    spread: f32,
    volume: f32,
    threshold: f32,
    reach: f32,
}

impl Certificate {
    pub(crate) const NONE: Self = Self {
        spread: f32::NAN,
        volume: f32::NAN,
        threshold: f32::NAN,
        reach: f32::NAN,
    };

    pub(crate) const ROOT: Self = Self {
        spread: STRETCH * MARGIN_UP,
        volume: ROTATION_VOLUME * (MARGIN_DOWN - 1.0 / 262_144.0) - 1.0 / 16_384.0,
        threshold: (STRETCH * MARGIN_UP)
            * (STRETCH * MARGIN_UP)
            * (STRETCH * MARGIN_UP)
            * (1.0 / 512.0 + 1.0 / 131_072.0)
            * MARGIN_UP,
        reach: (COEFFICIENT_LIMIT * MARGIN_DOWN / (STRETCH * MARGIN_UP))
            * (COEFFICIENT_LIMIT * MARGIN_DOWN / (STRETCH * MARGIN_UP))
            * MARGIN_DOWN,
    };

    #[cfg_attr(feature = "attribution", inline(never))]
    pub(crate) fn of(matrix: Mat4) -> Self {
        let (x, y, z) = (matrix.x_axis, matrix.y_axis, matrix.z_axis);
        let rows = x * x + y * y + z * z;
        let rho = Vec3A::from_vec4(rows).max_element().sqrt();
        let rho3 = (rho * rho) * rho;
        let spread = rho * (STRETCH * MARGIN_UP);
        let determinant = Vec3A::from_vec4(x)
            .dot(Vec3A::from_vec4(y).cross(Vec3A::from_vec4(z)))
            .abs();
        let volume =
            ROTATION_VOLUME * (determinant * MARGIN_DOWN - rho3 / 262_144.0) - rho3 / 16_384.0;
        let threshold = ((spread * spread) * spread) * (1.0 / 512.0 + 1.0 / 131_072.0) * MARGIN_UP;
        let translation = Vec3A::from_vec4(matrix.w_axis).abs().max_element() * MARGIN_UP;
        let room = (COEFFICIENT_LIMIT * MARGIN_DOWN - translation) / spread;
        let certificate = Self {
            spread,
            volume,
            threshold,
            reach: (room * room) * MARGIN_DOWN,
        };
        if volume > 0.0 && room > 0.0 && threshold.is_finite() && spread.is_finite() {
            certificate
        } else {
            Self::NONE
        }
    }

    #[inline(always)]
    pub(crate) fn admits(&self, local: &Transform) -> bool {
        let near_unit = (local.rotation.length_squared() - 1.0).abs() <= 1.0e-3;
        let scale = Vec3A::from(local.scale).abs();
        let largest = scale.max_element();
        let product = (scale.x * scale.y) * scale.z;
        let cube = (largest * largest) * largest;
        let lower = product * self.volume;
        near_unit
            && lower >= (cube * self.threshold) * MARGIN_UP
            && lower >= 1.0 / 67_108_864.0
            && largest * self.spread <= COEFFICIENT_LIMIT * MARGIN_DOWN
            && local.translation.length_squared() <= self.reach
    }
}

#[cfg_attr(not(feature = "attribution"), inline(always))]
#[cfg_attr(feature = "attribution", inline(never))]
pub(super) fn valid_global_matrix(matrix: Mat4) -> bool {
    common_gate(matrix) || admission(matrix)
}

#[inline(always)]
fn common_gate(matrix: Mat4) -> bool {
    let limit = Vec4::splat(COEFFICIENT_LIMIT);
    let basis = matrix
        .x_axis
        .abs()
        .max(matrix.y_axis.abs())
        .max(matrix.z_axis.abs());
    let bounded = basis.cmple(limit) & matrix.w_axis.abs().cmple(limit);
    let affine = matrix.x_axis.cmpeq(Vec4::ZERO)
        & matrix.y_axis.cmpeq(Vec4::ZERO)
        & matrix.z_axis.cmpeq(Vec4::ZERO)
        & matrix.w_axis.cmpeq(Vec4::ONE);
    if !(bounded.all() && affine.bitmask() & 0b1000 != 0) {
        return false;
    }
    let x = Vec3A::from_vec4(matrix.x_axis);
    let y = Vec3A::from_vec4(matrix.y_axis);
    let z = Vec3A::from_vec4(matrix.z_axis);
    let determinant = x.dot(y.cross(z));
    let scale = Vec3A::from_vec4(basis).max_element();
    MAX_NORMAL_INVERSE_NORM >= PROVEN_NORMAL_INVERSE_NORM
        && scale >= MIN_LINEAR_SCALE
        && determinant.abs() >= COMMON_DETERMINANT_MARGIN * ((scale * scale) * scale)
}

#[inline(never)]
fn admission(matrix: Mat4) -> bool {
    let infinity = Vec4::INFINITY;
    let finite = (matrix.x_axis.abs().cmplt(infinity)
        & matrix.y_axis.abs().cmplt(infinity)
        & matrix.z_axis.abs().cmplt(infinity)
        & matrix.w_axis.abs().cmplt(infinity))
    .all();
    if !finite
        || matrix.x_axis.w != 0.0
        || matrix.y_axis.w != 0.0
        || matrix.z_axis.w != 0.0
        || matrix.w_axis.w != 1.0
    {
        return false;
    }
    let x = Vec3A::from_vec4(matrix.x_axis);
    let y = Vec3A::from_vec4(matrix.y_axis);
    let z = Vec3A::from_vec4(matrix.z_axis);
    let determinant = x.dot(y.cross(z));
    let scale = x.abs().max(y.abs()).max(z.abs()).max_element();
    common_is_valid(matrix, determinant, scale) || conditioned_fallback(matrix, determinant, scale)
}

#[cold]
#[inline(never)]
fn conditioned_fallback(matrix: Mat4, determinant: f32, scale: f32) -> bool {
    let basis = Mat3::from_mat4(matrix);
    let cofactors = Mat3::from_cols(
        basis.y_axis.cross(basis.z_axis),
        basis.z_axis.cross(basis.x_axis),
        basis.x_axis.cross(basis.y_axis),
    );
    if !determinant.is_finite()
        || determinant == 0.0
        || !cofactors.is_finite()
        || !inverse_is_finite(matrix, determinant, scale)
    {
        return false;
    }
    let basis = Mat3::from_cols(
        basis.x_axis / scale,
        basis.y_axis / scale,
        basis.z_axis / scale,
    );
    let cofactors = Mat3::from_cols(
        basis.y_axis.cross(basis.z_axis),
        basis.z_axis.cross(basis.x_axis),
        basis.x_axis.cross(basis.y_axis),
    );
    let cofactor_scale = cofactors
        .x_axis
        .abs()
        .max(cofactors.y_axis.abs())
        .max(cofactors.z_axis.abs())
        .max_element();
    let determinant = basis.x_axis.dot(cofactors.x_axis).abs();
    let maximum_column_sum = math::max(
        math::max(
            basis.x_axis.abs().element_sum(),
            basis.y_axis.abs().element_sum(),
        ),
        basis.z_axis.abs().element_sum(),
    );
    let inverse_norm = maximum_column_sum * (cofactor_scale / determinant);
    inverse_norm.is_finite() && inverse_norm <= MAX_NORMAL_INVERSE_NORM
}

#[inline]
fn common_is_valid(matrix: Mat4, determinant: f32, linear_max: f32) -> bool {
    MAX_NORMAL_INVERSE_NORM >= PROVEN_NORMAL_INVERSE_NORM
        && bounded_determinant(matrix, determinant, linear_max, COMMON_DETERMINANT_MARGIN)
}

#[inline]
fn inverse_is_finite(matrix: Mat4, determinant: f32, linear_max: f32) -> bool {
    bounded_determinant(matrix, determinant, linear_max, DETERMINANT_MARGIN)
        || matrix.inverse().is_finite()
}

#[inline]
fn bounded_determinant(matrix: Mat4, determinant: f32, linear_max: f32, margin: f32) -> bool {
    if !(MIN_LINEAR_SCALE..=COEFFICIENT_LIMIT).contains(&linear_max)
        || Vec3A::from_vec4(matrix.w_axis).abs().max_element() > COEFFICIENT_LIMIT
    {
        return false;
    }

    determinant.abs() >= margin * ((linear_max * linear_max) * linear_max)
}
