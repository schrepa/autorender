//! Sufficient bounds for the existing affine matrix admission predicate.
//!
//! This does not substitute an affine inverse or relax matrix admission.

use glam::{Mat3, Mat4, Vec3A, Vec4};

use super::MAX_NORMAL_INVERSE_NORM;
use crate::presentation_bounds::{COEFFICIENT_LIMIT, COMMON_DETERMINANT_MARGIN, MIN_LINEAR_SCALE};
use crate::{Transform, math};

const DETERMINANT_MARGIN: f32 = 1.0 / 4096.0; // 2^-12
const PROVEN_NORMAL_INVERSE_NORM: f32 = 6200.0;

const MARGIN_UP: f32 = 1.0 + 1.0 / 4096.0; // 1 + 2^-12
const MARGIN_DOWN: f32 = 1.0 - 1.0 / 4096.0; // 1 - 2^-12
// Largest singular value of the rotation matrix of a quaternion with
// |‖q‖² − 1| ≤ 1e-3, plus rounding of its entries, its scaling and the product
// with the parent (see `Certificate`).
const STRETCH: f32 = 1.002_02;
// Smallest |det| of that rotation matrix: (1 − 2.0006e-3)³ ≥ 0.99398.
const ROTATION_VOLUME: f32 = 0.9939;

/// Bounds of a stored, admitted parent global P that prove, without composing
/// it, that `P * child.matrix()` passes the common admission gate below for
/// every child accepted by [`Certificate::admits`].
///
/// With u = 2^-24, write the child's scale s, translation t and quaternion q
/// with n = ‖q‖² (|n − 1| ≤ 1.0003e-3 once the f32 check passed). glam's
/// rotation formula is exactly n·R + (1 − n)·I for a rotation R, so its
/// singular values lie in [1 − 2.0006e-3, 1 + 2.0006e-3] and its |det| is at
/// least 0.99398. Rounding the rotation entries (≤ 5.1u absolute), scaling
/// them (≤ u relative) and forming P's three-term row products (≤ 4u relative
/// to the absolute terms) keeps every composed linear coefficient within
/// ρ·|s_j|·(1.00201 + 17u) ≤ ρ·s_max·STRETCH, ρ being P's largest row 2-norm,
/// and every translation within (ρ‖t‖ + τ)(1 + 4u), τ = P's largest
/// translation magnitude. The same entry errors move the composed determinant
/// by less than 461u·ρ³·|s_x s_y s_z| from det(P)·det(rotation)·s_x s_y s_z,
/// and the gate's own determinant evaluation errs by less than 64u·S³. Every
/// bound below is widened by at least 2^-12 relative, far beyond the rounding
/// of these f32 evaluations. Rows of P's linear part and its determinant are
/// computed here once per stored parent; children pay a few scalar operations.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Certificate {
    // Upper bound on ρ·STRETCH: composed coefficients stay below spread·s_max.
    spread: f32,
    // Lower bound on |det| of the composed linear part per unit |s_x s_y s_z|.
    volume: f32,
    // Upper bound on the gate's 2^-9·S³ threshold plus its determinant
    // evaluation error, per unit s_max³.
    threshold: f32,
    // Largest squared child translation norm that keeps composed translations
    // within the coefficient limit.
    reach: f32,
}

impl Certificate {
    /// No proof: every comparison with NaN fails.
    pub(crate) const NONE: Self = Self {
        spread: f32::NAN,
        volume: f32::NAN,
        threshold: f32::NAN,
        reach: f32::NAN,
    };

    /// The certificate of `Mat4::IDENTITY` (roots): ρ = 1, |det| = 1, τ = 0,
    /// with the same widened bounds as `of` computes.
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

    /// Bounds of an admitted (finite, exactly affine) parent global.
    #[cfg_attr(feature = "attribution", inline(never))]
    pub(crate) fn of(matrix: Mat4) -> Self {
        let (x, y, z) = (matrix.x_axis, matrix.y_axis, matrix.z_axis);
        // Lanes 0..3 hold the squared 2-norms of the linear part's rows.
        let rows = x * x + y * y + z * z;
        let rho = Vec3A::from_vec4(rows).max_element().sqrt();
        let rho3 = (rho * rho) * rho;
        let spread = rho * (STRETCH * MARGIN_UP);
        // As the gate evaluates it: error below 64u·ρ³ = 2^-18·ρ³.
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
        // NaN or non-positive bounds prove nothing.
        if volume > 0.0 && room > 0.0 && threshold.is_finite() && spread.is_finite() {
            certificate
        } else {
            Self::NONE
        }
    }

    /// Whether the child `local` composes with this certificate's parent into
    /// a global the gate admits. Includes `local_matrix`'s rotation check, so
    /// `true` means eager evaluation would publish a valid global.
    #[inline(always)]
    pub(crate) fn admits(&self, local: &Transform) -> bool {
        let near_unit = (local.rotation.length_squared() - 1.0).abs() <= 1.0e-3;
        // SIMD lanes: `Vec3`'s scalar `max_element` compiles to float selects,
        // which browsers turn into branches on every leaf's scale.
        let scale = Vec3A::from(local.scale).abs();
        let largest = scale.max_element();
        let product = (scale.x * scale.y) * scale.z;
        let cube = (largest * largest) * largest;
        let lower = product * self.volume;
        // Written so that NaN or infinite inputs fail.
        near_unit
            && lower >= (cube * self.threshold) * MARGIN_UP
            && lower >= 1.0 / 67_108_864.0 // 2^-26: keeps S above 2^-10
            && largest * self.spread <= COEFFICIENT_LIMIT * MARGIN_DOWN
            && local.translation.length_squared() <= self.reach
    }
}

/// A sufficient vector gate for the common path, then the unchanged
/// predicate. Every coefficient within 2^20 (NaN fails) is finite; the
/// affine-row compares, raw determinant, scale and margin are the common
/// path's own, so a pass is exactly a common-path acceptance. Vector compares
/// replace per-lane extraction, which is costly in browser Wasm.
#[cfg_attr(not(feature = "attribution"), inline(always))]
#[cfg_attr(feature = "attribution", inline(never))]
pub(super) fn valid_global_matrix(matrix: Mat4) -> bool {
    common_gate(matrix) || admission(matrix)
}

/// The vector gate alone: `true` is exactly a common-path acceptance.
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

/// The original predicate, out of line behind the gate: its common path,
/// then everything outside the proven bound. Decisions are unchanged.
#[inline(never)]
fn admission(matrix: Mat4) -> bool {
    // One combined mask equals Mat4::is_finite: every lane of every column.
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
    // Pinned glam's Vec3A cross product and three-term dot product evaluate
    // the scalar Vec3 expressions in the same order ((x*x' + y*y') + z*z'),
    // on both the Wasm SIMD and SSE2 backends, without extracting a Mat3.
    // Absolute maxima of finite values are exact. These inputs are therefore
    // bit-identical to the original scalar determinant and scale.
    let x = Vec3A::from_vec4(matrix.x_axis);
    let y = Vec3A::from_vec4(matrix.y_axis);
    let z = Vec3A::from_vec4(matrix.z_axis);
    let determinant = x.dot(y.cross(z));
    let scale = x.abs().max(y.abs()).max(z.abs()).max_element();
    common_is_valid(matrix, determinant, scale) || conditioned_fallback(matrix, determinant, scale)
}

/// Preserve the original scalar operations outside the sufficient bound,
/// including their overflow, cancellation and conditioning-boundary behavior.
/// `matrix` is finite and exactly affine; `determinant` and `scale` are the
/// bit-identical raw determinant and maximum absolute basis coefficient.
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
    // The finite, exactly affine input has basis coefficients bounded by S.
    // Let epsilon=f32::EPSILON. The raw determinant error is less
    // than 32*epsilon*S^3 (the inverse proof below), and the rounded S^3 differs
    // from its exact value by less than 2*epsilon relative error. Our gate gives
    //   |exact determinant| / S^3
    //     >= 2^-9*(1-2*epsilon) - 32*epsilon > 0.001949.
    //
    // In the original normalized pass, each rounded basis entry remains <=1
    // in magnitude, with absolute division error <=epsilon. Its determinant's
    // six degree-three terms change by <19*epsilon from those input errors;
    // cross/dot evaluation adds <32*epsilon. A loose combined 128*epsilon
    // allowance therefore leaves the computed |determinant| >2^-10.
    // Each normalized cofactor is <=2, and each absolute column sum is <=3.
    // The original norm expression is consequently <=6144 in exact arithmetic
    // and <6200 with rounded division/multiplication. Keeping the constant
    // comparison here also disables this shortcut if the public bound tightens.
    //
    // The coefficient bound makes every raw cofactor finite, and the stronger
    // determinant margin implies the existing general-inverse finiteness proof.
    // No normalized matrix or remaining cofactors are needed on this path.
    MAX_NORMAL_INVERSE_NORM >= PROVEN_NORMAL_INVERSE_NORM
        && bounded_determinant(matrix, determinant, linear_max, COMMON_DETERMINANT_MARGIN)
}

/// The caller has checked finite, exactly affine input and finite nonzero raw
/// determinant/cofactors. `linear_max` is the maximum absolute basis coefficient;
/// `determinant` is x.dot(y.cross(z)), as computed by valid_global_matrix.
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

    // Audited against pinned glam 0.30.9 Mat4::inverse in these source files:
    // https://github.com/bitshifter/glam-rs/blob/0.30.9/src/f32/scalar/mat4.rs
    // https://github.com/bitshifter/glam-rs/blob/0.30.9/src/f32/wasm32/mat4.rs
    // The sse2, neon and coresimd siblings use the same cofactor construction.
    // Revisit this proof when that implementation changes.
    //
    // For exactly affine input, its determinant and our raw determinant each
    // evaluate three 2-product differences, three products and two sums. With
    // |basis coefficient| <= S and epsilon = f32::EPSILON, each evaluation's
    // absolute error is < 32*epsilon*S^3. Their difference is therefore less
    // than 2^-17*S^3. S >= 2^-10 gives epsilon*S^3 >= 2^-53, which also absorbs
    // subnormal absolute rounding errors. Translation does not enter the
    // nonzero determinant terms because the bottom row is exactly [0,0,0,1].
    //
    // The weaker 2^-12*S^3 gate leaves ample room for that difference and for
    // the two rounded multiplies computing S^3: glam's determinant has magnitude
    // > 2^-14*S^3 >= 2^-44. Its reciprocal is consequently finite and < 2^45.
    // All matrix coefficients, including translation and the affine 0/1 row,
    // are bounded by 2^20. The inverse's 2x2 minors are < 2^42 and its 3x3
    // cofactors are < 2^64, including rounding. Thus even its final products
    // are < 2^109, far below f32 overflow. No intermediate can create infinity
    // or NaN. Underflow to zero remains finite, just as in the original check.
    //
    // This only proves that the old inverse check returns true. Anything not
    // proved takes the original path, preserving its extreme-range behavior.
    determinant.abs() >= margin * ((linear_max * linear_max) * linear_max)
}
