//! Sufficient bounds for the existing affine matrix admission predicate.
//!
//! This does not substitute an affine inverse or relax matrix admission.

use glam::{Mat3, Mat4, Vec3A, Vec4};

use super::MAX_NORMAL_INVERSE_NORM;
use crate::Transform;

const COEFFICIENT_LIMIT: f32 = 1_048_576.0; // 2^20
const MIN_LINEAR_SCALE: f32 = 1.0 / 1024.0; // 2^-10
const DETERMINANT_MARGIN: f32 = 1.0 / 4096.0; // 2^-12
const COMMON_DETERMINANT_MARGIN: f32 = 1.0 / 512.0; // 2^-9
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
/// every child accepted by [`Certificate::admits`] (decision 016).
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
        let scale = local.scale.abs();
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
    let maximum_column_sum = basis
        .x_axis
        .abs()
        .element_sum()
        .max(basis.y_axis.abs().element_sum())
        .max(basis.z_axis.abs().element_sum());
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

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat3, Vec3};

    fn raw_inputs(matrix: Mat4) -> Option<(f32, f32)> {
        let basis = Mat3::from_mat4(matrix);
        let cofactors = Mat3::from_cols(
            basis.y_axis.cross(basis.z_axis),
            basis.z_axis.cross(basis.x_axis),
            basis.x_axis.cross(basis.y_axis),
        );
        let determinant = basis.x_axis.dot(cofactors.x_axis);
        let linear_max = basis
            .x_axis
            .abs()
            .max(basis.y_axis.abs())
            .max(basis.z_axis.abs())
            .max_element();
        (matrix.is_finite()
            && cofactors.is_finite()
            && determinant.is_finite()
            && determinant != 0.0)
            .then_some((determinant, linear_max))
    }

    fn compare(matrix: Mat4) -> bool {
        let Some((determinant, linear_max)) = raw_inputs(matrix) else {
            // The caller rejects these before reaching the inverse predicate.
            return false;
        };
        let expected = matrix.inverse().is_finite();
        assert_eq!(
            inverse_is_finite(matrix, determinant, linear_max),
            expected,
            "{matrix:?}"
        );
        let fast = bounded_determinant(matrix, determinant, linear_max, DETERMINANT_MARGIN);
        assert!(!fast || expected, "shortcut must prove the existing check");
        fast
    }

    fn random(state: &mut u32) -> f32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*state >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0
    }

    // The complete original predicate, including the general inverse. Keep its
    // operation ordering independent of the sufficient bounds under test.
    fn scalar_reference(matrix: Mat4) -> bool {
        if !matrix.is_finite()
            || matrix.x_axis.w != 0.0
            || matrix.y_axis.w != 0.0
            || matrix.z_axis.w != 0.0
            || matrix.w_axis.w != 1.0
        {
            return false;
        }
        let basis = Mat3::from_mat4(matrix);
        let cofactors = Mat3::from_cols(
            basis.y_axis.cross(basis.z_axis),
            basis.z_axis.cross(basis.x_axis),
            basis.x_axis.cross(basis.y_axis),
        );
        let determinant = basis.x_axis.dot(cofactors.x_axis);
        let scale = basis
            .x_axis
            .abs()
            .max(basis.y_axis.abs())
            .max(basis.z_axis.abs())
            .max_element();
        if !determinant.is_finite()
            || determinant == 0.0
            || !cofactors.is_finite()
            || !matrix.inverse().is_finite()
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
        let maximum_column_sum = basis
            .x_axis
            .abs()
            .element_sum()
            .max(basis.y_axis.abs().element_sum())
            .max(basis.z_axis.abs().element_sum());
        let inverse_norm = maximum_column_sum * (cofactor_scale / determinant);
        inverse_norm.is_finite() && inverse_norm <= MAX_NORMAL_INVERSE_NORM
    }

    fn compare_admission(matrix: Mat4) -> bool {
        let expected = scalar_reference(matrix);
        assert_eq!(valid_global_matrix(matrix), expected, "{matrix:?}");
        expected
            && raw_inputs(matrix)
                .is_some_and(|(determinant, scale)| common_is_valid(matrix, determinant, scale))
    }

    #[test]
    fn gated_admission_matches_the_predicate() {
        let mut seed = 0x6a09_e667;
        let mut accepted = 0;
        for index in 0..60_000 {
            let scale = 2.0_f32.powi([-60, -12, -10, -9, -1, 0, 1, 10, 19, 20, 21, 40][index % 12]);
            let mut column =
                |scale| Vec3::new(random(&mut seed), random(&mut seed), random(&mut seed)) * scale;
            let mut matrix = Mat4::from_cols(
                column(scale).extend(0.0),
                column(scale).extend(0.0),
                column(scale).extend(0.0),
                column([1.0, 1.0e6, 1_048_576.0, 2.0e6][index % 4]).extend(1.0),
            );
            match index % 29 {
                0 => matrix.x_axis.w = -0.0,
                1 => matrix.y_axis.w = f32::MIN_POSITIVE,
                2 => matrix.w_axis.w = 1.0_f32.next_up(),
                3 => matrix.z_axis.y = f32::NAN,
                4 => matrix.w_axis.x = f32::INFINITY,
                5 => matrix.x_axis = matrix.y_axis,
                6 => matrix.z_axis *= 1.0e-4,
                _ => {}
            }
            let expected = admission(matrix);
            assert_eq!(valid_global_matrix(matrix), expected, "{matrix:?}");
            accepted += usize::from(expected);
        }
        for scale in [
            MIN_LINEAR_SCALE.next_down(),
            MIN_LINEAR_SCALE,
            COEFFICIENT_LIMIT,
            COEFFICIENT_LIMIT.next_up(),
        ] {
            let matrix = Mat4::from_scale(Vec3::splat(scale));
            assert_eq!(valid_global_matrix(matrix), admission(matrix));
        }
        assert!(accepted > 20_000, "{accepted}");
    }

    fn random_quat(state: &mut u32, stretch: f32) -> glam::Quat {
        let axis = Vec3::new(random(state), random(state), random(state)).normalize_or(Vec3::Y);
        let rotation = glam::Quat::from_axis_angle(axis, random(state) * 3.2);
        // Scale the quaternion to |q|² = stretch (near the 1e-3 tolerance).
        glam::Quat::from_vec4(glam::Vec4::from(rotation) * stretch.sqrt())
    }

    #[test]
    fn certified_children_pass_the_common_gate() {
        let mut seed = 0x2545_f491;
        let (mut certified, mut tried) = (0, 0);
        let mut tightest = f32::INFINITY;
        let stretches = [
            1.0,
            1.0 - 9.9e-4,
            1.0 + 9.9e-4,
            1.0 - 1.0e-3,
            1.0 + 1.0e-3,
            0.99,
            1.02,
        ];
        for index in 0..400_000 {
            // Parents: rotations, anisotropic and mirrored scales, shear from a
            // grandparent, translations up to the coefficient limit.
            let magnitude = 2.0_f32.powi([-9, -4, -1, 0, 0, 1, 3, 8, 14][index % 9]);
            let scale = |state: &mut u32| {
                let value = (0.05 + random(state).abs() * 2.0) * magnitude;
                if random(state) < -0.8 { -value } else { value }
            };
            let parent_scale = Vec3::new(scale(&mut seed), scale(&mut seed), scale(&mut seed));
            let parent = Mat4::from_scale_rotation_translation(
                Vec3::new(1.0, 1.0 + random(&mut seed).abs() * 3.0, 1.0),
                random_quat(&mut seed, 1.0),
                Vec3::ZERO,
            ) * Mat4::from_scale_rotation_translation(
                parent_scale,
                random_quat(&mut seed, 1.0),
                Vec3::new(random(&mut seed), random(&mut seed), random(&mut seed))
                    * [1.0, 1.0e3, 1.0e6, 1_048_000.0][index % 4],
            );
            if !valid_global_matrix(parent) {
                continue;
            }
            let certificate = Certificate::of(parent);
            // Children: tall and flat boxes near the determinant bound, tiny,
            // zero and huge scales, off-unit quaternions, far translations.
            let child_magnitude = 2.0_f32.powi([-12, -6, -2, 0, 0, 2, 6, 12][(index / 9) % 8]);
            let mut child_scale = Vec3::new(scale(&mut seed), scale(&mut seed), scale(&mut seed))
                * child_magnitude
                / magnitude;
            match index % 13 {
                0 => child_scale.y *= 40.0,
                1 => child_scale.x *= 1.0e-3,
                2 => child_scale.z = 0.0,
                3 => child_scale.x = f32::NAN,
                4 => child_scale.y = f32::INFINITY,
                _ => {}
            }
            let child = Transform {
                translation: Vec3::new(random(&mut seed), random(&mut seed), random(&mut seed))
                    * [0.0, 1.0, 1.0e3, 1.0e5, 1.0e6, 4.0e6][index % 6],
                rotation: random_quat(&mut seed, stretches[index % stretches.len()]),
                scale: child_scale,
            };
            tried += 1;
            if certificate.admits(&child) {
                certified += 1;
                let matrix = parent * child.matrix();
                assert!(common_gate(matrix), "{parent:?} {child:?} {matrix:?}");
                let x = Vec3A::from_vec4(matrix.x_axis);
                let y = Vec3A::from_vec4(matrix.y_axis);
                let z = Vec3A::from_vec4(matrix.z_axis);
                let largest = x.abs().max(y.abs()).max(z.abs()).max_element();
                let ratio = x.dot(y.cross(z)).abs() / (COMMON_DETERMINANT_MARGIN * largest.powi(3));
                tightest = tightest.min(ratio);
            }
            // The root certificate covers identity parents the same way.
            if Certificate::ROOT.admits(&child) {
                assert!(common_gate(child.matrix()), "{child:?}");
            }
        }
        assert!(
            tried > 100_000 && certified > tried / 10,
            "{certified}/{tried}"
        );
        // Certified cases reach within 10% of the gate's determinant bound.
        assert!(tightest < 1.1, "{tightest}");
    }

    #[test]
    fn certificate_threshold_sweep_passes_the_gate() {
        // Flat children at the determinant bound: scale (1, 1, k) has
        // |s_x s_y s_z| / s_max³ = k, swept across the certificate's own
        // threshold with the most stretched and shrunk quaternions allowed.
        let parents = [
            Mat4::IDENTITY,
            Mat4::from_scale(Vec3::splat(3.0)),
            Mat4::from_scale_rotation_translation(
                Vec3::new(1.2, 0.9, 0.8),
                glam::Quat::from_rotation_y(0.45),
                Vec3::new(10.0, -2.0, 7.0),
            ),
            Mat4::from_scale_rotation_translation(
                Vec3::new(-2.0, 0.5, 1.0),
                glam::Quat::from_euler(glam::EulerRot::XYZ, 0.3, -1.1, 2.0),
                Vec3::new(-1.0e5, 3.0e5, 2.0),
            ),
        ];
        let mut certified = 0;
        for parent in parents {
            assert!(valid_global_matrix(parent));
            let certificate = Certificate::of(parent);
            for stretch in [1.0 - 1.0e-3, 1.0 - 5.0e-4, 1.0, 1.0 + 5.0e-4, 1.0 + 1.0e-3] {
                for angle in [0.0, 0.3, std::f32::consts::FRAC_PI_4, 1.3, 3.0] {
                    for axis in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::ONE.normalize()] {
                        let rotation = glam::Quat::from_vec4(
                            glam::Vec4::from(glam::Quat::from_axis_angle(axis, angle))
                                * f32::sqrt(stretch),
                        );
                        for step in 0..400 {
                            let k = 1.9e-3 + step as f32 * 1.0e-6;
                            for scale in [Vec3::new(1.0, 1.0, k), Vec3::new(k, 1.0, 1.0)] {
                                let child = Transform {
                                    translation: Vec3::new(5.0, -3.0, 1.0),
                                    rotation,
                                    scale,
                                };
                                if certificate.admits(&child) {
                                    certified += 1;
                                    let matrix = parent * child.matrix();
                                    assert!(common_gate(matrix), "{parent:?} {child:?}");
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(certified > 1000, "{certified}");
    }

    #[test]
    fn root_certificate_is_the_identity_certificate() {
        let computed = Certificate::of(Mat4::IDENTITY);
        let root = Certificate::ROOT;
        assert!(root.spread >= computed.spread && root.threshold >= computed.threshold);
        assert!(root.volume <= computed.volume && root.reach <= computed.reach);
        assert!(root.volume > 0.99 && root.reach > 1.0e12);
    }

    #[test]
    fn vector_determinant_and_scale_match_scalar_bits() {
        let mut seed = 0x51f1_5eed;
        for index in 0..20_000 {
            let scale = 2.0_f32.powi([-60, -30, -10, -1, 0, 1, 10, 30][index % 8]);
            let mut column =
                |scale| Vec3::new(random(&mut seed), random(&mut seed), random(&mut seed)) * scale;
            let matrix = Mat4::from_cols(
                column(scale).extend(0.0),
                column(scale).extend(0.0),
                column(scale).extend(0.0),
                column(1.0e6).extend(1.0),
            );
            let basis = Mat3::from_mat4(matrix);
            let scalar_determinant = basis.x_axis.dot(basis.y_axis.cross(basis.z_axis));
            let scalar_scale = basis
                .x_axis
                .abs()
                .max(basis.y_axis.abs())
                .max(basis.z_axis.abs())
                .max_element();
            let x = Vec3A::from_vec4(matrix.x_axis);
            let y = Vec3A::from_vec4(matrix.y_axis);
            let z = Vec3A::from_vec4(matrix.z_axis);
            assert_eq!(
                x.dot(y.cross(z)).to_bits(),
                scalar_determinant.to_bits(),
                "{matrix:?}"
            );
            assert_eq!(
                x.abs().max(y.abs()).max(z.abs()).max_element().to_bits(),
                scalar_scale.to_bits()
            );
            assert_eq!(
                Vec3A::from_vec4(matrix.w_axis).abs().max_element(),
                matrix.w_axis.truncate().abs().max_element()
            );
        }
    }

    #[test]
    fn full_admission_matches_scalar_reference_across_seeded_scales_and_shear() {
        let mut seed = 0x156c_eab7;
        let mut fast_count = 0;
        for index in 0..4096 {
            let scale = 2.0_f32.powi([-50, -20, -4, 0, 4, 20, 40, 60][index % 8]);
            let mut column =
                |scale| Vec3::new(random(&mut seed), random(&mut seed), random(&mut seed)) * scale;
            let matrix = Mat4::from_cols(
                column(scale).extend(0.0),
                column(scale).extend(0.0),
                column(scale).extend(0.0),
                column(100_000.0).extend(1.0),
            );
            fast_count += usize::from(compare_admission(matrix));
        }
        assert!(fast_count > 1800, "ordinary matrices must use the bound");
        assert!(fast_count < 2500, "extremes must exercise the fallback");
    }

    #[test]
    fn admission_gate_and_conditioning_boundaries_match_scalar_reference() {
        for scale in [
            MIN_LINEAR_SCALE.next_down(),
            MIN_LINEAR_SCALE,
            MIN_LINEAR_SCALE.next_up(),
            0.01,
            COEFFICIENT_LIMIT.next_down(),
            COEFFICIENT_LIMIT,
            COEFFICIENT_LIMIT.next_up(),
        ] {
            assert_eq!(
                compare_admission(Mat4::from_scale(Vec3::splat(scale))),
                (MIN_LINEAR_SCALE..=COEFFICIENT_LIMIT).contains(&scale)
            );
        }
        for determinant in [
            COMMON_DETERMINANT_MARGIN.next_down(),
            COMMON_DETERMINANT_MARGIN,
            COMMON_DETERMINANT_MARGIN.next_up(),
        ] {
            assert_eq!(
                compare_admission(Mat4::from_scale(Vec3::new(1.0, 1.0, determinant))),
                determinant >= COMMON_DETERMINANT_MARGIN
            );
        }
        let boundary = MAX_NORMAL_INVERSE_NORM.recip();
        for thin_scale in [boundary.next_down(), boundary, boundary.next_up()] {
            for exponent in [-40, -20, 0, 20, 40] {
                for sign in [-1.0, 1.0] {
                    compare_admission(Mat4::from_scale(
                        Vec3::new(1.0, sign, thin_scale) * 2.0_f32.powi(exponent),
                    ));
                }
            }
        }
        for exponent in -24..=-8 {
            for sign in [-1.0, 1.0] {
                let delta = sign * 2.0_f32.powi(exponent);
                compare_admission(Mat4::from_cols(
                    Vec3::ONE.extend(0.0),
                    Vec3::new(1.0, 1.0 + delta, 1.0).extend(0.0),
                    Vec3::new(1.0, 1.0, 1.0 - delta).extend(0.0),
                    Vec3::new(17.0, -31.0, 47.0).extend(1.0),
                ));
            }
        }
    }

    #[test]
    fn admission_rejects_non_affine_nonfinite_and_extreme_inverse_overflow() {
        let rejected = |matrix| {
            assert!(!valid_global_matrix(matrix), "{matrix:?}");
            compare_admission(matrix);
        };
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut matrix = Mat4::IDENTITY;
            matrix.x_axis.x = invalid;
            rejected(matrix);
        }
        for column in 0..3 {
            let mut matrix = Mat4::IDENTITY;
            matrix.col_mut(column).w = f32::MIN_POSITIVE;
            rejected(matrix);
        }
        let mut matrix = Mat4::IDENTITY;
        matrix.w_axis.w = 1.0_f32.next_down();
        rejected(matrix);
        for (scale, translation) in [
            (0.0, 0.0),
            (0.5, f32::MAX),
            (1.0e10, 1.0e30),
            (f32::MIN_POSITIVE, 0.0),
            (f32::MAX, 0.0),
        ] {
            let matrix = Mat4::from_scale_rotation_translation(
                Vec3::splat(scale),
                glam::Quat::IDENTITY,
                Vec3::splat(translation),
            );
            rejected(matrix);
        }
    }

    #[test]
    fn seeded_affine_shear_and_reflection_match_the_general_inverse() {
        let mut seed = 0x724d_13a9;
        let mut fast_count = 0;
        for _ in 0..4096 {
            let mut column =
                |scale| Vec3::new(random(&mut seed), random(&mut seed), random(&mut seed)) * scale;
            let matrix = Mat4::from_cols(
                column(16.0).extend(0.0),
                column(16.0).extend(0.0),
                column(16.0).extend(0.0),
                column(100_000.0).extend(1.0),
            );
            fast_count += usize::from(compare(matrix));
        }
        assert!(
            fast_count > 4000,
            "ordinary cases must exercise the shortcut"
        );
    }

    #[test]
    fn coefficient_and_cancellation_boundaries_keep_the_original_result() {
        for coefficient in [
            COEFFICIENT_LIMIT.next_down(),
            COEFFICIENT_LIMIT,
            COEFFICIENT_LIMIT.next_up(),
        ] {
            assert_eq!(
                compare(Mat4::from_translation(Vec3::splat(coefficient))),
                coefficient <= COEFFICIENT_LIMIT
            );
            assert_eq!(
                compare(Mat4::from_scale(Vec3::splat(coefficient))),
                coefficient <= COEFFICIENT_LIMIT
            );
        }
        for determinant in [
            DETERMINANT_MARGIN.next_down(),
            DETERMINANT_MARGIN,
            DETERMINANT_MARGIN.next_up(),
        ] {
            assert_eq!(
                compare(Mat4::from_scale(Vec3::new(1.0, 1.0, determinant))),
                determinant >= DETERMINANT_MARGIN
            );
        }
        // Nearly dependent columns stress different scalar/SIMD sum orders.
        for exponent in -24..=-8 {
            for sign in [-1.0, 1.0] {
                let delta = sign * 2.0_f32.powi(exponent);
                compare(Mat4::from_cols(
                    Vec3::new(1.0, 1.0, 1.0).extend(0.0),
                    Vec3::new(1.0, 1.0 + delta, 1.0).extend(0.0),
                    Vec3::new(1.0, 1.0, 1.0 - delta).extend(0.0),
                    Vec3::new(17.0, -31.0, 47.0).extend(1.0),
                ));
            }
        }
    }

    #[test]
    fn extreme_scales_and_inverse_translation_keep_the_fallback() {
        for exponent in [-45, -40, -20, 0, 20, 40] {
            for sign in [-1.0, 1.0] {
                let scale = Vec3::splat(sign * 2.0_f32.powi(exponent));
                for translation in [Vec3::ZERO, Vec3::splat(1.0e30), Vec3::splat(f32::MAX)] {
                    compare(Mat4::from_scale_rotation_translation(
                        scale,
                        glam::Quat::IDENTITY,
                        translation,
                    ));
                }
            }
        }
        let translation_overflow = Mat4::from_scale_rotation_translation(
            Vec3::splat(0.5),
            glam::Quat::IDENTITY,
            Vec3::splat(f32::MAX),
        );
        assert!(!translation_overflow.inverse().is_finite());
        assert!(!compare(translation_overflow));

        // An affine-specific inverse could admit this, but glam's general
        // inverse overflows intermediate cofactors: preserve that rejection.
        let cofactor_overflow = Mat4::from_scale_rotation_translation(
            Vec3::splat(1.0e10),
            glam::Quat::IDENTITY,
            Vec3::splat(1.0e30),
        );
        assert!(
            glam::Affine3A::from_mat4(cofactor_overflow)
                .inverse()
                .is_finite()
        );
        assert!(!cofactor_overflow.inverse().is_finite());
        assert!(!compare(cofactor_overflow));
    }
}
