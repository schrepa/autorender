//! Four independent, admitted local poses in component-major SIMD lanes.
//!
//! The caller owns validity, history selection, resets, and parent-first matrix
//! composition. These bounded stack arrays are not another retained pose store.

use glam::{Mat4, Vec4};
#[cfg(test)]
use glam::{Quat, Vec3};

use super::{LINEAR_ROTATION_DOT_THRESHOLD, blend_rotation};
use crate::{Transform, hierarchy::same_local_bits, packed};

// The scalar policy decides boundary cases itself. This small guard also avoids
// making its threshold decision depend on a backend's dot reduction order.
const PACKED_ROTATION_DOT_THRESHOLD: f32 = LINEAR_ROTATION_DOT_THRESHOLD + 8.0 * f32::EPSILON;

/// Blended components of four poses in component-major lanes.
struct Lanes {
    translation: [Vec4; 3],
    rotation: [Vec4; 4],
    scale: [Vec4; 3],
}

/// Local matrices of four blended poses, bit-identical to building each
/// [`blend_block`] result's matrix, plus whether each blended pose differs
/// bitwise from its current endpoint.
pub(super) struct BlendedMatrices {
    pub(super) matrices: [Mat4; 4],
    pub(super) changed: [bool; 4],
}

/// Blend and build local matrices without returning poses to ordinary
/// transforms in between. The caller owns eligibility, resets and parent-first
/// composition; `changed` is exact for finite current endpoints, which admitted
/// poses always are.
pub(super) fn blend_block_matrices(
    previous: &[Transform; 4],
    current: &[Transform; 4],
    alpha: f32,
) -> BlendedMatrices {
    if alpha == 0.0 || alpha == 1.0 {
        // Exact endpoints are copies; keep the scalar path's exact values.
        let poses = if alpha == 0.0 { previous } else { current };
        return BlendedMatrices {
            matrices: poses.map(|pose| pose.matrix()),
            changed: std::array::from_fn(|lane| !same_local_bits(&poses[lane], &current[lane])),
        };
    }
    // The current endpoint is gathered once for both the blend and the exact
    // unchanged test.
    let end = Endpoint::gather(current);
    let lanes = blend_lanes(previous, current, &end, alpha);
    // Bitwise equality of non-NaN lanes: equal values with equal sign bits
    // (which distinguishes the two zeros). Blends of finite poses that differ
    // from a finite endpoint compare unequal, including NaN results.
    let same = |left: Vec4, right: Vec4| {
        left.cmpeq(right).bitmask() & !(left.is_negative_bitmask() ^ right.is_negative_bitmask())
    };
    let mut unchanged = 0b1111;
    for axis in 0..3 {
        unchanged &= same(lanes.translation[axis], end.translation[axis]);
        unchanged &= same(lanes.scale[axis], end.scale[axis]);
    }
    for axis in 0..4 {
        unchanged &= same(lanes.rotation[axis], end.rotation[axis]);
    }
    BlendedMatrices {
        matrices: packed::local_matrices(lanes.translation, lanes.rotation, lanes.scale),
        changed: std::array::from_fn(|lane| unchanged & (1 << lane) == 0),
    }
}

/// Reference pose form of the packed blend; production composes matrices.
#[cfg(test)]
pub(super) fn blend_block(
    previous: &[Transform; 4],
    current: &[Transform; 4],
    alpha: f32,
) -> [Transform; 4] {
    if alpha == 0.0 {
        return *previous;
    }
    if alpha == 1.0 {
        return *current;
    }
    let lanes = blend_lanes(previous, current, &Endpoint::gather(current), alpha);
    std::array::from_fn(|lane| Transform {
        translation: Vec3::new(
            lanes.translation[0][lane],
            lanes.translation[1][lane],
            lanes.translation[2][lane],
        ),
        rotation: Quat::from_xyzw(
            lanes.rotation[0][lane],
            lanes.rotation[1][lane],
            lanes.rotation[2][lane],
            lanes.rotation[3][lane],
        ),
        scale: Vec3::new(
            lanes.scale[0][lane],
            lanes.scale[1][lane],
            lanes.scale[2][lane],
        ),
    })
}

/// Raw components of four poses in component-major lanes.
struct Endpoint {
    translation: [Vec4; 3],
    rotation: [Vec4; 4],
    scale: [Vec4; 3],
}

impl Endpoint {
    #[inline]
    fn gather(poses: &[Transform; 4]) -> Self {
        Self {
            translation: std::array::from_fn(|axis| {
                gather(poses.map(|pose| pose.translation[axis]))
            }),
            rotation: std::array::from_fn(|axis| {
                gather(poses.map(|pose| pose.rotation.to_array()[axis]))
            }),
            scale: std::array::from_fn(|axis| gather(poses.map(|pose| pose.scale[axis]))),
        }
    }
}

/// `end` holds the gathered components of `current`.
fn blend_lanes(
    previous: &[Transform; 4],
    current: &[Transform; 4],
    end: &Endpoint,
    alpha: f32,
) -> Lanes {
    let inverse = Vec4::splat(1.0 - alpha);
    let weight = Vec4::splat(alpha);
    let start = Endpoint::gather(previous);
    // Weighted endpoints deliberately avoid (current - previous), which can
    // overflow for otherwise admitted translations or scales.
    let translation: [Vec4; 3] = std::array::from_fn(|axis| {
        start.translation[axis] * inverse + end.translation[axis] * weight
    });
    let scale: [Vec4; 3] =
        std::array::from_fn(|axis| start.scale[axis] * inverse + end.scale[axis] * weight);
    let start = normalize(start.rotation);
    let end = normalize(end.rotation);
    let dot = dot(start, end);
    let end_weight = Vec4::select(dot.cmplt(Vec4::ZERO), -weight, weight);
    let mut rotation = normalize(std::array::from_fn(|axis| {
        start[axis] * inverse + end[axis] * end_weight
    }));
    let packed = dot
        .abs()
        .cmpge(Vec4::splat(PACKED_ROTATION_DOT_THRESHOLD))
        .bitmask();
    for lane in 0..4 {
        if packed & (1 << lane) == 0 {
            // Large arcs retain the existing SLERP implementation. Threshold
            // neighbors also retain the scalar NLERP/SLERP policy decision.
            let scalar =
                blend_rotation(previous[lane].rotation, current[lane].rotation, alpha).to_array();
            for (axis, value) in rotation.iter_mut().zip(scalar) {
                axis[lane] = value;
            }
        }
    }
    Lanes {
        translation,
        rotation,
        scale,
    }
}

#[inline]
fn gather(values: [f32; 4]) -> Vec4 {
    Vec4::from_array(values)
}

#[inline]
fn dot(left: [Vec4; 4], right: [Vec4; 4]) -> Vec4 {
    // Match glam 0.30.9's Wasm SIMD and SSE2 quaternion reduction order.
    (left[0] * right[0] + left[2] * right[2]) + (left[1] * right[1] + left[3] * right[3])
}

#[inline]
fn normalize(value: [Vec4; 4]) -> [Vec4; 4] {
    let length = sqrt(dot(value, value));
    // Division matches the existing quaternion normalization; an approximate
    // reciprocal square root would require a different numerical contract.
    value.map(|axis| axis / length)
}

#[inline]
fn sqrt(value: Vec4) -> Vec4 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        // This Wasm intrinsic is safe and computes one square root per pose.
        core::arch::wasm32::f32x4_sqrt(value.into()).into()
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        Vec4::from_array(value.to_array().map(f32::sqrt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(pose: Transform) -> [u32; 10] {
        [
            pose.translation.x,
            pose.translation.y,
            pose.translation.z,
            pose.rotation.x,
            pose.rotation.y,
            pose.rotation.z,
            pose.rotation.w,
            pose.scale.x,
            pose.scale.y,
            pose.scale.z,
        ]
        .map(f32::to_bits)
    }

    fn normalized(value: [f64; 4]) -> [f64; 4] {
        let length = value.iter().map(|x| x * x).sum::<f64>().sqrt();
        value.map(|x| x / length)
    }

    fn reference(previous: Quat, current: Quat, alpha: f32) -> [f64; 4] {
        let previous = normalized(previous.to_array().map(f64::from));
        let mut current = normalized(current.to_array().map(f64::from));
        let mut dot: f64 = previous.iter().zip(current).map(|(a, b)| a * b).sum();
        if dot < 0.0 {
            current = current.map(|x| -x);
            dot = -dot;
        }
        let theta = dot.min(1.0).acos();
        if theta < 1.0e-12 {
            return previous;
        }
        let alpha = f64::from(alpha);
        let left = ((1.0 - alpha) * theta).sin() / theta.sin();
        let right = (alpha * theta).sin() / theta.sin();
        normalized(std::array::from_fn(|i| {
            previous[i] * left + current[i] * right
        }))
    }

    fn angular_error(actual: Quat, expected: [f64; 4]) -> f64 {
        let mut actual = normalized(actual.to_array().map(f64::from));
        if actual.iter().zip(expected).map(|(a, b)| a * b).sum::<f64>() < 0.0 {
            actual = actual.map(|x| -x);
        }
        let difference = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let sum = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a + b).powi(2))
            .sum::<f64>()
            .sqrt();
        4.0 * difference.atan2(sum)
    }

    #[test]
    fn blended_matrices_match_reference_pose_matrices_and_change_flags() {
        let mut state = 0x6c07_8965_u32;
        let mut random = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0
        };
        let threshold_angle = 2.0 * LINEAR_ROTATION_DOT_THRESHOLD.acos();
        for case in 0..4_000 {
            let previous: [Transform; 4] = std::array::from_fn(|_| Transform {
                translation: Vec3::new(random(), random(), random()) * 100.0,
                rotation: Quat::from_euler(
                    glam::EulerRot::XYZ,
                    random() * 3.0,
                    random() * 3.0,
                    random() * 3.0,
                ),
                scale: Vec3::new(random(), random(), random()) * 3.0,
            });
            let current: [Transform; 4] = std::array::from_fn(|lane| {
                let angle = [0.0, 0.01, threshold_angle + random() * 1.0e-5, 2.5][lane];
                let mut pose = Transform {
                    translation: previous[lane].translation + Vec3::new(random(), 0.0, 0.0),
                    rotation: previous[lane].rotation
                        * Quat::from_axis_angle(Vec3::new(1.0, 2.0, -0.5).normalize(), angle),
                    scale: previous[lane].scale,
                };
                if (case + lane) % 5 == 0 {
                    // Stationary, sign-flipped and signed-zero lanes.
                    pose = previous[lane];
                    pose.rotation = -pose.rotation * [1.0, -1.0][case % 2];
                    pose.translation.y = [0.0, -0.0][case % 2];
                }
                pose
            });
            let alpha = [0.0, 0.25, 0.5, 0.999_999, random().abs()][case % 5];
            let reference = blend_block(&previous, &current, alpha);
            let packed = blend_block_matrices(&previous, &current, alpha);
            for lane in 0..4 {
                assert_eq!(
                    packed.matrices[lane].to_cols_array().map(f32::to_bits),
                    reference[lane].matrix().to_cols_array().map(f32::to_bits),
                    "{:?} -> {:?} at {alpha}",
                    previous[lane],
                    current[lane]
                );
                assert_eq!(
                    packed.changed[lane],
                    !same_local_bits(&reference[lane], &current[lane])
                );
            }
        }
    }

    #[test]
    fn exact_endpoints_preserve_admitted_nonunit_rotations_and_signed_zero() {
        let previous = std::array::from_fn(|lane| Transform {
            translation: Vec3::new(-0.0, lane as f32, 0.0),
            rotation: Quat::from_rotation_y(lane as f32) * 0.9996,
            scale: Vec3::new(-2.0, 1.0, 3.0),
        });
        let current = std::array::from_fn(|lane| Transform {
            translation: Vec3::new(0.0, -(lane as f32), -0.0),
            rotation: Quat::from_rotation_x(lane as f32) * -1.0004,
            scale: Vec3::new(-3.0, 2.0, 1.0),
        });
        for alpha in [0.0, -0.0, 1.0] {
            let expected = if alpha == 0.0 { previous } else { current };
            assert_eq!(
                blend_block(&previous, &current, alpha).map(bits),
                expected.map(bits)
            );
        }
    }

    #[test]
    fn mixed_lanes_preserve_scalar_arcs_and_the_rotation_error_budget() {
        let threshold_angle = 2.0 * LINEAR_ROTATION_DOT_THRESHOLD.acos();
        let bases = [
            Quat::IDENTITY,
            Quat::from_rotation_x(1.3),
            Quat::from_rotation_y(-2.4) * Quat::from_rotation_z(0.7),
            Quat::from_axis_angle(Vec3::new(1.0, -2.0, 3.0).normalize(), 3.0),
        ];
        let axis = Vec3::new(2.0, 3.0, -1.0).normalize();
        for neighbor in [-2.0e-5, -2.0e-7, 0.0, 2.0e-7, 2.0e-5] {
            let angles = [0.0, 0.02, threshold_angle + neighbor, 2.8];
            for signs in [[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, -1.0, 1.0]] {
                let previous = std::array::from_fn(|lane| Transform {
                    translation: Vec3::new(f32::MAX, lane as f32, -f32::MAX),
                    rotation: bases[lane] * 0.9996,
                    scale: Vec3::new(-f32::MAX, 1.0, 3.0),
                });
                let current = std::array::from_fn(|lane| Transform {
                    translation: Vec3::new(-f32::MAX, -(lane as f32), f32::MAX),
                    rotation: (bases[lane] * Quat::from_axis_angle(axis, angles[lane]))
                        * (1.0004 * signs[lane]),
                    scale: Vec3::new(-1.0, 2.0, f32::MAX),
                });
                for alpha in [0.001, 0.125, 0.211_334_5, 0.5, 0.788_665_5, 0.875, 0.999] {
                    let actual = blend_block(&previous, &current, alpha);
                    for lane in 0..4 {
                        let scalar = super::super::blend(&previous[lane], &current[lane], alpha);
                        assert_eq!(
                            actual[lane].translation.to_array().map(f32::to_bits),
                            scalar.translation.to_array().map(f32::to_bits)
                        );
                        assert_eq!(
                            actual[lane].scale.to_array().map(f32::to_bits),
                            scalar.scale.to_array().map(f32::to_bits)
                        );
                        assert!(actual[lane].translation.is_finite());
                        assert!(actual[lane].scale.is_finite());
                        assert!(
                            angular_error(
                                actual[lane].rotation,
                                reference(previous[lane].rotation, current[lane].rotation, alpha)
                            ) <= 1.0e-5
                        );
                        let scalar_dot = previous[lane]
                            .rotation
                            .normalize()
                            .dot(current[lane].rotation.normalize())
                            .abs();
                        if scalar_dot < LINEAR_ROTATION_DOT_THRESHOLD {
                            assert_eq!(
                                actual[lane].rotation.to_array().map(f32::to_bits),
                                scalar.rotation.to_array().map(f32::to_bits)
                            );
                        }
                    }
                }
            }
        }
    }
}
