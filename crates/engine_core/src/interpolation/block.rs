use glam::{Mat4, Vec4};

use super::{LINEAR_ROTATION_DOT_THRESHOLD, blend_rotation};
use crate::{Transform, hierarchy::same_local_bits, packed};

const PACKED_ROTATION_DOT_THRESHOLD: f32 = LINEAR_ROTATION_DOT_THRESHOLD + 8.0 * f32::EPSILON;

struct Lanes {
    translation: [Vec4; 3],
    rotation: [Vec4; 4],
    scale: [Vec4; 3],
}

pub(super) struct BlendedMatrices {
    pub(super) matrices: [Mat4; 4],
    pub(super) changed: [bool; 4],
}

pub(super) fn blend_block_matrices(
    previous: &[Transform; 4],
    current: &[Transform; 4],
    alpha: f32,
) -> BlendedMatrices {
    if alpha == 0.0 || alpha == 1.0 {
        let poses = if alpha == 0.0 { previous } else { current };
        return BlendedMatrices {
            matrices: poses.map(|pose| pose.matrix()),
            changed: std::array::from_fn(|lane| !same_local_bits(&poses[lane], &current[lane])),
        };
    }
    let end = Endpoint::gather(current);
    let lanes = blend_lanes(previous, current, &end, alpha);
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

fn blend_lanes(
    previous: &[Transform; 4],
    current: &[Transform; 4],
    end: &Endpoint,
    alpha: f32,
) -> Lanes {
    let inverse = Vec4::splat(1.0 - alpha);
    let weight = Vec4::splat(alpha);
    let start = Endpoint::gather(previous);
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
    (left[0] * right[0] + left[2] * right[2]) + (left[1] * right[1] + left[3] * right[3])
}

#[inline]
fn normalize(value: [Vec4; 4]) -> [Vec4; 4] {
    let length = sqrt(dot(value, value));
    value.map(|axis| axis / length)
}

#[inline]
fn sqrt(value: Vec4) -> Vec4 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        core::arch::wasm32::f32x4_sqrt(value.into()).into()
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        Vec4::from_array(value.to_array().map(f32::sqrt))
    }
}
