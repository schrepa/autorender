//! Skinned pedestrians (decision 036): a low-poly figure of boxes bound to
//! eleven joints, outfits in vertex colors, and walk and idle clips
//! (decision 035) that turn the joints. Built in meters, facing +X, feet on
//! the origin; the city scales each person's root.

use std::f32::consts::TAU;

use engine::prelude::*;

use super::builder::{Builder, Map};

/// Outfits the city draws people in.
pub(crate) const VARIANTS: u8 = 8;
/// Joints of every person, parents before children.
pub(crate) const JOINTS: usize = 11;

const HIPS: u16 = 0;
const SPINE: u16 = 1;
const HEAD: u16 = 2;
const THIGH_L: u16 = 3;
const SHIN_L: u16 = 4;
const THIGH_R: u16 = 5;
const SHIN_R: u16 = 6;
const ARM_L: u16 = 7;
const FOREARM_L: u16 = 8;
const ARM_R: u16 = 9;
const FOREARM_R: u16 = 10;

/// Each joint's parent and rest translation from it (the root for `None`).
pub(crate) const SKELETON: [(Option<usize>, [f32; 3]); JOINTS] = [
    (None, [0.0, 0.95, 0.0]),
    (Some(0), [0.0, 0.08, 0.0]),
    (Some(1), [0.0, 0.52, 0.0]),
    (Some(0), [0.0, -0.02, 0.1]),
    (Some(3), [0.0, -0.45, 0.0]),
    (Some(0), [0.0, -0.02, -0.1]),
    (Some(5), [0.0, -0.45, 0.0]),
    (Some(1), [0.0, 0.46, 0.23]),
    (Some(7), [0.0, -0.3, 0.0]),
    (Some(1), [0.0, 0.46, -0.23]),
    (Some(9), [0.0, -0.3, 0.0]),
];

/// Joint positions at rest, in the person's space.
fn rest_positions() -> [Vec3; JOINTS] {
    let mut positions = [Vec3::ZERO; JOINTS];
    for (index, (parent, offset)) in SKELETON.iter().enumerate() {
        positions[index] =
            parent.map_or(Vec3::ZERO, |parent| positions[parent]) + Vec3::from(*offset);
    }
    positions
}

/// Inverse bind matrices: from the person's space into each joint's.
pub(crate) fn inverse_binds() -> Vec<Mat4> {
    rest_positions()
        .iter()
        .map(|position| Mat4::from_translation(-*position))
        .collect()
}

/// Skin, hair, top, bottom and shoe colors of `variant`.
fn outfit(variant: u8) -> [[f32; 3]; 5] {
    let skins = [
        [0.95, 0.76, 0.62],
        [0.78, 0.55, 0.4],
        [0.55, 0.36, 0.24],
        [0.36, 0.23, 0.15],
    ];
    let hair = [
        [0.08, 0.06, 0.05],
        [0.35, 0.2, 0.1],
        [0.8, 0.65, 0.35],
        [0.5, 0.5, 0.5],
    ];
    let tops = [
        [0.85, 0.2, 0.25],
        [0.2, 0.45, 0.85],
        [0.95, 0.85, 0.3],
        [0.2, 0.7, 0.45],
        [0.95, 0.95, 0.95],
        [0.15, 0.15, 0.18],
        [0.65, 0.35, 0.8],
        [1.0, 0.55, 0.2],
    ];
    let bottoms = [
        [0.15, 0.2, 0.35],
        [0.25, 0.25, 0.28],
        [0.55, 0.45, 0.3],
        [0.1, 0.1, 0.12],
    ];
    let index = variant as usize;
    [
        skins[index % skins.len()],
        hair[(index / 2) % hair.len()],
        tops[index % tops.len()],
        bottoms[(index * 3) % bottoms.len()],
        [0.08, 0.08, 0.09],
    ]
}

/// The skinned mesh of outfit `variant`, in bind space.
pub(crate) fn mesh(variant: u8) -> Result<Mesh, String> {
    let [skin, hair, top, bottom, shoes] = outfit(variant);
    let mut b = Builder::default();
    let map = Map::Rect(0.0, 0.0, 1.0, 1.0);
    let v = Vec3::new;
    let part = |b: &mut Builder, joint: u16, color: [f32; 3], min: Vec3, max: Vec3| {
        b.joint(joint);
        b.paint(color);
        b.block(min, max, map);
    };
    // Legs and feet (the figure faces +X; left is +Z).
    for (side, thigh, shin) in [(0.1, THIGH_L, SHIN_L), (-0.1, THIGH_R, SHIN_R)] {
        part(
            &mut b,
            thigh,
            bottom,
            v(-0.08, 0.5, side - 0.07),
            v(0.08, 0.95, side + 0.07),
        );
        part(
            &mut b,
            shin,
            bottom,
            v(-0.07, 0.08, side - 0.06),
            v(0.07, 0.52, side + 0.06),
        );
        part(
            &mut b,
            shin,
            shoes,
            v(-0.08, 0.0, side - 0.06),
            v(0.16, 0.09, side + 0.06),
        );
    }
    // Hips, torso, neck, head and hair.
    part(
        &mut b,
        HIPS,
        bottom,
        v(-0.1, 0.86, -0.18),
        v(0.1, 1.04, 0.18),
    );
    part(&mut b, SPINE, top, v(-0.11, 1.02, -0.2), v(0.11, 1.5, 0.2));
    part(
        &mut b,
        HEAD,
        skin,
        v(-0.05, 1.48, -0.05),
        v(0.05, 1.56, 0.05),
    );
    part(
        &mut b,
        HEAD,
        skin,
        v(-0.1, 1.55, -0.09),
        v(0.11, 1.78, 0.09),
    );
    part(&mut b, HEAD, hair, v(-0.12, 1.72, -0.1), v(0.08, 1.82, 0.1));
    // Arms and hands.
    for (side, arm, forearm) in [(0.26, ARM_L, FOREARM_L), (-0.26, ARM_R, FOREARM_R)] {
        part(
            &mut b,
            arm,
            top,
            v(-0.05, 1.14, side - 0.05),
            v(0.05, 1.5, side + 0.05),
        );
        part(
            &mut b,
            forearm,
            skin,
            v(-0.04, 0.88, side - 0.04),
            v(0.04, 1.17, side + 0.04),
        );
        part(
            &mut b,
            forearm,
            skin,
            v(-0.045, 0.8, side - 0.035),
            v(0.045, 0.89, side + 0.035),
        );
    }
    b.finish(false).map(|(mesh, _)| mesh)
}

/// A looping clip of `duration` seconds keyed at `keys` + 1 evenly spaced
/// times, each joint turning about Z by `angle(joint, phase)` (phase in
/// turns), and the hips at `hips(phase)`.
fn clip(
    duration: f32,
    keys: u32,
    angle: impl Fn(u16, f32) -> f32,
    hips: impl Fn(f32) -> Vec3,
) -> Result<AnimationClip, String> {
    let times: Vec<f32> = (0..=keys)
        .map(|key| duration * key as f32 / keys as f32)
        .collect();
    let phases: Vec<f32> = (0..=keys).map(|key| key as f32 / keys as f32).collect();
    let mut channels = Vec::with_capacity(JOINTS + 1);
    for joint in 0..JOINTS as u16 {
        let rotations: Vec<Quat> = phases
            .iter()
            .map(|&phase| {
                let turn = angle(joint, phase);
                if joint == SPINE {
                    Quat::from_rotation_y(turn)
                } else {
                    Quat::from_rotation_z(turn)
                }
            })
            .collect();
        channels.push(
            AnimationChannel::new(
                u32::from(joint),
                AnimatedProperty::Rotation,
                KeyInterpolation::Linear,
                times.clone(),
                Keyframes::Rotations(rotations.into_boxed_slice()),
            )
            .map_err(|error| error.to_string())?,
        );
    }
    let positions: Vec<Vec3> = phases.iter().map(|&phase| hips(phase)).collect();
    channels.push(
        AnimationChannel::new(
            u32::from(HIPS),
            AnimatedProperty::Translation,
            KeyInterpolation::Linear,
            times,
            Keyframes::Vectors(positions.into_boxed_slice()),
        )
        .map_err(|error| error.to_string())?,
    );
    AnimationClip::new(channels).map_err(|error| error.to_string())
}

/// One stride cycle a second: legs swing, knees bend on the way forward,
/// arms swing against the legs and the hips bob twice.
pub(crate) fn walk() -> Result<AnimationClip, String> {
    let rest = Vec3::from(SKELETON[0].1);
    clip(
        1.0,
        16,
        |joint, phase| {
            let swing = math::sin(phase * TAU);
            let knee = |offset: f32| -math::max(math::sin(phase * TAU + offset), 0.0) * 0.9;
            match joint {
                THIGH_L => swing * 0.5,
                THIGH_R => -swing * 0.5,
                SHIN_L => knee(0.6),
                SHIN_R => knee(0.6 + TAU * 0.5),
                ARM_L => -swing * 0.45,
                ARM_R => swing * 0.45,
                FOREARM_L | FOREARM_R => 0.35,
                SPINE => swing * 0.12,
                HEAD => -0.05,
                _ => 0.0,
            }
        },
        |phase| rest + Vec3::Y * (math::cos(phase * TAU * 2.0) * 0.025),
    )
}

/// Standing: a slow sway, breathing and a glance around.
pub(crate) fn idle() -> Result<AnimationClip, String> {
    let rest = Vec3::from(SKELETON[0].1);
    clip(
        4.0,
        8,
        |joint, phase| {
            let breath = math::sin(phase * TAU);
            match joint {
                ARM_L => -0.08 - breath * 0.03,
                ARM_R => 0.08 + breath * 0.03,
                FOREARM_L | FOREARM_R => 0.2,
                SPINE => math::sin(phase * TAU * 0.5) * 0.25,
                HEAD => breath * 0.08,
                _ => 0.0,
            }
        },
        |phase| rest + Vec3::Y * (math::sin(phase * TAU) * 0.01),
    )
}
