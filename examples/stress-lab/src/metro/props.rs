//! Keyframed clips of the metro's animated models (decision 035): the city
//! places each model's parts under a node and a player binds them, so every
//! clip animates local transforms and serves every copy of its model.

use std::f32::consts::TAU;

use engine::prelude::*;

use super::materials::Clip;
use super::models::{FLOOR, TOWER_FLOORS};

fn channel(
    target: u32,
    times: &[f32],
    keys: Keyframes,
    property: AnimatedProperty,
    interpolation: KeyInterpolation,
) -> Result<AnimationChannel, String> {
    AnimationChannel::new(target, property, interpolation, times.to_vec(), keys)
        .map_err(|error| error.to_string())
}

fn turns(
    target: u32,
    times: &[f32],
    rotations: Vec<Quat>,
    interpolation: KeyInterpolation,
) -> Result<AnimationChannel, String> {
    channel(
        target,
        times,
        Keyframes::Rotations(rotations.into_boxed_slice()),
        AnimatedProperty::Rotation,
        interpolation,
    )
}

fn moves(target: u32, times: &[f32], positions: Vec<Vec3>) -> Result<AnimationChannel, String> {
    channel(
        target,
        times,
        Keyframes::Vectors(positions.into_boxed_slice()),
        AnimatedProperty::Translation,
        KeyInterpolation::Linear,
    )
}

/// A full turn about `axis` in `period` seconds, keyed every quarter.
fn spin(target: u32, axis: Vec3, period: f32) -> Result<AnimationChannel, String> {
    let times: Vec<f32> = (0..=4).map(|key| period * key as f32 / 4.0).collect();
    let rotations = (0..=4)
        .map(|key| Quat::from_axis_angle(axis, TAU * key as f32 / 4.0))
        .collect();
    turns(target, &times, rotations, KeyInterpolation::Linear)
}

/// Height a gondola rides to on towers of class `class`.
pub(crate) fn gondola_reach(class: u8) -> f32 {
    TOWER_FLOORS[class as usize] as f32 * FLOOR - 3.0
}

pub(crate) fn clip(clip: Clip) -> Result<AnimationClip, String> {
    let channels = match clip {
        Clip::Flag => {
            // Flapping on the pole: smooth cubic swings.
            let times = [0.0, 0.6, 1.2, 1.8, 2.4];
            let angles = [0.0, 0.35, -0.1, 0.3, 0.0];
            let mut rotations = Vec::new();
            for (index, angle) in angles.iter().enumerate() {
                let tangent = if index % 2 == 0 { 0.15 } else { -0.15 };
                rotations.push(Quat::from_xyzw(0.0, tangent, 0.0, 0.0));
                rotations.push(Quat::from_rotation_y(*angle) * Quat::from_rotation_x(angle * 0.3));
                rotations.push(Quat::from_xyzw(0.0, tangent, 0.0, 0.0));
            }
            vec![turns(0, &times, rotations, KeyInterpolation::CubicSpline)?]
        }
        Clip::Dish => {
            let times = [0.0, 4.0, 7.0, 12.0, 16.0];
            let angles = [0.0, 1.1, 1.1, -0.7, 0.0];
            vec![turns(
                0,
                &times,
                angles
                    .iter()
                    .map(|angle| Quat::from_rotation_y(*angle))
                    .collect(),
                KeyInterpolation::Linear,
            )?]
        }
        Clip::Trivision => {
            // Hold each face, then turn a third about the prism's axis.
            let times = [0.0, 3.0, 3.6, 6.6, 7.2, 10.2, 10.8, 12.0];
            let thirds = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
            vec![turns(
                0,
                &times,
                thirds
                    .iter()
                    .map(|third| Quat::from_rotation_x(TAU / 3.0 * third))
                    .collect(),
                KeyInterpolation::Linear,
            )?]
        }
        Clip::Drawbridge => {
            // Both leaves rise, hold, lower and rest.
            let times = [0.0, 20.0, 26.0, 36.0, 42.0, 60.0];
            let lift = [0.0, 0.0, 1.2, 1.2, 0.0, 0.0];
            // The east leaf faces west: it rises by turning the other way.
            let rotations = |sign: f32| {
                lift.iter()
                    .map(|angle| Quat::from_rotation_z(angle * sign))
                    .collect()
            };
            vec![
                turns(0, &times, rotations(1.0), KeyInterpolation::Linear)?,
                turns(1, &times, rotations(-1.0), KeyInterpolation::Linear)?,
            ]
        }
        Clip::Gondola(class) => {
            // Down a tower face in steps, then back up.
            let top = gondola_reach(class);
            let mut times = Vec::new();
            let mut positions = Vec::new();
            let stops = 6;
            for stop in 0..=stops {
                let y = top - (top - 1.0) * stop as f32 / stops as f32;
                let at = stop as f32 * 8.0;
                times.extend([at, at + 5.0]);
                positions.extend([Vec3::Y * y, Vec3::Y * y]);
            }
            let end = times[times.len() - 1];
            times.push(end + 20.0);
            positions.push(Vec3::Y * top);
            vec![moves(0, &times, positions)?]
        }
        Clip::Door => vec![spin(0, Vec3::Y, 8.0)?],
        Clip::Turbine => vec![spin(0, Vec3::X, 4.0)?],
        Clip::Carousel => vec![spin(0, Vec3::Y, 12.0)?],
        Clip::Ring => vec![spin(0, Vec3::Y, 90.0)?],
        Clip::Excavator => {
            // Swing to dig, lower, curl, lift and swing back to dump.
            let times = [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0];
            let swing = [0.0, 0.0, 0.0, 0.0, 1.4, 1.4, 0.0];
            let boom = [0.35, 0.05, -0.1, 0.3, 0.45, 0.45, 0.35];
            let stick = [-0.6, -0.2, 0.3, 0.5, -0.2, -0.8, -0.6];
            let rotations = |angles: &[f32], axis: Vec3| -> Vec<Quat> {
                angles
                    .iter()
                    .map(|angle| Quat::from_axis_angle(axis, *angle))
                    .collect()
            };
            vec![
                turns(
                    0,
                    &times,
                    rotations(&swing, Vec3::Y),
                    KeyInterpolation::Linear,
                )?,
                turns(
                    1,
                    &times,
                    rotations(&boom, Vec3::Z),
                    KeyInterpolation::Linear,
                )?,
                turns(
                    2,
                    &times,
                    rotations(&stick, Vec3::Z),
                    KeyInterpolation::Linear,
                )?,
            ]
        }
        Clip::Walk | Clip::Idle => return Err("people clips come from people.rs".into()),
    };
    AnimationClip::new(channels).map_err(|error| error.to_string())
}
