use glam::{Mat3, Quat, Vec3};

use crate::bound::nlerp;
use engine_codec::Error;

use crate::error::Fault;
use crate::{Posed, Skeleton};

fn lengths(skeleton: &Skeleton, held: &[(&'static str, usize)]) -> Result<(), Error> {
    for &(item, got) in held {
        if got != skeleton.len() {
            return Err(Error::from(Fault::Length {
                item,
                expected: skeleton.len(),
                got,
            }));
        }
    }
    Ok(())
}

pub fn blend(
    skeleton: &Skeleton,
    from: &[Posed],
    to: &[Posed],
    weights: &[f32],
    root_motion: Option<usize>,
    out: &mut [Posed],
) -> Result<(), Error> {
    lengths(
        skeleton,
        &[
            ("blend from", from.len()),
            ("blend to", to.len()),
            ("blend weights", weights.len()),
            ("blended joints", out.len()),
        ],
    )?;
    for joint in 0..skeleton.len() {
        let weight = weights[joint];
        let (a, b) = (from[joint], to[joint]);
        let rotation = nlerp(a.rotation, b.rotation, weight);
        let parent = skeleton.joints()[joint].parent;
        let position = match parent {
            Some(parent) if root_motion != Some(joint) => {
                let offset = |pose: &[Posed]| {
                    pose[parent].rotation.conjugate()
                        * (pose[joint].position - pose[parent].position)
                };
                let (near, far) = (offset(from), offset(to));
                out[parent].position + out[parent].rotation * (near + (far - near) * weight)
            }
            Some(_) | None => a.position + (b.position - a.position) * weight,
        };
        out[joint] = Posed { rotation, position };
    }
    Ok(())
}

fn index(skeleton: &Skeleton, joint: usize) -> Result<(), Error> {
    if joint < skeleton.len() {
        Ok(())
    } else {
        Err(Error::from(Fault::JointIndex {
            joint,
            joints: skeleton.len(),
        }))
    }
}

fn turn_below(skeleton: &Skeleton, posed: &mut [Posed], top: usize, turn: Quat) {
    let pivot = posed[top].position;
    for (joint, held) in posed.iter_mut().enumerate().skip(top) {
        if skeleton.below(top, joint) {
            *held = Posed {
                rotation: (turn * held.rotation).normalize(),
                position: pivot + turn * (held.position - pivot),
            };
        }
    }
}

pub fn turn(
    skeleton: &Skeleton,
    posed: &mut [Posed],
    turns: &[(usize, Quat)],
) -> Result<(), Error> {
    lengths(skeleton, &[("turned joints", posed.len())])?;
    for &(top, turn) in turns {
        index(skeleton, top)?;
        turn_below(skeleton, posed, top, turn);
    }
    Ok(())
}

pub fn shift(
    skeleton: &Skeleton,
    posed: &mut [Posed],
    top: usize,
    offset: Vec3,
) -> Result<(), Error> {
    lengths(skeleton, &[("shifted joints", posed.len())])?;
    index(skeleton, top)?;
    for (joint, held) in posed.iter_mut().enumerate().skip(top) {
        if skeleton.below(top, joint) {
            held.position += offset;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reach {
    pub target: Vec3,
    pub pole: Vec3,
    pub end_rotation: Option<Quat>,
    pub weight: f32,
}

fn held_between(value: f32, low: f32, high: f32) -> f32 {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}

fn frame(first: Vec3, normal: Vec3) -> Mat3 {
    let first = first.normalize();
    let normal = normal.normalize();
    Mat3::from_cols(first, normal, first.cross(normal))
}

struct Goal {
    middle: Vec3,
    end: Vec3,
    bend: Quat,
}

fn goal(posed: &[Posed], [upper, middle, end]: [usize; 3], reach: &Reach) -> Option<Goal> {
    let (a, b, c) = (
        posed[upper].position,
        posed[middle].position,
        posed[end].position,
    );
    let (near, far) = ((b - a).length(), (c - b).length());
    let toward = reach.target - a;
    let span = toward.length();
    let along = if span > 0.0 {
        toward / span
    } else {
        (c - a).try_normalize()?
    };
    let distance = held_between(span, (near - far).abs(), near + far);
    let flat = |direction: Vec3| direction - along * direction.dot(along);
    let side = [flat(reach.pole), flat(b - a)]
        .into_iter()
        .find(|direction| direction.length_squared() > 0.0)
        .map_or_else(|| along.any_orthonormal_vector(), Vec3::normalize);
    let out = if distance > 0.0 {
        (near * near - far * far + distance * distance) / (2.0 * distance)
    } else {
        0.0
    };
    let square = near * near - out * out;
    let up = if square > 0.0 { square.sqrt() } else { 0.0 };
    let middle_goal = a + along * out + side * up;
    let bent = (b - a).cross(c - a);
    let bent = if bent.length_squared() > 0.0 {
        bent
    } else {
        side.cross(b - a)
    };
    let from = frame(b - a, bent);
    let to = frame(middle_goal - a, side.cross(along));
    Some(Goal {
        middle: middle_goal,
        end: a + along * distance,
        bend: Quat::from_mat3(&(to * from.transpose())).normalize(),
    })
}

pub fn reach(
    skeleton: &Skeleton,
    posed: &mut [Posed],
    chain: [usize; 3],
    reach: &Reach,
) -> Result<(), Error> {
    lengths(skeleton, &[("reaching joints", posed.len())])?;
    let [upper, middle, end] = chain;
    for joint in chain {
        index(skeleton, joint)?;
    }
    let ordered = upper < middle
        && middle < end
        && skeleton.below(upper, middle)
        && skeleton.below(middle, end);
    let lengths_held = (posed[middle].position - posed[upper].position).length_squared() > 0.0
        && (posed[end].position - posed[middle].position).length_squared() > 0.0;
    if !ordered || !lengths_held {
        return Err(Error::from(Fault::Chain { upper, middle, end }));
    }
    let Some(goal) = goal(posed, chain, reach) else {
        return Ok(());
    };
    let weighted = |turn: Quat| nlerp(Quat::IDENTITY, turn, reach.weight);
    turn_below(skeleton, posed, upper, weighted(goal.bend));
    let now = (posed[end].position - posed[middle].position).normalize();
    let wanted = (goal.end - goal.middle).normalize();
    turn_below(
        skeleton,
        posed,
        middle,
        weighted(Quat::from_rotation_arc(now, wanted)),
    );
    if let Some(rotation) = reach.end_rotation {
        let turn = (rotation * posed[end].rotation.conjugate()).normalize();
        turn_below(skeleton, posed, end, weighted(turn));
    }
    Ok(())
}
