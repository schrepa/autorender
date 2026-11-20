use engine_core::glam::{Mat3, Quat, Vec3};
use engine_core::{Error, ErrorCode, Expected};
use engine_rig::{Posed, Skeleton};

use super::rig::Rig;

pub(super) struct Tree {
    below: Vec<Vec<usize>>,
}

impl Tree {
    pub(super) fn of(skeleton: &Skeleton) -> Self {
        let mut below = vec![Vec::new(); skeleton.len()];
        for joint in 0..skeleton.len() {
            let mut held = Some(joint);
            while let Some(at) = held {
                below[at].push(joint);
                held = skeleton.joints()[at].parent;
            }
        }
        Self { below }
    }

    fn holds(&self, top: usize, joint: usize) -> bool {
        self.below[top].binary_search(&joint).is_ok()
    }
}

fn nlerp(from: Quat, to: Quat, fraction: f32) -> Quat {
    let to = if from.dot(to) < 0.0 { -to } else { to };
    (from + (to - from) * fraction).normalize()
}

fn length(skeleton: &Skeleton, item: &str, got: usize) -> Result<(), Error> {
    if got == skeleton.len() {
        return Ok(());
    }
    Err(Error::new(ErrorCode::InvalidInput, format!("$.{item}"))
        .got(format!(
            "Length: {item} holds {got}, expected {}",
            skeleton.len()
        ))
        .expected(Expected::Matching))
}

fn index(skeleton: &Skeleton, joint: usize) -> Result<(), Error> {
    if joint < skeleton.len() {
        return Ok(());
    }
    Err(Error::new(ErrorCode::OutOfRange, "$.joint")
        .got(format!("JointIndex: joint {joint} of {}", skeleton.len()))
        .expected(Expected::Present))
}

fn turn_below(tree: &Tree, posed: &mut [Posed], top: usize, turn: Quat) {
    let pivot = posed[top].position;
    for &joint in &tree.below[top] {
        let held = &mut posed[joint];
        *held = Posed {
            rotation: (turn * held.rotation).normalize(),
            position: pivot + turn * (held.position - pivot),
        };
    }
}

pub(super) fn turn(rig: &Rig, posed: &mut [Posed], turns: &[(usize, Quat)]) -> Result<(), Error> {
    length(&rig.skeleton, "turned joints", posed.len())?;
    for &(top, turn) in turns {
        index(&rig.skeleton, top)?;
        turn_below(&rig.tree, posed, top, turn);
    }
    Ok(())
}

pub(super) fn shift(rig: &Rig, posed: &mut [Posed], top: usize, offset: Vec3) -> Result<(), Error> {
    length(&rig.skeleton, "shifted joints", posed.len())?;
    index(&rig.skeleton, top)?;
    for &joint in &rig.tree.below[top] {
        posed[joint].position += offset;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Reach {
    pub(super) target: Vec3,
    pub(super) pole: Vec3,
    pub(super) end_rotation: Option<Quat>,
    pub(super) weight: f32,
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

pub(super) fn reach(
    rig: &Rig,
    posed: &mut [Posed],
    chain: [usize; 3],
    reach: &Reach,
) -> Result<(), Error> {
    let tree = &rig.tree;
    length(&rig.skeleton, "reaching joints", posed.len())?;
    let [upper, middle, end] = chain;
    for joint in chain {
        index(&rig.skeleton, joint)?;
    }
    let ordered =
        upper < middle && middle < end && tree.holds(upper, middle) && tree.holds(middle, end);
    let lengths_held = (posed[middle].position - posed[upper].position).length_squared() > 0.0
        && (posed[end].position - posed[middle].position).length_squared() > 0.0;
    if !ordered || !lengths_held {
        return Err(Error::new(ErrorCode::InvalidInput, "$.chain")
            .got(format!(
                "Chain: joints {upper}, {middle}, {end} are not an ancestor chain with two bones of length"
            ))
            .expected(Expected::Acyclic));
    }
    let Some(goal) = goal(posed, chain, reach) else {
        return Ok(());
    };
    let weighted = |turn: Quat| nlerp(Quat::IDENTITY, turn, reach.weight);
    turn_below(tree, posed, upper, weighted(goal.bend));
    let now = (posed[end].position - posed[middle].position).normalize();
    let wanted = (goal.end - goal.middle).normalize();
    turn_below(
        tree,
        posed,
        middle,
        weighted(Quat::from_rotation_arc(now, wanted)),
    );
    if let Some(rotation) = reach.end_rotation {
        let turn = (rotation * posed[end].rotation.conjugate()).normalize();
        turn_below(tree, posed, end, weighted(turn));
    }
    Ok(())
}
