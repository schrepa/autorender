use glam::{Vec2, Vec3};

use super::Capsule;
use super::shapes::{least, most, within};

const TINY: f32 = 1e-12;
const HALF_ROOT: f32 = std::f32::consts::FRAC_1_SQRT_2;
const SPREAD: [Vec3; 8] = [
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(HALF_ROOT, 0.0, HALF_ROOT),
    Vec3::new(0.0, 0.0, 1.0),
    Vec3::new(-HALF_ROOT, 0.0, HALF_ROOT),
    Vec3::new(-1.0, 0.0, 0.0),
    Vec3::new(-HALF_ROOT, 0.0, -HALF_ROOT),
    Vec3::new(0.0, 0.0, -1.0),
    Vec3::new(HALF_ROOT, 0.0, -HALF_ROOT),
];

struct Axis {
    x: f32,
    z: f32,
    low: f32,
    high: f32,
    radius: f32,
}

fn span(capsule: Capsule) -> (Vec3, f32) {
    (
        capsule.feet + Vec3::Y * capsule.radius,
        most(capsule.height - 2.0 * capsule.radius, 0.0),
    )
}

fn joined(moving: Capsule, still: Capsule) -> (Vec3, Axis) {
    let (bottom, lift) = span(moving);
    let (base, rise) = span(still);
    (
        bottom,
        Axis {
            x: base.x,
            z: base.z,
            low: base.y - lift,
            high: base.y + rise,
            radius: moving.radius + still.radius,
        },
    )
}

fn closest(axis: &Axis, point: Vec3) -> Vec3 {
    Vec3::new(axis.x, within(point.y, axis.low, axis.high), axis.z)
}

fn earliest(a: f32, b: f32, c: f32) -> Option<f32> {
    let disc = b * b - a * c;
    (a > TINY && c >= 0.0 && disc >= 0.0).then(|| (-b - disc.sqrt()) / a)
}

fn touch(axis: &Axis, origin: Vec3, motion: Vec3) -> Option<(f32, Vec3)> {
    let at = |t: f32| origin + motion * t;
    let mut best: Option<f32> = None;
    let mut keep = |t: f32| {
        if (0.0..=1.0).contains(&t) && best.is_none_or(|held| t < held) {
            best = Some(t);
        }
    };
    let side = Vec2::new(origin.x - axis.x, origin.z - axis.z);
    let level = Vec2::new(motion.x, motion.z);
    if let Some(t) = earliest(
        level.dot(level),
        side.dot(level),
        side.dot(side) - axis.radius * axis.radius,
    ) && (axis.low..=axis.high).contains(&at(t).y)
    {
        keep(t);
    }
    for (centre, below) in [(axis.low, true), (axis.high, false)] {
        let from = origin - Vec3::new(axis.x, centre, axis.z);
        if let Some(t) = earliest(
            motion.dot(motion),
            from.dot(motion),
            from.dot(from) - axis.radius * axis.radius,
        ) && (if below {
            at(t).y <= centre
        } else {
            at(t).y >= centre
        }) {
            keep(t);
        }
    }
    let t = best?;
    let point = at(t);
    let normal = (point - closest(axis, point)) / axis.radius;
    (normal.dot(motion) < 0.0).then_some((t, normal))
}

pub(super) fn sweep(moving: Capsule, motion: Vec3, peers: &[Capsule]) -> Option<(f32, Vec3)> {
    let mut best: Option<(f32, Vec3)> = None;
    for still in peers {
        let (origin, axis) = joined(moving, *still);
        if let Some(found) = touch(&axis, origin, motion)
            && best.is_none_or(|held| found.0 < held.0)
        {
            best = Some(found);
        }
    }
    best
}

pub(super) fn overlap(moving: Capsule, still: Capsule) -> Option<f32> {
    let (point, axis) = joined(moving, still);
    let distance = (point - closest(&axis, point)).length();
    (distance < axis.radius).then_some(axis.radius - distance)
}

fn away(own: (u32, Capsule), other: (u32, Capsule)) -> Vec3 {
    let level = Vec3::new(
        own.1.feet.x - other.1.feet.x,
        0.0,
        own.1.feet.z - other.1.feet.z,
    );
    if level.length_squared() > TINY {
        return level.normalize();
    }
    let (low, high) = (own.0.min(other.0), own.0.max(other.0));
    let spread = SPREAD[((low * 7 + high * 3) % 8) as usize];
    if own.0 == low { spread } else { -spread }
}

pub fn apart(own: (u32, Capsule), peers: &[(u32, Capsule)], reach: f32) -> Vec3 {
    let mut shove = Vec3::ZERO;
    for &other in peers {
        if other.0 == own.0 {
            continue;
        }
        if let Some(depth) = overlap(own.1, other.1) {
            shove += away(own, other) * least(depth, reach);
        }
    }
    let length = shove.length();
    if length > reach && length > 0.0 {
        shove * (reach / length)
    } else {
        shove
    }
}
