use engine_nav::{NavMesh, Spot};
use glam::{Vec2, Vec3};

use crate::play::numbers::NavNumbers;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cover {
    pub center: Vec3,
    pub half: Vec3,
    pub eye: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tactics {
    covers: Vec<Cover>,
    numbers: NavNumbers,
}

impl Tactics {
    pub fn build(covers: Vec<Cover>, numbers: &NavNumbers) -> Tactics {
        Tactics {
            covers,
            numbers: numbers.clone(),
        }
    }

    pub fn best_cover_from(
        &self,
        mesh: &NavMesh,
        threat: Vec3,
        near: Spot,
        radius: f32,
        max_distance: f32,
        mut sees: impl FnMut(Vec3, Vec3) -> bool,
    ) -> Option<Spot> {
        let reach = self.numbers.cover_gap_m + mesh.weld();
        let mut best: Option<(f32, Spot)> = None;
        for &Cover { center, half, eye: height } in &self.covers {
            for at in behind(center, half, threat, radius + self.numbers.cover_gap_m) {
                let limit = best.map_or(max_distance, |(distance, _)| distance);
                if at.distance(flat(near.point)) - reach > limit {
                    continue;
                }
                let base = Vec3::new(at.x, center.y - half.y, at.y);
                let Some(spot) = mesh.locate(base, self.numbers.locate_height_m, reach) else {
                    continue;
                };
                let distance = spot.point.distance(near.point);
                let closer = best.map_or(distance <= max_distance, |(best, _)| distance < best);
                if !closer {
                    continue;
                }
                let eye = spot.point + Vec3::Y * height;
                if segment_hits_box(eye, threat, center - half, center + half)
                    && mesh.joined(spot, near)
                    && mesh.distance_to_edge(spot) >= radius - mesh.weld()
                    && !sees(eye, threat)
                {
                    best = Some((distance, spot));
                }
            }
        }
        best.map(|(_, spot)| spot)
    }
}

fn flat(point: Vec3) -> Vec2 {
    Vec2::new(point.x, point.z)
}

fn behind(center: Vec3, half: Vec3, threat: Vec3, room: f32) -> impl Iterator<Item = Vec2> {
    let middle = flat(center);
    let away = middle - flat(threat);
    let length = away.length();
    let away = if length > 0.0 { away / length } else { Vec2::ZERO };
    let reach = Vec2::new(half.x + room, half.z + room);
    let along = |extent: f32, part: f32| {
        if part == 0.0 {
            f32::INFINITY
        } else {
            extent / part.abs()
        }
    };
    let t = engine_math::min(along(reach.x, away.x), along(reach.y, away.y));
    let first = (length > 0.0).then_some(middle + away * t);
    let corners = [
        Vec2::new(-1.0, -1.0),
        Vec2::new(1.0, -1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(-1.0, 1.0),
    ]
    .into_iter()
    .map(move |sign| reach * sign)
    .filter(move |corner| corner.dot(away) > 0.0)
    .map(move |corner| middle + corner);
    first.into_iter().chain(corners)
}

fn segment_hits_box(from: Vec3, to: Vec3, min: Vec3, max: Vec3) -> bool {
    let along = to - from;
    let mut enter = 0.0_f32;
    let mut leave = 1.0_f32;
    for axis in 0..3 {
        let (start, step, low, high) = (from[axis], along[axis], min[axis], max[axis]);
        if step == 0.0 {
            if start < low || start > high {
                return false;
            }
            continue;
        }
        let (a, b) = ((low - start) / step, (high - start) / step);
        enter = engine_math::max(enter, engine_math::min(a, b));
        leave = engine_math::min(leave, engine_math::max(a, b));
        if enter > leave {
            return false;
        }
    }
    true
}
