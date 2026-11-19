use glam::{Vec2, Vec3};

use super::{NavMesh, NavNumbers, Spot, flat, greater, lesser};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoverClass {
    Full,
    Half,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cover {
    pub center: Vec3,
    pub half: Vec3,
    pub class: CoverClass,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eyes {
    pub stand: f32,
    pub crouch: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoverSpot {
    pub cover: usize,
    pub spot: Spot,
    pub distance: f32,
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

    #[expect(clippy::too_many_arguments, reason = "the published query shape")]
    pub fn positions_with_cover_from(
        &self,
        mesh: &NavMesh,
        threat: Vec3,
        near: Spot,
        eyes: Eyes,
        radius: f32,
        max_distance: f32,
        mut sees: impl FnMut(Vec3, Vec3) -> bool,
    ) -> Vec<CoverSpot> {
        let mut found: Vec<(CoverSpot, usize)> = Vec::new();
        for (index, &Cover {
            center,
            half,
            class,
        }) in self.covers.iter().enumerate()
        {
            let eye_height = match class {
                CoverClass::Full => eyes.stand,
                CoverClass::Half => eyes.crouch,
            };
            let candidates = behind(center, half, threat, radius + self.numbers.cover_gap_m);
            for (candidate, at) in candidates.into_iter().enumerate() {
                let base = Vec3::new(at.x, center.y - half.y, at.y);
                let Some(spot) = mesh.locate(
                    base,
                    self.numbers.locate_height_m,
                    self.numbers.cover_gap_m + mesh.weld,
                ) else {
                    continue;
                };
                let roomy = mesh.distance_to_edge(spot) >= radius - mesh.weld;
                let joined = mesh.island(spot.polygon) == mesh.island(near.polygon);
                let eye = spot.point + Vec3::Y * eye_height;
                let shielded = segment_hits_box(eye, threat, center - half, center + half);
                let distance = spot.point.distance(near.point);
                if roomy && joined && shielded && distance <= max_distance && !sees(eye, threat) {
                    found.push((
                        CoverSpot {
                            cover: index,
                            spot,
                            distance,
                        },
                        candidate,
                    ));
                }
            }
        }
        found.sort_by(|(x, i), (y, j)| {
            x.distance
                .total_cmp(&y.distance)
                .then(x.cover.cmp(&y.cover))
                .then(i.cmp(j))
        });
        found.into_iter().map(|(spot, _)| spot).collect()
    }
}

fn behind(center: Vec3, half: Vec3, threat: Vec3, room: f32) -> Vec<Vec2> {
    let middle = flat(center);
    let away = middle - flat(threat);
    let length = away.length();
    if length <= 0.0 {
        return Vec::new();
    }
    let away = away / length;
    let reach = Vec2::new(half.x + room, half.z + room);
    let along = |extent: f32, part: f32| {
        if part == 0.0 {
            f32::INFINITY
        } else {
            extent / part.abs()
        }
    };
    let t = lesser(along(reach.x, away.x), along(reach.y, away.y));
    let mut spots = vec![middle + away * t];
    for sign in [
        Vec2::new(-1.0, -1.0),
        Vec2::new(1.0, -1.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(-1.0, 1.0),
    ] {
        let corner = reach * sign;
        if corner.dot(away) > 0.0 {
            spots.push(middle + corner);
        }
    }
    spots
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
        enter = greater(enter, lesser(a, b));
        leave = lesser(leave, greater(a, b));
        if enter > leave {
            return false;
        }
    }
    true
}
