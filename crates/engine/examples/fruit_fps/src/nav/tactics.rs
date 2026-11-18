use glam::{Vec2, Vec3};

use super::{Costs, NavMesh, NavNumbers, PolygonId, Spot, flat, greater, lesser};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoverClass {
    Full,
    Half,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlaceShape {
    Area {
        min: Vec3,
        max: Vec3,
    },
    Choke {
        a: Vec3,
        b: Vec3,
    },
    Cover {
        center: Vec3,
        half: Vec3,
        class: CoverClass,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    pub name: String,
    pub shape: PlaceShape,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlaceFacts {
    pub place: Place,
    pub polygons: Vec<PolygonId>,
    pub anchor: Option<Spot>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eyes {
    pub stand: f32,
    pub crouch: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoverSpot {
    pub place: usize,
    pub spot: Spot,
    pub eye: Vec3,
    pub class: CoverClass,
    pub distance: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tactics {
    facts: Vec<PlaceFacts>,
    travel: Vec<Option<f32>>,
    numbers: NavNumbers,
}

impl Tactics {
    pub fn build(mesh: &NavMesh, places: Vec<Place>, radius: f32, numbers: &NavNumbers) -> Tactics {
        let facts: Vec<PlaceFacts> = places
            .into_iter()
            .map(|place| PlaceFacts {
                polygons: covered(mesh, &place.shape, radius, numbers),
                anchor: anchor(mesh, &place.shape, radius, numbers),
                place,
            })
            .collect();
        let count = facts.len();
        let costs = Costs::new(mesh);
        let mut travel = vec![None; count * count];
        for i in 0..count {
            travel[i * count + i] = facts[i].anchor.map(|_| 0.0);
            for j in i + 1..count {
                let length = match (facts[i].anchor, facts[j].anchor) {
                    (Some(a), Some(b)) => mesh.path(a, b, radius, &costs).map(|p| p.length),
                    _ => None,
                };
                travel[i * count + j] = length;
                travel[j * count + i] = length;
            }
        }
        Tactics {
            facts,
            travel,
            numbers: numbers.clone(),
        }
    }

    pub fn place(&self, name: &str) -> Option<usize> {
        self.facts.iter().position(|f| f.place.name == name)
    }

    pub fn places(&self) -> &[PlaceFacts] {
        &self.facts
    }

    pub fn travel(&self, from: usize, to: usize) -> Option<f32> {
        let count = self.facts.len();
        if from >= count || to >= count {
            return None;
        }
        self.travel[from * count + to]
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
        for (index, facts) in self.facts.iter().enumerate() {
            let PlaceShape::Cover {
                center,
                half,
                class,
            } = facts.place.shape
            else {
                continue;
            };
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
                    let place = index;
                    found.push((
                        CoverSpot {
                            place,
                            spot,
                            eye,
                            class,
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
                .then(x.place.cmp(&y.place))
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

fn covered(
    mesh: &NavMesh,
    shape: &PlaceShape,
    radius: f32,
    numbers: &NavNumbers,
) -> Vec<PolygonId> {
    let reach = numbers.locate_height_m;
    let (outline, low, high) = match *shape {
        PlaceShape::Area { min, max } => (rectangle(flat(min), flat(max)), min.y, max.y),
        PlaceShape::Choke { a, b } => (
            vec![flat(a), flat(b)],
            lesser(a.y, b.y) - reach,
            greater(a.y, b.y) + reach,
        ),
        PlaceShape::Cover { center, half, .. } => {
            let grow = Vec2::splat(radius + numbers.cover_gap_m);
            let extent = Vec2::new(half.x, half.z) + grow;
            let base = center.y - half.y;
            (
                rectangle(flat(center) - extent, flat(center) + extent),
                base - reach,
                base + reach,
            )
        }
    };
    let (min, max) = outline.iter().fold(
        (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    );
    mesh.grid
        .gather(min, max)
        .into_iter()
        .filter(|&id| {
            let polygon = mesh.polygon(id);
            let level = polygon.low <= high && polygon.high >= low;
            level && overlap(&flat_points(mesh, id), &outline)
        })
        .collect()
}

fn anchor(mesh: &NavMesh, shape: &PlaceShape, radius: f32, numbers: &NavNumbers) -> Option<Spot> {
    let height = numbers.locate_height_m;
    match *shape {
        PlaceShape::Area { min, max } => {
            let middle = (min + max) * 0.5;
            let spread = (flat(max) - flat(min)).max_element() * 0.5;
            mesh.locate(Vec3::new(middle.x, min.y, middle.z), height, spread)
        }
        PlaceShape::Choke { a, b } => mesh.locate((a + b) * 0.5, height, numbers.locate_distance_m),
        PlaceShape::Cover { center, half, .. } => {
            let spread = Vec2::new(half.x, half.z).length() + radius + numbers.cover_gap_m;
            mesh.locate(center - Vec3::Y * half.y, height, spread)
        }
    }
}

fn rectangle(min: Vec2, max: Vec2) -> Vec<Vec2> {
    vec![min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)]
}

fn flat_points(mesh: &NavMesh, polygon: PolygonId) -> Vec<Vec2> {
    mesh.polygon_points(polygon).into_iter().map(flat).collect()
}

fn overlap(a: &[Vec2], b: &[Vec2]) -> bool {
    !separated(a, b) && !separated(b, a)
}

fn separated(shape: &[Vec2], other: &[Vec2]) -> bool {
    (0..shape.len()).any(|i| {
        let edge = shape[(i + 1) % shape.len()] - shape[i];
        let axis = Vec2::new(edge.y, -edge.x);
        let project = |points: &[Vec2]| {
            points
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| {
                    let d = axis.dot(*p);
                    (lesser(lo, d), greater(hi, d))
                })
        };
        let (a0, a1) = project(shape);
        let (b0, b1) = project(other);
        a1 < b0 || b1 < a0
    })
}
