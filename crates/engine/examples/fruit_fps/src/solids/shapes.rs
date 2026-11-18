use glam::Vec3;

const TINY: f32 = 1e-12;

pub(super) fn least(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}

pub(super) fn most(a: f32, b: f32) -> f32 {
    if a < b { b } else { a }
}

pub(super) fn within(x: f32, low: f32, high: f32) -> f32 {
    least(most(x, low), high)
}

const EDGE_SLACK: f32 = 1e-6;
const INWARD: f32 = 1e-6;
const DEGENERATE_CROSS: f32 = 1e-8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Triangle {
    pub a: Vec3,
    pub b: Vec3,
    pub c: Vec3,
    pub normal: Vec3,
}

impl Triangle {
    pub fn new(a: Vec3, b: Vec3, c: Vec3) -> Option<Triangle> {
        if !(a.is_finite() && b.is_finite() && c.is_finite()) {
            return None;
        }
        let cross = (b - a).cross(c - a);
        let length = cross.length();
        (length.is_finite() && length > DEGENERATE_CROSS).then(|| Triangle {
            a,
            b,
            c,
            normal: cross / length,
        })
    }

    pub fn low(&self) -> Vec3 {
        self.a.min(self.b).min(self.c)
    }

    pub fn high(&self) -> Vec3 {
        self.a.max(self.b).max(self.c)
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Swept {
    pub bottom: Vec3,
    pub lift: f32,
    pub radius: f32,
}

pub(super) fn facing(normal: Vec3, toward: Vec3) -> Vec3 {
    if normal.dot(toward) < 0.0 {
        -normal
    } else {
        normal
    }
}

pub(super) fn ray_triangle(origin: Vec3, along: Vec3, triangle: &Triangle) -> Option<f32> {
    let edge_b = triangle.b - triangle.a;
    let edge_c = triangle.c - triangle.a;
    let p = along.cross(edge_c);
    let det = edge_b.dot(p);
    if det == 0.0 {
        return None;
    }
    let inverse = 1.0 / det;
    let offset = origin - triangle.a;
    let u = offset.dot(p) * inverse;
    if !(-EDGE_SLACK..=1.0 + EDGE_SLACK).contains(&u) {
        return None;
    }
    let q = offset.cross(edge_b);
    let v = along.dot(q) * inverse;
    if v < -EDGE_SLACK || u + v > 1.0 + EDGE_SLACK {
        return None;
    }
    Some(edge_c.dot(q) * inverse)
}

pub(super) fn closest_on_triangle(p: Vec3, triangle: &Triangle) -> Vec3 {
    let (a, b, c) = (triangle.a, triangle.b, triangle.c);
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let scale = 1.0 / (va + vb + vc);
    a + ab * (vb * scale) + ac * (vc * scale)
}

fn closest_segments(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3) {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);
    if a <= TINY && e <= TINY {
        return (p1, p2);
    }
    let (s, t) = if a <= TINY {
        (0.0, within(f / e, 0.0, 1.0))
    } else {
        let c = d1.dot(r);
        if e <= TINY {
            (within(-c / a, 0.0, 1.0), 0.0)
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            let s = if denom > 0.0 {
                within((b * f - c * e) / denom, 0.0, 1.0)
            } else {
                0.0
            };
            let t = (b * s + f) / e;
            if t < 0.0 {
                (within(-c / a, 0.0, 1.0), 0.0)
            } else if t > 1.0 {
                (within((b - c) / a, 0.0, 1.0), 1.0)
            } else {
                (s, t)
            }
        }
    };
    (p1 + d1 * s, p2 + d2 * t)
}

pub(super) fn segment_triangle(p: Vec3, q: Vec3, triangle: &Triangle) -> (Vec3, Vec3) {
    let along = q - p;
    if let Some(t) = ray_triangle(p, along, triangle)
        && (0.0..=1.0).contains(&t)
    {
        let crossing = p + along * t;
        return (crossing, crossing);
    }
    let mut best = (p, closest_on_triangle(p, triangle));
    let mut best_gap = best.0.distance_squared(best.1);
    let mut offer = |pair: (Vec3, Vec3)| {
        let gap = pair.0.distance_squared(pair.1);
        if gap < best_gap {
            best_gap = gap;
            best = pair;
        }
    };
    offer((q, closest_on_triangle(q, triangle)));
    let corners = [triangle.a, triangle.b, triangle.c];
    for i in 0..3 {
        offer(closest_segments(p, q, corners[i], corners[(i + 1) % 3]));
    }
    best
}

fn keep(best: &mut Option<(f32, Vec3)>, found: Option<(f32, Vec3)>) {
    if let Some(found) = found
        && best.is_none_or(|held| found.0 < held.0)
    {
        *best = Some(found);
    }
}

fn entering(t: f32, lower: f32, normal: Vec3, motion: Vec3) -> Option<(f32, Vec3)> {
    (t >= lower && t <= 1.0 && normal.dot(motion) < 0.0).then_some((most(t, 0.0), normal))
}

fn sphere(
    origin: Vec3,
    motion: Vec3,
    centre: Vec3,
    radius: f32,
    lower: f32,
) -> Option<(f32, Vec3)> {
    let m = origin - centre;
    let a = motion.dot(motion);
    let b = m.dot(motion);
    let c = m.dot(m) - radius * radius;
    let disc = b * b - a * c;
    if a <= TINY || disc < 0.0 {
        return None;
    }
    let t = (-b - disc.sqrt()) / a;
    let at = origin + motion * most(t, 0.0);
    let normal = (at - centre).normalize_or_zero();
    entering(t, lower, normal, motion)
}

fn cylinder(
    origin: Vec3,
    motion: Vec3,
    ends: (Vec3, Vec3),
    radius: f32,
    lower: f32,
) -> Option<(f32, Vec3)> {
    let axis = ends.1 - ends.0;
    let m = origin - ends.0;
    let dd = axis.dot(axis);
    let nn = motion.dot(motion);
    let nd = motion.dot(axis);
    let md = m.dot(axis);
    let qa = dd * nn - nd * nd;
    if dd <= TINY || qa <= TINY * dd * nn {
        return None;
    }
    let qb = dd * m.dot(motion) - nd * md;
    let qc = dd * (m.dot(m) - radius * radius) - md * md;
    let disc = qb * qb - qa * qc;
    if disc < 0.0 {
        return None;
    }
    let t = (-qb - disc.sqrt()) / qa;
    let s = (md + most(t, 0.0) * nd) / dd;
    if !(0.0..=1.0).contains(&s) {
        return None;
    }
    let at = origin + motion * most(t, 0.0);
    let normal = (at - (ends.0 + axis * s)).normalize_or_zero();
    entering(t, lower, normal, motion)
}

fn inside(polygon: &[Vec3], normal: Vec3, point: Vec3) -> bool {
    (0..polygon.len()).all(|i| {
        let from = polygon[i];
        let to = polygon[(i + 1) % polygon.len()];
        (to - from).cross(point - from).dot(normal) >= 0.0
    })
}

fn face(
    polygon: &[Vec3],
    normal: Vec3,
    origin: Vec3,
    motion: Vec3,
    radius: f32,
    lower: f32,
) -> Option<(f32, Vec3)> {
    let mut best = None;
    for side in [normal, -normal] {
        let approach = side.dot(motion);
        if approach >= 0.0 {
            continue;
        }
        let height = side.dot(origin - polygon[0]) - radius;
        let t = height / -approach;
        if t < lower || t > 1.0 {
            continue;
        }
        let touch = origin + motion * most(t, 0.0) - side * radius;
        if inside(polygon, normal, touch) {
            keep(&mut best, Some((most(t, 0.0), side)));
        }
    }
    best
}

fn minkowski(triangle: &Triangle, shape: &Swept, motion: Vec3, lower: f32) -> Option<(f32, Vec3)> {
    let origin = shape.bottom;
    let radius = shape.radius;
    let corners = [triangle.a, triangle.b, triangle.c];
    let mut best = None;
    keep(
        &mut best,
        face(&corners, triangle.normal, origin, motion, radius, lower),
    );
    for i in 0..3 {
        let next = corners[(i + 1) % 3];
        keep(
            &mut best,
            cylinder(origin, motion, (corners[i], next), radius, lower),
        );
        keep(&mut best, sphere(origin, motion, corners[i], radius, lower));
    }
    if shape.lift <= 0.0 {
        return best;
    }
    let drop = Vec3::new(0.0, -shape.lift, 0.0);
    let lowered = corners.map(|corner| corner + drop);
    keep(
        &mut best,
        face(&lowered, triangle.normal, origin, motion, radius, lower),
    );
    for i in 0..3 {
        let j = (i + 1) % 3;
        keep(
            &mut best,
            cylinder(origin, motion, (lowered[i], lowered[j]), radius, lower),
        );
        keep(&mut best, sphere(origin, motion, lowered[i], radius, lower));
        keep(
            &mut best,
            cylinder(origin, motion, (corners[i], lowered[i]), radius, lower),
        );
        let side = (corners[j] - corners[i]).cross(drop);
        let length = side.length();
        if length > DEGENERATE_CROSS {
            let quad = [corners[i], corners[j], lowered[j], lowered[i]];
            keep(
                &mut best,
                face(&quad, side / length, origin, motion, radius, lower),
            );
        }
    }
    best
}

pub(super) fn sweep(
    triangle: &Triangle,
    shape: &Swept,
    motion: Vec3,
    lower: f32,
) -> Option<(f32, Vec3)> {
    let top = shape.bottom + Vec3::Y * shape.lift;
    let (on_axis, on_triangle) = segment_triangle(shape.bottom, top, triangle);
    let gap = on_axis - on_triangle;
    let gap_squared = gap.length_squared();
    if gap_squared < shape.radius * shape.radius {
        let normal = if gap_squared > TINY {
            gap / gap_squared.sqrt()
        } else {
            facing(triangle.normal, -motion)
        };
        return (normal.dot(motion) < -INWARD * motion.length()).then_some((0.0, normal));
    }
    minkowski(triangle, shape, motion, lower)
}

pub(super) fn penetration(
    triangle: &Triangle,
    bottom: Vec3,
    top: Vec3,
    radius: f32,
) -> Option<(f32, Vec3)> {
    let (on_axis, on_triangle) = segment_triangle(bottom, top, triangle);
    let gap = on_axis - on_triangle;
    let gap_squared = gap.length_squared();
    if gap_squared >= radius * radius {
        return None;
    }
    if gap_squared > TINY {
        let distance = gap_squared.sqrt();
        return Some((radius - distance, gap / distance));
    }
    let normal = triangle.normal;
    let low = normal.dot(bottom - triangle.a);
    let high = normal.dot(top - triangle.a);
    let up = radius - least(low, high);
    let down = radius + most(low, high);
    Some(if up <= down {
        (up, normal)
    } else {
        (down, -normal)
    })
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Boxed {
    pub centre: Vec3,
    pub half: Vec3,
    pub axes: [Vec3; 3],
}

impl Boxed {
    fn local(&self, p: Vec3) -> Vec3 {
        let offset = p - self.centre;
        Vec3::new(
            self.axes[0].dot(offset),
            self.axes[1].dot(offset),
            self.axes[2].dot(offset),
        )
    }

    fn signed_distance(&self, p: Vec3) -> f32 {
        let q = self.local(p).abs() - self.half;
        q.max(Vec3::ZERO).length() + least(q.max_element(), 0.0)
    }

    fn outward(&self, p: Vec3) -> Vec3 {
        let local = self.local(p);
        let q = local.abs() - self.half;
        let sign = Vec3::new(
            if local.x < 0.0 { -1.0 } else { 1.0 },
            if local.y < 0.0 { -1.0 } else { 1.0 },
            if local.z < 0.0 { -1.0 } else { 1.0 },
        );
        let outside = q.max(Vec3::ZERO);
        let direction = if outside.length_squared() > TINY {
            outside * sign
        } else if q.x >= q.y && q.x >= q.z {
            Vec3::new(sign.x, 0.0, 0.0)
        } else if q.y >= q.z {
            Vec3::new(0.0, sign.y, 0.0)
        } else {
            Vec3::new(0.0, 0.0, sign.z)
        };
        (self.axes[0] * direction.x + self.axes[1] * direction.y + self.axes[2] * direction.z)
            .normalize_or_zero()
    }

    pub fn penetration(&self, bottom: Vec3, top: Vec3, radius: f32) -> Option<(f32, Vec3)> {
        let along = top - bottom;
        let distance_at = |s: f32| self.signed_distance(bottom + along * s);
        let mut low = 0.0f32;
        let mut high = 1.0f32;
        for _ in 0..BOX_SEARCH_STEPS {
            let first = low + (high - low) * GOLDEN_LOW;
            let second = low + (high - low) * GOLDEN_HIGH;
            if distance_at(first) <= distance_at(second) {
                high = second;
            } else {
                low = first;
            }
        }
        let mut best = (low + high) * 0.5;
        for end in [0.0, 1.0] {
            if distance_at(end) < distance_at(best) {
                best = end;
            }
        }
        let point = bottom + along * best;
        let depth = radius - self.signed_distance(point);
        (depth > 0.0).then(|| (depth, self.outward(point)))
    }
}

const BOX_SEARCH_STEPS: usize = 40;
const GOLDEN_LOW: f32 = 0.381_966;
const GOLDEN_HIGH: f32 = 0.618_034;
