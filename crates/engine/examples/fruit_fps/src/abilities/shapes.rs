use glam::{Mat3, Vec3};

pub(super) fn cos_degrees(degrees: f32) -> f32 {
    let wrapped = (degrees as f64).rem_euclid(360.0);
    let folded = if wrapped > 180.0 {
        360.0 - wrapped
    } else {
        wrapped
    };
    let radians = folded * std::f64::consts::PI / 180.0;
    let square = radians * radians;
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    for k in 1..=14u32 {
        let n = (2 * k) as f64;
        term *= -square / ((n - 1.0) * n);
        sum += term;
    }
    sum as f32
}

pub(super) fn lesser(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}

pub(super) fn greater(a: f32, b: f32) -> f32 {
    if b > a { b } else { a }
}

pub(super) fn unit(x: f32) -> f32 {
    greater(0.0, lesser(x, 1.0))
}

pub(super) fn flat(v: Vec3) -> Option<Vec3> {
    let ground = Vec3::new(v.x, 0.0, v.z);
    let length = ground.length();
    if length > 0.0 {
        Some(ground / length)
    } else {
        None
    }
}

pub(super) fn frame(forward: Vec3) -> Mat3 {
    Mat3::from_cols(Vec3::Y.cross(forward), Vec3::Y, forward)
}

pub(super) fn closest_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let length_squared = ab.dot(ab);
    if length_squared <= 0.0 {
        return a;
    }
    let t = unit((p - a).dot(ab) / length_squared);
    a + ab * t
}

pub(super) fn segment_distance(p0: Vec3, p1: Vec3, q0: Vec3, q1: Vec3) -> f32 {
    let d1 = p1 - p0;
    let d2 = q1 - q0;
    let r = p0 - q0;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);
    let (s, t) = if a <= 0.0 && e <= 0.0 {
        (0.0, 0.0)
    } else if a <= 0.0 {
        (0.0, unit(f / e))
    } else {
        let c = d1.dot(r);
        if e <= 0.0 {
            (unit(-c / a), 0.0)
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            let mut s = if denom > 0.0 {
                unit((b * f - c * e) / denom)
            } else {
                0.0
            };
            let mut t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = unit(-c / a);
            } else if t > 1.0 {
                t = 1.0;
                s = unit((b - c) / a);
            }
            (s, t)
        }
    };
    ((p0 + d1 * s) - (q0 + d2 * t)).length()
}

pub(super) fn ray_sphere(
    origin: Vec3,
    direction: Vec3,
    max: f32,
    centre: Vec3,
    radius: f32,
) -> Option<f32> {
    let m = origin - centre;
    let c = m.dot(m) - radius * radius;
    if c <= 0.0 {
        return Some(0.0);
    }
    let b = m.dot(direction);
    if b > 0.0 {
        return None;
    }
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return None;
    }
    let t = -b - discriminant.sqrt();
    if t <= max {
        Some(if t < 0.0 { 0.0 } else { t })
    } else {
        None
    }
}

pub(super) fn ray_box(
    origin: Vec3,
    direction: Vec3,
    max: f32,
    centre: Vec3,
    half_extents: Vec3,
    axes: Mat3,
) -> Option<f32> {
    let to_local = axes.transpose();
    let o = to_local * (origin - centre);
    let d = to_local * direction;
    let mut near = 0.0f32;
    let mut far = max;
    for axis in 0..3 {
        let (oi, di, hi) = (o[axis], d[axis], half_extents[axis]);
        if di == 0.0 {
            if oi < -hi || oi > hi {
                return None;
            }
            continue;
        }
        let t1 = (-hi - oi) / di;
        let t2 = (hi - oi) / di;
        let (lo, hi_t) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        if lo > near {
            near = lo;
        }
        if hi_t < far {
            far = hi_t;
        }
        if near > far {
            return None;
        }
    }
    Some(near)
}

pub(super) fn segment_hits_sphere(a: Vec3, b: Vec3, centre: Vec3, radius: f32) -> bool {
    (centre - closest_on_segment(centre, a, b)).length() <= radius
}

pub(super) fn segment_hits_box(
    a: Vec3,
    b: Vec3,
    centre: Vec3,
    half_extents: Vec3,
    axes: Mat3,
) -> bool {
    let span = b - a;
    let length = span.length();
    if length > 0.0 {
        ray_box(a, span / length, length, centre, half_extents, axes).is_some()
    } else {
        ray_box(a, Vec3::X, 0.0, centre, half_extents, axes).is_some()
    }
}
