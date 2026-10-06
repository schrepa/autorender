//! Caves for the highlands (decision 079): a tube of rough rock around a
//! winding centerline, its floor flattened to walk on, its walls bulging
//! and receding by layered noise, flaring a little at its mouth, where it
//! meets the hole cut in the slope, and widening into a chamber at its end
//! before it closes. Its triangles face in; its material shows both sides,
//! so where its shell rises out of the slope at the mouth it reads as an
//! overhang.

use engine::prelude::*;

/// Meters between the rings along the centerline, and points around each.
const RING_STEP: f32 = 1.5;
const RING_POINTS: u32 = 28;

/// Meters of the cave one repeat of the rock maps covers.
const TILING: f32 = 4.0;

/// The cave through world `points`, `radius` meters across and up from
/// the centerline, roughened by `seed`'s noise, as a mesh around
/// `points[0]`.
pub(crate) fn cave(points: &[Vec3], radius: [f32; 2], seed: u32) -> Result<Mesh, String> {
    let (centers, tangents) = centerline(points);
    let rings = centers.len();
    if rings < 2 {
        return Err("a cave needs a centerline at least a ring long".into());
    }
    let origin = points[0];
    let offset = Vec3::new(
        (seed % 97) as f32 * 3.1,
        (seed % 89) as f32 * 1.7,
        (seed % 83) as f32 * 2.3,
    );
    let mut vertices = Vec::with_capacity(rings * (RING_POINTS as usize + 1));
    let mut along = 0.0;
    for (index, (&center, &tangent)) in centers.iter().zip(&tangents).enumerate() {
        if index > 0 {
            along += center.distance(centers[index - 1]);
        }
        let share = index as f32 / (rings - 1) as f32;
        // Flaring at the mouth, widening into the chamber, closing at the
        // end.
        let flare = 1.0 + 0.15 * (1.0 - smooth(share / 0.08));
        let chamber = 1.0 + 0.8 * smooth((share - 0.7) / 0.2);
        // A rounded end: the rings shrink as a sphere's do.
        let end = math::clamp((share - 0.9) / 0.1, 0.0, 1.0);
        let closing = (1.0 - end * end).sqrt();
        let size = flare * chamber * math::max(closing, 0.02);
        let up = (Vec3::Y - tangent * Vec3::Y.dot(tangent)).normalize_or(Vec3::Y);
        let across = tangent.cross(up);
        for point in 0..=RING_POINTS {
            let angle = point as f32 / RING_POINTS as f32 * std::f32::consts::TAU;
            let (sin, cos) = math::sin_cos(angle);
            let direction = across * cos + up * sin;
            let rough = |position: Vec3| {
                0.26 * (noise(position * 0.22 + offset) - 0.5)
                    + 0.1 * (noise(position * 0.9 - offset) - 0.5)
            };
            let base = center + (across * (cos * radius[0]) + up * (sin * radius[1])) * size;
            let mut position = center
                + (across * (cos * radius[0]) + up * (sin * radius[1]))
                    * size
                    * (1.0 + rough(base));
            // A floor to walk on.
            let floor = center.y - 0.55 * radius[1] * size;
            position.y = math::max(position.y, floor);
            let u = along / TILING;
            let v = angle * (radius[0] + radius[1]) * 0.5 / TILING;
            vertices.push((position - origin, -direction, [u, v]));
        }
    }
    let columns = RING_POINTS + 1;
    let mut indices = Vec::with_capacity((rings - 1) * RING_POINTS as usize * 6);
    for ring in 0..rings as u32 - 1 {
        for point in 0..RING_POINTS {
            let a = ring * columns + point;
            let b = a + 1;
            let c = a + columns;
            let d = c + 1;
            // Wound to face the centerline.
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    // Normals from the faces, facing in.
    let mut normals = vec![Vec3::ZERO; vertices.len()];
    for triangle in indices.as_chunks::<3>().0 {
        let [a, b, c] = triangle.map(|index| vertices[index as usize].0);
        let face = (b - a).cross(c - a);
        for &index in triangle {
            normals[index as usize] += face;
        }
    }
    let vertices = vertices
        .iter()
        .zip(&normals)
        .map(|(&(position, inward, uv), &normal)| {
            let normal = if normal.dot(inward) >= 0.0 {
                normal
            } else {
                -normal
            };
            MeshVertex {
                position: position.to_array(),
                normal: normal.normalize_or(inward).to_array(),
                uv,
            }
        })
        .collect();
    Mesh::new(vertices, indices).map_err(|error| format!("cave: {error}"))
}

/// Points every [`RING_STEP`] meters along a Catmull-Rom curve through
/// `points` (its ends continued straight), and the curve's directions
/// there.
fn centerline(points: &[Vec3]) -> (Vec<Vec3>, Vec<Vec3>) {
    let count = points.len();
    if count < 2 {
        return (points.to_vec(), vec![Vec3::X; count]);
    }
    let at = |index: isize| -> Vec3 {
        if index < 0 {
            2.0 * points[0] - points[1]
        } else if index as usize >= count {
            2.0 * points[count - 1] - points[count - 2]
        } else {
            points[index as usize]
        }
    };
    let mut centers = vec![points[0]];
    for segment in 0..count - 1 {
        let index = segment as isize;
        let [p0, p1, p2, p3] = [at(index - 1), at(index), at(index + 1), at(index + 2)];
        let steps = math::max((p1.distance(p2) / RING_STEP).ceil(), 1.0) as usize;
        for step in 1..=steps {
            let t = step as f32 / steps as f32;
            let (t2, t3) = (t * t, t * t * t);
            centers.push(
                0.5 * (2.0 * p1
                    + (p2 - p0) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3),
            );
        }
    }
    let last = centers.len() - 1;
    let tangents = (0..centers.len())
        .map(|index| {
            let (back, ahead) = (index.saturating_sub(1), (index + 1).min(last));
            (centers[ahead] - centers[back]).normalize_or(Vec3::X)
        })
        .collect();
    (centers, tangents)
}

/// Hermite smoothing of `value` clamped to [0, 1].
fn smooth(value: f32) -> f32 {
    let t = math::clamp(value, 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn hash(value: u32) -> u32 {
    let mut h = value.wrapping_mul(0x9e37_79b1);
    h = (h ^ (h >> 16)).wrapping_mul(0x7feb_352d);
    h = (h ^ (h >> 15)).wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

fn unit(value: u32) -> f32 {
    (value >> 8) as f32 / 16_777_216.0
}

/// Value noise from 0 to 1 at `point`, smooth between lattice points.
fn noise(point: Vec3) -> f32 {
    let cell = point.floor();
    let f = point - cell;
    let u = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let corner = |dx: i32, dy: i32, dz: i32| {
        let key = (cell.x as i32 + dx) as u32
            ^ ((cell.y as i32 + dy) as u32).wrapping_mul(0x8da6_b343)
            ^ ((cell.z as i32 + dz) as u32).wrapping_mul(0xd816_3841);
        unit(hash(key))
    };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), u.x);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), u.x);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), u.x);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), u.x);
    lerp(lerp(x00, x10, u.y), lerp(x01, x11, u.y), u.z)
}
