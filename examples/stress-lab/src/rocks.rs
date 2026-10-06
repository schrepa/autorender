//! Boulders for the highlands: icospheres displaced by layered noise,
//! squashed and stretched by their seed, split along a few planes into the
//! flat faces weathered stone breaks into, and flattened where they sit in
//! the ground, with texture coordinates projected along the axis each
//! vertex's normal turns most toward, so the terrain's rock maps lie on
//! them. Cliffs are outcrops of the same making, many meters across once
//! placed: their rock fluted by ridges running down it and broken into
//! ledges along their strata.

use std::collections::HashMap;

use engine::prelude::*;

/// Meters one repeat of the rock maps covers on a boulder.
const TILING: f32 = 3.0;

/// Planes each boulder is split along.
const CUTS: u32 = 9;

/// Boulder `seed`'s mesh: about a meter across before its stretch, its
/// base flattened just below its origin.
pub(crate) fn boulder(seed: u32) -> Result<Mesh, String> {
    let (mut positions, triangles) = icosphere(3);
    let random = |index: u32| unit(hash(seed.wrapping_mul(0x9e37_79b9) ^ index));
    let stretch = Vec3::new(
        0.9 + 0.6 * random(1),
        0.55 + 0.35 * random(2),
        0.8 + 0.5 * random(3),
    );
    let offset = Vec3::new(random(4), random(5), random(6)) * 37.0;
    // Each cut: a direction, and how far along it the face lies, as a
    // share of the stretched sphere's reach that way.
    let cuts: Vec<(Vec3, f32)> = (0..CUTS)
        .map(|cut| {
            let at = 10 + cut * 4;
            let direction = Vec3::new(
                random(at) * 2.0 - 1.0,
                random(at + 1) * 1.6 - 0.4,
                random(at + 2) * 2.0 - 1.0,
            )
            .normalize_or(Vec3::Y);
            let reach = (direction * stretch * 0.5).length();
            (direction, reach * (0.72 + 0.2 * random(at + 3)))
        })
        .collect();
    for position in &mut positions {
        let direction = *position;
        let broad = noise(direction * 1.7 + offset);
        let fine = noise(direction * 4.3 + offset * 1.3);
        let ridges = 1.0 - (noise(direction * 2.6 - offset) * 2.0 - 1.0).abs();
        let radius = 1.0 + 0.32 * (broad - 0.5) + 0.12 * (fine - 0.5) + 0.1 * ridges;
        let mut point = direction * radius * stretch * 0.5;
        for &(normal, distance) in &cuts {
            point -= normal * math::max(point.dot(normal) - distance, 0.0);
        }
        // Settled in the ground: the bottom flattened.
        point.y = math::max(point.y, -0.18 * stretch.y);
        *position = point;
    }
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for triangle in &triangles {
        let [a, b, c] = triangle.map(|index| positions[index as usize]);
        let face = (b - a).cross(c - a);
        for &index in triangle {
            normals[index as usize] += face;
        }
    }
    let vertices = positions
        .iter()
        .zip(&normals)
        .map(|(&position, &normal)| {
            let normal = normal.normalize_or(Vec3::Y);
            let axis = normal.abs();
            let uv = if axis.y >= axis.x && axis.y >= axis.z {
                [position.x, position.z]
            } else if axis.x >= axis.z {
                [position.z, -position.y]
            } else {
                [position.x, -position.y]
            };
            MeshVertex {
                position: position.to_array(),
                normal: normal.to_array(),
                uv: uv.map(|value| value / TILING),
            }
        })
        .collect();
    Mesh::new(vertices, triangles.into_iter().flatten().collect())
        .map_err(|error| format!("boulder {seed}: {error}"))
}

/// Planes each cliff is split along, and the most each one's details may
/// stray from its finest at scale 1 (meters), coarsest last.
const CLIFF_CUTS: u32 = 5;
pub(crate) const CLIFF_ERRORS: [f32; 2] = [0.012, 0.035];

/// Units of a cliff at scale 1 one repeat of the rock maps covers: a cliff
/// about twenty meters across repeats them every four.
const CLIFF_TILING: f32 = 0.2;

/// Cliff `seed`'s mesh at `detail` (0 the finest, up to 2): a block of
/// stratified rock about a meter across before its stretch, its sides
/// steep and nearly flat, as a superellipsoid's are, with rounded edges;
/// its beds stepping back in ledges where each tilted stratum ends, its
/// faces fluted by ridges running down them and split along a few upright
/// planes, and its bottom flattened below its origin, where it sinks into
/// the ground.
pub(crate) fn cliff(seed: u32, detail: u32) -> Result<Mesh, String> {
    let (mut positions, triangles) = icosphere(4 - detail.min(2));
    let random = |index: u32| unit(hash(seed.wrapping_mul(0x632b_e5ab) ^ index));
    // Wide across the slope it faces down (x), shallower into it (z).
    let stretch = Vec3::new(
        1.7 + 0.7 * random(1),
        0.8 + 0.5 * random(2),
        0.9 + 0.4 * random(3),
    );
    let offset = Vec3::new(random(4), random(5), random(6)) * 53.0;
    // How boxy: the superellipsoid's exponent.
    let boxy = 4.0 + 2.0 * random(7);
    // Strata, a few to a unit of height, tilted a little.
    let layers = 5.0 + 3.0 * random(8);
    let dip = Vec3::new(random(9) - 0.5, 3.0, random(10) - 0.5).normalize();
    let cuts: Vec<(Vec3, f32)> = (0..CLIFF_CUTS)
        .map(|cut| {
            let at = 20 + cut * 4;
            let direction = Vec3::new(
                random(at) * 2.0 - 1.0,
                random(at + 1) * 0.5 - 0.1,
                random(at + 2) * 2.0 - 1.0,
            )
            .normalize_or(Vec3::X);
            let reach = (direction * stretch * 0.5).length();
            (direction, reach * (0.78 + 0.15 * random(at + 3)))
        })
        .collect();
    for position in &mut positions {
        let direction = *position;
        let a = direction.abs();
        let block = (a.x.powf(boxy) + a.y.powf(boxy) + a.z.powf(boxy)).powf(-1.0 / boxy);
        let broad = noise(direction * 1.4 + offset);
        // Ridges running down the faces: noise drawn out along the height.
        let flutes =
            noise(Vec3::new(direction.x * 7.0, direction.y * 1.2, direction.z * 7.0) + offset);
        let fluted = 1.0 - (flutes * 2.0 - 1.0).abs();
        let radius = block * (1.0 + 0.16 * (broad - 0.5) + 0.07 * fluted);
        let mut point = direction * radius * stretch * 0.5;
        // Each stratum's top steps back into the rock as a ledge.
        let height = point.dot(dip) * layers + 0.35 * noise(direction * 2.5 - offset);
        let within = height - height.floor();
        let ledge = math::clamp((within - 0.6) / 0.25, 0.0, 1.0);
        let ledge = ledge * ledge * (3.0 - 2.0 * ledge);
        point -= Vec3::new(point.x, 0.0, point.z) * (0.09 * ledge);
        for &(normal, distance) in &cuts {
            point -= normal * math::max(point.dot(normal) - distance, 0.0);
        }
        // Sunk in the ground: the bottom flattened.
        point.y = math::max(point.y, -0.3 * stretch.y);
        *position = point;
    }
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for triangle in &triangles {
        let [a, b, c] = triangle.map(|index| positions[index as usize]);
        let face = (b - a).cross(c - a);
        for &index in triangle {
            normals[index as usize] += face;
        }
    }
    let vertices = positions
        .iter()
        .zip(&normals)
        .map(|(&position, &normal)| {
            let normal = normal.normalize_or(Vec3::Y);
            let axis = normal.abs();
            let uv = if axis.y >= axis.x && axis.y >= axis.z {
                [position.x, position.z]
            } else if axis.x >= axis.z {
                [position.z, -position.y]
            } else {
                [position.x, -position.y]
            };
            MeshVertex {
                position: position.to_array(),
                normal: normal.to_array(),
                uv: uv.map(|value| value / CLIFF_TILING),
            }
        })
        .collect();
    Mesh::new(vertices, triangles.into_iter().flatten().collect())
        .map_err(|error| format!("cliff {seed}: {error}"))
}

/// A unit icosphere subdivided `levels` times: its vertices and
/// counter-clockwise triangles seen from outside.
fn icosphere(levels: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let mut positions: Vec<Vec3> = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ]
    .iter()
    .map(|&point| Vec3::from_array(point).normalize())
    .collect();
    let mut triangles: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..levels {
        let mut middles: HashMap<(u32, u32), u32> = HashMap::new();
        let mut middle = |a: u32, b: u32, positions: &mut Vec<Vec3>| {
            *middles.entry((a.min(b), a.max(b))).or_insert_with(|| {
                positions.push(((positions[a as usize] + positions[b as usize]) * 0.5).normalize());
                positions.len() as u32 - 1
            })
        };
        let mut finer = Vec::with_capacity(triangles.len() * 4);
        for [a, b, c] in triangles {
            let ab = middle(a, b, &mut positions);
            let bc = middle(b, c, &mut positions);
            let ca = middle(c, a, &mut positions);
            finer.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        triangles = finer;
    }
    (positions, triangles)
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
