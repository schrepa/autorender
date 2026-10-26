//! Boulders for the highlands: icospheres displaced by smooth,
//! low-frequency noise into rounded, weathered stones, squashed and
//! stretched by their seed and settled flatter underneath, their normals
//! smooth throughout; and talus blocks, the same lumpy shapes less worn and
//! cut flat along the joints rock breaks on as it falls from a face, before
//! weather rounds it. Texture coordinates project along the axis each
//! vertex's normal turns most toward; the highlands lay their rock maps
//! from the world's axes instead. Each shape is built at one detail; its
//! coarser levels are its mesh's own, simplified offline. Every vertex's
//! color bakes in how much the shape around it hides it, darker in
//! hollows; where terrains place them, their bases blend into the ground.

use engine::collections::HashMap;

use engine::prelude::*;

use crate::kit::random::{golden_hash, noise, unit};
use crate::kit::records;

/// Boulder `seed`'s mesh as a record ([`records::mesh`]): its handle,
/// reserved until the record is here.
pub(crate) fn boulder_record(world: &mut World, seed: u32) -> engine::Result<MeshHandle> {
    records::mesh(world, "highlands_boulder", &seed.to_le_bytes(), move || {
        boulder(seed)
    })
}

/// Talus block `seed`'s mesh as a record ([`records::mesh`]): its handle,
/// reserved until the record is here.
pub(crate) fn block_record(world: &mut World, seed: u32) -> engine::Result<MeshHandle> {
    records::mesh(world, "highlands_block", &seed.to_le_bytes(), move || {
        block(seed)
    })
}

/// Icosphere subdivisions of boulders, and of blocks, whose flat faces
/// need fewer.
const LEVELS: u32 = 4;
const BLOCK_LEVELS: u32 = 3;

/// Meters one repeat of the rock maps covers on a boulder.
const TILING: f32 = 3.0;

/// How a talus block breaks: the share of its reach from its middle along
/// a joint's normal at which the joint cuts it, the nearest and the
/// farthest; the most its bedding tilts from level and its two steep joint
/// sets from upright, and turn from square to each other (radians); and the
/// meters below its origin its flat underside rests, settled into the
/// scree. Jointed rock breaks along its bedding and two steep joint sets,
/// each cutting both sides of a block.
const CUT_DEPTH: [f32; 2] = [0.62, 0.85];
const CUT_TILT: [f32; 3] = [0.35, 0.25, 0.3];
const BLOCK_BASE: f32 = 0.15;

/// Boulder `seed`'s mesh: about a meter across before its stretch, lumpy
/// with two octaves of smooth noise, its underside settling flatter than
/// its top from its middle down, below its origin, where it rests in the
/// ground.
fn boulder(seed: u32) -> engine::Result<Mesh> {
    let random = |index: u32| unit(golden_hash(seed.wrapping_mul(0x9e37_79b9) ^ index));
    let stretch = Vec3::new(
        0.9 + 0.6 * random(1),
        0.55 + 0.35 * random(2),
        0.8 + 0.5 * random(3),
    );
    let offset = Vec3::new(random(4), random(5), random(6)) * 37.0;
    // Below its middle the underside flattens by up to half over a quarter
    // of its height, easing in, so no edge rings it.
    let settling = 0.25 * stretch.y;
    let shape = |direction: Vec3| {
        let broad = noise(direction * 1.5 + offset);
        let lumps = noise(direction * 3.2 - offset * 0.7);
        let radius = 1.0 + 0.4 * (broad - 0.5) + 0.14 * (lumps - 0.5);
        let mut point = direction * radius * stretch * 0.5;
        let settled = math::clamp(-point.y / settling, 0.0, 1.0);
        point.y *= 1.0 - 0.5 * settled * settled * (3.0 - 2.0 * settled);
        point
    };
    let (vertices, indices, colors) = shaped(&shape, LEVELS);
    Mesh::new(vertices, indices)
        .and_then(|mesh| mesh.with_colors(colors))
        .map_err(|error| engine::Error::from(error).context(format!("boulder {seed}")))
}

/// Talus block `seed`'s mesh: about a meter across before its stretch,
/// lumpy with a little smooth noise, cut flat over its top by its bedding
/// and on its sides by its two steep joint sets ([`CUT_DEPTH`],
/// [`CUT_TILT`]), and resting on its flat underside [`BLOCK_BASE`] below
/// its origin.
fn block(seed: u32) -> engine::Result<Mesh> {
    let random = |index: u32| unit(golden_hash(seed.wrapping_mul(0x632b_e5ab) ^ index));
    let stretch = Vec3::new(
        0.9 + 0.5 * random(1),
        0.8 + 0.35 * random(2),
        0.8 + 0.45 * random(3),
    );
    let offset = Vec3::new(random(4), random(5), random(6)) * 41.0;
    let tilt = |index: u32, most: f32| most * (2.0 * random(index) - 1.0);
    // The joints' normals: the bedding over the top, and each steep set
    // both ways, the second set about square to the first.
    let turn = random(7) * std::f32::consts::TAU;
    let steep = |set: u32| {
        let (sin, cos) =
            math::sin_cos(turn + set as f32 * (std::f32::consts::FRAC_PI_2 + tilt(8, CUT_TILT[2])));
        Vec3::new(cos, tilt(9 + set, CUT_TILT[1]), sin).normalize_or(Vec3::X)
    };
    let joints = [
        Vec3::new(tilt(11, CUT_TILT[0]), 1.0, tilt(12, CUT_TILT[0])).normalize_or(Vec3::Y),
        steep(0),
        -steep(0),
        steep(1),
        -steep(1),
    ];
    // Each joint cuts at its share of the block's reach along its normal,
    // the stretched sphere's.
    let cuts: [(Vec3, f32); 5] = std::array::from_fn(|index| {
        let share = CUT_DEPTH[0] + (CUT_DEPTH[1] - CUT_DEPTH[0]) * random(20 + index as u32);
        (
            joints[index],
            share * 0.5 * (stretch * joints[index]).length(),
        )
    });
    let shape = |direction: Vec3| {
        let lumps = noise(direction * 2.4 + offset);
        let mut point = direction * (1.0 + 0.12 * (lumps - 0.5)) * stretch * 0.5;
        for &(normal, depth) in &cuts {
            point -= normal * math::max(point.dot(normal) - depth, 0.0);
        }
        point.y = math::max(point.y, -BLOCK_BASE);
        point
    };
    let (vertices, indices, colors) = shaped(&shape, BLOCK_LEVELS);
    Mesh::new(vertices, indices)
        .and_then(|mesh| mesh.with_colors(colors))
        .map_err(|error| engine::Error::from(error).context(format!("talus block {seed}")))
}

/// The vertices, indices and colors of `shape` (each unit direction's
/// point) over an icosphere subdivided `levels` times: each vertex's
/// normal and color ([`baked`]), and its texture coordinates along the axis
/// its normal turns most toward, repeating every [`TILING`] units.
fn shaped(
    shape: &impl Fn(Vec3) -> Vec3,
    levels: u32,
) -> (Vec<MeshVertex>, Vec<u32>, Vec<[f32; 4]>) {
    let (directions, triangles) = icosphere(levels);
    let points: Vec<Vec3> = directions.into_iter().map(shape).collect();
    let (normals, colors) = baked(&points, &triangles);
    let vertices = points
        .iter()
        .zip(&normals)
        .map(|(&position, &normal)| {
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
    let indices = triangles.into_iter().flatten().collect();
    (vertices, indices, colors)
}

/// Each of `points`' normals and colors over `triangles`: its normal the
/// area-weighted mean of the faces around it; its color how little the
/// shape around it hides it, from how far its neighbors stand out along
/// its normal against the edges' length, so hollows darken.
fn baked(points: &[Vec3], triangles: &[[u32; 3]]) -> (Vec<Vec3>, Vec<[f32; 4]>) {
    let mut normals = vec![Vec3::ZERO; points.len()];
    let mut sums = vec![(Vec3::ZERO, 0.0_f32, 0_u32); points.len()];
    for &[a, b, c] in triangles {
        let [pa, pb, pc] = [a, b, c].map(|index| points[index as usize]);
        let face = (pb - pa).cross(pc - pa);
        for (index, others) in [(a, [pb, pc]), (b, [pc, pa]), (c, [pa, pb])] {
            let index = index as usize;
            normals[index] += face;
            let entry = &mut sums[index];
            for other in others {
                entry.0 += other;
                entry.1 += other.distance(points[index]);
                entry.2 += 1;
            }
        }
    }
    let normals: Vec<Vec3> = normals
        .into_iter()
        .map(|normal| normal.normalize_or(Vec3::Y))
        .collect();
    let colors = points
        .iter()
        .zip(normals.iter().zip(&sums))
        .map(|(&point, (&normal, &(sum, length, count)))| {
            let count = count.max(1) as f32;
            let around = sum / count - point;
            let edge = math::max(length / count, 1.0e-4);
            let lit = 1.0 - math::clamp(around.dot(normal) / edge * 2.5, 0.0, 0.45);
            [lit, lit, lit, 1.0]
        })
        .collect();
    (normals, colors)
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
        let mut middles: HashMap<(u32, u32), u32> = HashMap::default();
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
