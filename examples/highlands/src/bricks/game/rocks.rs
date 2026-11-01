use engine_core::collections::HashMap;
use engine_core::{Error, Mesh, MeshVertex, math};
use glam::Vec3;

use crate::bricks::game::random::{golden_hash, noise, unit};

const LEVELS: u32 = 4;

const TILING: f32 = 3.0;

pub(crate) fn boulder(seed: u32) -> Result<Mesh, Error> {
    let random = |index: u32| unit(golden_hash(seed.wrapping_mul(0x9e37_79b9) ^ index));
    let stretch = Vec3::new(
        0.9 + 0.6 * random(1),
        0.55 + 0.35 * random(2),
        0.8 + 0.5 * random(3),
    );
    let offset = Vec3::new(random(4), random(5), random(6)) * 37.0;
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
    let (vertices, indices, colors) = shaped(&shape);
    Mesh::new(vertices, indices)
        .and_then(|mesh| mesh.with_colors(colors))
        .map_err(|error| error.within(&format!("$.boulders[{seed}]")))
}

fn shaped(shape: &impl Fn(Vec3) -> Vec3) -> (Vec<MeshVertex>, Vec<u32>, Vec<[f32; 4]>) {
    let (directions, triangles) = icosphere(LEVELS);
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
