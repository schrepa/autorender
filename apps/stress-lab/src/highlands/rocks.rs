//! Boulders for the highlands: icospheres displaced by layered noise,
//! squashed and stretched by their seed, split along a few planes into the
//! faces weathered stone breaks into, and flattened where they sit in the
//! ground. The planes cut softly, so each face rounds over into the
//! weathered surface across a few vertices and its edge never shows the
//! steps of the triangles it crosses; the faces undulate a little, as
//! fractures do. Texture coordinates project along the axis
//! each vertex's normal turns most toward; the highlands lay their rock
//! maps from the world's axes instead. Each shape comes at a few details,
//! the coarser with the most they stray from the finest. Cliffs are
//! outcrops of the same making, many meters across once placed: their rock
//! fluted by ridges running down it and broken into ledges along their
//! strata, whose edges keep their crease. Every vertex's color bakes in
//! how much the shape around it hides it, darker in hollows and under
//! ledges; the ground stains their bases where terrains place them.

use engine::collections::HashMap;
use std::f32::consts::PI;

use engine::prelude::*;

use crate::kit::random::{golden_hash, noise, unit};

/// Icosphere subdivisions of the finest detail of boulders and cliffs.
const LEVELS: u32 = 4;

/// Details coarser than the finest each shape comes at.
pub(crate) const COARSER: u32 = 2;

/// Meters one repeat of the rock maps covers on a boulder.
const TILING: f32 = 3.0;

/// Planes each boulder is split along.
const CUTS: u32 = 9;

/// How far a cut rounds over into the surface around it, in units of a
/// shape at scale 1: a few vertices of the finest detail.
const ROUNDING: f32 = 0.08;

/// The sharpest turn between faces a cliff's normals smooth across
/// (radians), about 50 degrees: its ledges keep their edges.
const LEDGE_CREASE: f32 = 0.87;

/// Boulder `seed`'s mesh at `detail` (0 the finest, up to [`COARSER`]) and
/// the most it strays from the finest: about a meter across before its
/// stretch, its base flattened just below its origin.
pub(crate) fn boulder(seed: u32, detail: u32) -> Result<(Mesh, f32), String> {
    let random = |index: u32| unit(golden_hash(seed.wrapping_mul(0x9e37_79b9) ^ index));
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
    let floor = -0.18 * stretch.y;
    let shape = |direction: Vec3| {
        let broad = noise(direction * 1.7 + offset);
        let fine = noise(direction * 4.3 + offset * 1.3);
        let ridges = 1.0 - (noise(direction * 2.6 - offset) * 2.0 - 1.0).abs();
        let radius = 1.0 + 0.32 * (broad - 0.5) + 0.12 * (fine - 0.5) + 0.1 * ridges;
        let mut point = direction * radius * stretch * 0.5;
        let rough = 0.025 * (noise(direction * 6.0 + offset * 0.7) - 0.5);
        for &(normal, distance) in &cuts {
            point -= normal * beyond(point.dot(normal) - distance - rough, ROUNDING);
        }
        // Settled in the ground: the bottom flattened.
        point.y += beyond(floor - point.y, ROUNDING);
        point
    };
    let (vertices, indices, colors, error) = shaped(&shape, detail, PI, TILING);
    Mesh::new(vertices, indices)
        .and_then(|mesh| mesh.with_colors(colors))
        .map(|mesh| (mesh, error))
        .map_err(|error| format!("boulder {seed}: {error}"))
}

/// Planes each cliff is split along.
const CLIFF_CUTS: u32 = 5;

/// Units of a cliff at scale 1 one repeat of the rock maps covers: a cliff
/// about twenty meters across repeats them every four.
const CLIFF_TILING: f32 = 0.2;

/// Cliff `seed`'s mesh at `detail` (0 the finest, up to [`COARSER`]) and
/// the most it strays from the finest: a block of stratified rock about a
/// meter across before its stretch, its sides steep and nearly flat, as a
/// superellipsoid's are, with rounded edges; its beds stepping back in
/// ledges where each tilted stratum ends, its faces fluted by ridges
/// running down them and split along a few upright planes, and its bottom
/// flattened below its origin, where it sinks into the ground.
pub(crate) fn cliff(seed: u32, detail: u32) -> Result<(Mesh, f32), String> {
    let random = |index: u32| unit(golden_hash(seed.wrapping_mul(0x632b_e5ab) ^ index));
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
    let floor = -0.3 * stretch.y;
    let shape = |direction: Vec3| {
        let a = direction.abs();
        let block = (a.x.powf(boxy) + a.y.powf(boxy) + a.z.powf(boxy)).powf(-1.0 / boxy);
        let broad = noise(direction * 1.4 + offset);
        // Ridges running down the faces: noise drawn out along the height.
        let flutes =
            noise(Vec3::new(direction.x * 7.0, direction.y * 1.2, direction.z * 7.0) + offset);
        let fluted = 1.0 - (flutes * 2.0 - 1.0).abs();
        let radius = block * (1.0 + 0.16 * (broad - 0.5) + 0.07 * fluted);
        let mut point = direction * radius * stretch * 0.5;
        // Each stratum's top steps back into the rock as a ledge, the
        // strata thickening and thinning across the block and some cut
        // deeper than others.
        let height = point.dot(dip) * layers
            + 0.35 * noise(direction * 2.5 - offset)
            + 0.8 * noise(direction * 0.8 + offset * 0.4);
        let stratum = height.floor();
        let within = height - stratum;
        let depth = 0.03 + 0.1 * random(1000u32.wrapping_add(stratum as i32 as u32));
        let ledge = math::clamp((within - 0.6) / 0.25, 0.0, 1.0);
        let ledge = ledge * ledge * (3.0 - 2.0 * ledge);
        point -= Vec3::new(point.x, 0.0, point.z) * (depth * ledge);
        let rough = 0.02 * (noise(direction * 5.0 - offset * 0.6) - 0.5);
        for &(normal, distance) in &cuts {
            point -= normal * beyond(point.dot(normal) - distance - rough, 0.5 * ROUNDING);
        }
        // Sunk in the ground: the bottom flattened.
        point.y += beyond(floor - point.y, 0.5 * ROUNDING);
        point
    };
    let (vertices, indices, colors, error) = shaped(&shape, detail, LEDGE_CREASE, CLIFF_TILING);
    Mesh::new(vertices, indices)
        .and_then(|mesh| mesh.with_colors(colors))
        .map(|mesh| (mesh, error))
        .map_err(|error| format!("cliff {seed}: {error}"))
}

/// How far `excess` reaches past 0, rounded over `width` around it: none
/// well below, all of it well above, a quarter of `width` at 0 (a
/// polynomial smooth maximum).
fn beyond(excess: f32, width: f32) -> f32 {
    let h = math::max(width - excess.abs(), 0.0) / width;
    math::max(excess, 0.0) + 0.25 * width * h * h
}

/// The vertices, indices and colors ([`baked`]) of `shape` (each unit
/// direction's point) over an icosphere subdivided [`LEVELS`] less
/// `detail` times, normals creased past `crease` and texture coordinates
/// repeating every `tiling` units; and the most it strays from the finest
/// detail: the distance from each edge's middle to the finer shape's point
/// there, the largest of each detail between summed.
fn shaped(
    shape: &impl Fn(Vec3) -> Vec3,
    detail: u32,
    crease: f32,
    tiling: f32,
) -> (Vec<MeshVertex>, Vec<u32>, Vec<[f32; 4]>, f32) {
    let level = LEVELS - detail.min(COARSER);
    let error = (level..LEVELS)
        .map(|coarser| {
            let (directions, triangles) = icosphere(coarser);
            let points: Vec<Vec3> = directions
                .iter()
                .map(|&direction| shape(direction))
                .collect();
            triangles
                .iter()
                .flat_map(|&[a, b, c]| [(a, b), (b, c), (c, a)])
                .map(|(a, b)| {
                    let middle = (directions[a as usize] + directions[b as usize]).normalize();
                    let edge = (points[a as usize] + points[b as usize]) * 0.5;
                    shape(middle).distance(edge)
                })
                .fold(0.0, math::max)
        })
        .sum();
    let (directions, triangles) = icosphere(level);
    let points: Vec<Vec3> = directions.into_iter().map(shape).collect();
    let baked = baked(&points, &triangles);
    let (positions, normals, indices, sources) = creased(&points, &triangles, crease);
    let colors = sources
        .iter()
        .map(|&source| baked[source as usize])
        .collect();
    let vertices = positions
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
                uv: uv.map(|value| value / tiling),
            }
        })
        .collect();
    (vertices, indices, colors, error)
}

/// Each of `points`' colors over `triangles`: how little the shape around
/// it hides it, from how far its neighbors stand out along its normal
/// against the edges' length, so hollows and the undersides of ledges
/// darken.
fn baked(points: &[Vec3], triangles: &[[u32; 3]]) -> Vec<[f32; 4]> {
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
    points
        .iter()
        .zip(normals.iter().zip(&sums))
        .map(|(&point, (&normal, &(sum, length, count)))| {
            let count = count.max(1) as f32;
            let normal = normal.normalize_or(Vec3::Y);
            let around = sum / count - point;
            let edge = math::max(length / count, 1.0e-4);
            let lit = 1.0 - math::clamp(around.dot(normal) / edge * 2.5, 0.0, 0.45);
            [lit, lit, lit, 1.0]
        })
        .collect()
}

/// The vertices of `triangles` over `positions`, each corner's normal the
/// area-weighted mean of the faces around it turning less than `crease`
/// (radians) from its own, so sharper edges stay crisp; corners alike share
/// a vertex. Returns positions, normals, indices and each vertex's index in
/// `positions`.
fn creased(
    positions: &[Vec3],
    triangles: &[[u32; 3]],
    crease: f32,
) -> (Vec<Vec3>, Vec<Vec3>, Vec<u32>, Vec<u32>) {
    let faces: Vec<Vec3> = triangles
        .iter()
        .map(|triangle| {
            let [a, b, c] = triangle.map(|index| positions[index as usize]);
            (b - a).cross(c - a)
        })
        .collect();
    let mut around: Vec<Vec<u32>> = vec![Vec::new(); positions.len()];
    for (face, triangle) in triangles.iter().enumerate() {
        for &index in triangle {
            around[index as usize].push(face as u32);
        }
    }
    let limit = math::cos(crease);
    let mut shared: HashMap<(u32, [i32; 3]), u32> = HashMap::default();
    let (mut out_positions, mut out_normals, mut sources) = (Vec::new(), Vec::new(), Vec::new());
    let mut indices = Vec::with_capacity(triangles.len() * 3);
    for (face, triangle) in triangles.iter().enumerate() {
        let own = faces[face].normalize_or(Vec3::Y);
        for &index in triangle {
            let sum: Vec3 = around[index as usize]
                .iter()
                .map(|&other| faces[other as usize])
                .filter(|other| other.normalize_or(Vec3::Y).dot(own) >= limit)
                .sum();
            let normal = sum.normalize_or(own);
            let key = (
                index,
                normal
                    .to_array()
                    .map(|value| (value * 1024.0).round() as i32),
            );
            let vertex = *shared.entry(key).or_insert_with(|| {
                out_positions.push(positions[index as usize]);
                out_normals.push(normal);
                sources.push(index);
                (out_positions.len() - 1) as u32
            });
            indices.push(vertex);
        }
    }
    (out_positions, out_normals, indices, sources)
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
