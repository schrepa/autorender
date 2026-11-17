use engine_core::collections::HashMap;
use engine_core::{Error, Mesh, MeshVertex, math};
use glam::{Quat, Vec2, Vec3};

use super::random::{golden_hash, noise, unit};

const SHAPE_SHIFT: u32 = 8;
const OUTCROP: u32 = 1;
const LEDGE: u32 = 2;
const BLOCK: u32 = 3;

const LEVELS: u32 = 3;

const TILING: f32 = 3.0;

const SMOOTH: f32 = 0.788;
const OVERHANG: f32 = 0.15;
const THINNEST: f32 = 0.16;

const CRAG: Face = Face {
    crest: [0.8, 1.3],
    bottom: -1.5,
    steep: 4.0,
    lead: [0.15, 0.5],
};
const BRINK: Face = Face {
    crest: [0.15, 0.45],
    bottom: -2.6,
    steep: 6.0,
    lead: [0.5, 1.0],
};

pub(crate) fn boulder(rock: u32) -> Result<Mesh, Error> {
    let seed = rock & ((1 << SHAPE_SHIFT) - 1);
    let shape = match rock >> SHAPE_SHIFT {
        OUTCROP => stack(seed, OUTCROP, &CRAG),
        LEDGE => stack(seed, LEDGE, &BRINK),
        BLOCK => block(seed),
        _ => cobble(seed),
    };
    let (vertices, indices, colors) = shape.finished();
    Mesh::new(vertices, indices)
        .and_then(|mesh| mesh.with_colors(colors))
        .map_err(|error| error.within(&format!("$.rocks[{rock}]")))
}

struct Dice {
    seed: u32,
    rolls: u32,
}

impl Dice {
    fn new(seed: u32, shape: u32) -> Self {
        Self {
            seed: (seed + 101 * shape).wrapping_mul(0x9e37_79b9),
            rolls: 0,
        }
    }

    fn roll(&mut self) -> f32 {
        self.rolls += 1;
        unit(golden_hash(self.seed ^ self.rolls))
    }

    fn between(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.roll()
    }

    fn count(&mut self, low: u32, high: u32) -> u32 {
        (low + (self.roll() * (high - low + 1) as f32) as u32).min(high)
    }

    fn direction(&mut self) -> Vec3 {
        let height = self.between(-1.0, 1.0);
        let (sin, cos) = math::sin_cos(self.between(0.0, std::f32::consts::TAU));
        let ring = math::max(1.0 - height * height, 0.0).sqrt();
        Vec3::new(ring * cos, height, ring * sin)
    }
}

#[derive(Default)]
struct Shape {
    points: Vec<Vec3>,
    normals: Vec<Vec3>,
    shade: Vec<f32>,
    triangles: Vec<[u32; 3]>,
}

impl Shape {
    fn vertex(&mut self, point: Vec3, normal: Vec3, shade: f32) -> u32 {
        self.points.push(point);
        self.normals.push(normal);
        self.shade.push(shade);
        self.points.len() as u32 - 1
    }

    fn face(&mut self, [a, b, c]: [u32; 3], outward: Vec3) {
        let [pa, pb, pc] = [a, b, c].map(|index| self.points[index as usize]);
        if (pb - pa).cross(pc - pa).dot(outward) >= 0.0 {
            self.triangles.push([a, b, c]);
        } else {
            self.triangles.push([a, c, b]);
        }
    }

    fn turned(mut self, turn: Quat) -> Self {
        for point in self.points.iter_mut().chain(&mut self.normals) {
            *point = turn * *point;
        }
        self
    }

    fn finished(self) -> (Vec<MeshVertex>, Vec<u32>, Vec<[f32; 4]>) {
        let vertices = self
            .points
            .iter()
            .zip(&self.normals)
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
        let colors = self
            .shade
            .iter()
            .map(|&shade| [shade, shade, shade, 1.0])
            .collect();
        let indices = self.triangles.into_iter().flatten().collect();
        (vertices, indices, colors)
    }
}

fn winding(outline: &[Vec2]) -> f32 {
    let mut doubled = 0.0;
    for (index, &point) in outline.iter().enumerate() {
        doubled += point.perp_dot(outline[(index + 1) % outline.len()]);
    }
    if doubled > 0.0 { 1.0 } else { -1.0 }
}

fn ears(outline: &[Vec2]) -> Vec<[usize; 3]> {
    let sign = winding(outline);
    let turn = |a: Vec2, b: Vec2, c: Vec2| sign * (b - a).perp_dot(c - a);
    let mut left: Vec<usize> = (0..outline.len()).collect();
    let mut triangles = Vec::with_capacity(outline.len().saturating_sub(2));
    while left.len() > 3 {
        let count = left.len();
        let corners = |at: usize| [left[(at + count - 1) % count], left[at], left[(at + 1) % count]];
        let ear = (0..count).find(|&at| {
            let [a, b, c] = corners(at).map(|index| outline[index]);
            turn(a, b, c) > 1.0e-9
                && left.iter().all(|other| {
                    let point = outline[*other];
                    corners(at).contains(other)
                        || turn(a, b, point) < 0.0
                        || turn(b, c, point) < 0.0
                        || turn(c, a, point) < 0.0
                })
        });
        if let Some(at) = ear {
            triangles.push(corners(at));
        }
        left.remove(ear.unwrap_or(0));
    }
    if let [a, b, c] = left[..] {
        triangles.push([a, b, c]);
    }
    triangles
}

struct Bed<'a> {
    outline: &'a [Vec2],
    base: f32,
    top: f32,
    tilt: Vec2,
    lean: Vec2,
    bevel: f32,
    tone: f32,
    foot: f32,
}

fn prism(shape: &mut Shape, bed: &Bed<'_>) {
    let outline = bed.outline;
    let count = outline.len();
    let sign = winding(outline);
    let before = |at: usize| (at + count - 1) % count;
    let outward: Vec<Vec2> = (0..count)
        .map(|at| {
            let along = (outline[(at + 1) % count] - outline[at]).normalize_or(Vec2::X);
            Vec2::new(along.y, -along.x) * sign
        })
        .collect();
    let inset: Vec<Vec2> = (0..count)
        .map(|at| {
            let (first, second) = (-outward[before(at)], -outward[at]);
            let reach = bed.bevel / math::max(1.0 + first.dot(second), 0.35);
            outline[at] + (first + second) * reach
        })
        .collect();
    let center = outline.iter().copied().sum::<Vec2>() / count as f32;
    let height = |point: Vec2, level: f32| level + bed.tilt.dot(point - center);
    let thickness = math::max(bed.top - bed.base, 1.0e-3);
    let up = Vec3::new(-bed.tilt.x, 1.0, -bed.tilt.y).normalize();
    let tops: Vec<u32> = inset
        .iter()
        .map(|&point| {
            let moved = point + bed.lean;
            let top = Vec3::new(moved.x, height(point, bed.top), moved.y);
            shape.vertex(top, up, bed.tone)
        })
        .collect();
    for [a, b, c] in ears(&inset) {
        shape.face([tops[a], tops[b], tops[c]], up);
    }
    let grounded = bed.tone * bed.foot;
    let floor: Vec<u32> = outline
        .iter()
        .map(|&point| shape.vertex(Vec3::new(point.x, bed.base, point.y), Vec3::NEG_Y, grounded))
        .collect();
    for [a, b, c] in ears(outline) {
        shape.face([floor[a], floor[b], floor[c]], Vec3::NEG_Y);
    }
    let slope = bed.lean / thickness;
    let raised = bed.lean * ((thickness - bed.bevel) / thickness);
    for at in 0..count {
        let next = (at + 1) % count;
        let here = outward[at];
        let ends = [(at, outward[before(at)]), (next, outward[next])].map(|(corner, other)| {
            let turned = if here.dot(other) >= SMOOTH {
                (here + other).normalize_or(here)
            } else {
                here
            };
            let normal = Vec3::new(turned.x, -turned.dot(slope), turned.y).normalize();
            let point = outline[corner];
            let moved = point + raised;
            let high = Vec3::new(moved.x, height(point, bed.top) - bed.bevel, moved.y);
            (
                shape.vertex(Vec3::new(point.x, bed.base, point.y), normal, grounded),
                shape.vertex(high, normal, bed.tone),
            )
        });
        let [(low, high), (low_next, high_next)] = ends;
        let facing = Vec3::new(here.x, 0.0, here.y);
        shape.face([low, low_next, high_next], facing);
        shape.face([low, high_next, high], facing);
        shape.face([high, high_next, tops[next]], facing + up);
        shape.face([high, tops[next], tops[at]], facing + up);
    }
}

struct Face {
    crest: [f32; 2],
    bottom: f32,
    steep: f32,
    lead: [f32; 2],
}

fn joints(dice: &mut Dice, half: f32) -> Vec<f32> {
    let mut lines = vec![-half];
    let mut last = -half;
    while last < half - 0.6 {
        last = math::min(last + dice.between(0.6, 1.4), half);
        lines.push(last);
    }
    if let Some(end) = lines.last_mut() {
        *end = half;
    }
    lines
}

fn bench_fronts(dice: &mut Dice, lines: &[f32], wall: f32, floor: &[f32]) -> Vec<f32> {
    let columns = lines.len() - 1;
    let half = lines[columns];
    let mut gone = vec![0.0_f32; columns];
    for _ in 0..dice.count(0, 2) {
        let start = dice.count(0, columns as u32 - 1) as usize;
        let length = dice.count(1, 2) as usize;
        let depth = dice.between(0.25, 0.6);
        for lost in gone.iter_mut().skip(start).take(length) {
            *lost = math::max(*lost, depth);
        }
    }
    (0..columns)
        .map(|column| {
            let middle = 0.5 * (lines[column] + lines[column + 1]) / half;
            let front = wall - gone[column] - 0.8 * math::pow(middle.abs(), 2.5);
            math::max(front, floor[column])
        })
        .collect()
}

fn bed_outline(dice: &mut Dice, lines: &[f32], fronts: &[f32], back: f32, shear: f32) -> Vec<Vec2> {
    let mut outline: Vec<Vec2> = Vec::with_capacity(2 * fronts.len() + 2);
    for (column, &front) in fronts.iter().enumerate() {
        let start = Vec2::new(
            front + dice.between(-0.07, 0.07),
            lines[column] + dice.between(-0.06, 0.06),
        );
        let end = Vec2::new(
            front + dice.between(-0.07, 0.07),
            lines[column + 1] + dice.between(-0.06, 0.06),
        );
        match outline.last_mut() {
            Some(last) if (last.x - start.x).abs() < 0.16 => *last = (*last + start) * 0.5,
            _ => outline.push(start),
        }
        outline.push(end);
    }
    outline.push(Vec2::new(back + dice.between(-0.1, 0.1), lines[lines.len() - 1]));
    outline.push(Vec2::new(back + dice.between(-0.1, 0.1), lines[0]));
    for point in &mut outline {
        point.y += shear * point.x;
    }
    outline
}

fn stack(seed: u32, kind: u32, face: &Face) -> Shape {
    let mut dice = Dice::new(seed, kind);
    let mut shape = Shape::default();
    let half = dice.between(1.9, 2.5);
    let lines = joints(&mut dice, half);
    let columns = lines.len() - 1;
    let shear = dice.between(-0.22, 0.22);
    let tilt = Vec2::new(dice.between(-0.03, 0.03), dice.between(-0.03, 0.03));
    let crest = dice.between(face.crest[0], face.crest[1]);
    let lead = dice.between(face.lead[0], face.lead[1]);
    let mut levels = vec![crest];
    let mut level = crest;
    while level > face.bottom + 0.05 {
        level -= dice.between(0.5, 1.1);
        if level - face.bottom < 0.4 {
            level = face.bottom;
        }
        levels.push(level);
    }
    let mut floor = vec![f32::NEG_INFINITY; columns];
    for (bench, span) in levels.windows(2).enumerate() {
        let (top, base) = (span[0], span[1]);
        let wall = lead + (crest - top) / face.steep + dice.between(-0.1, 0.2);
        let fronts = bench_fronts(&mut dice, &lines, wall, &floor);
        floor = fronts.iter().map(|front| front - OVERHANG).collect();
        let (mut from, mut to) = (0, columns);
        if bench == 0 && columns > 3 {
            from = dice.count(0, 1) as usize;
            to = columns - dice.count(0, 1) as usize;
        }
        let nearest = fronts[from..to].iter().copied().fold(f32::INFINITY, math::min);
        let back = nearest - dice.between(1.4, 1.9);
        let beds = if top - base < 0.8 {
            dice.count(1, 2)
        } else {
            dice.count(2, 3)
        } as usize;
        let mut cuts: Vec<f32> = (1..beds).map(|_| dice.roll()).collect();
        cuts.sort_by(f32::total_cmp);
        let mut low = base;
        for bed in 0..beds {
            let high = cuts
                .get(bed)
                .map_or(top, |cut| base + (top - base) * (0.25 + 0.5 * cut));
            if high - low < THINNEST && bed + 1 < beds {
                continue;
            }
            let mut offset = dice.between(-0.05, 0.05);
            if high - low < 0.3 && dice.roll() < 0.4 {
                offset -= dice.between(0.06, 0.14);
            }
            let moved: Vec<f32> = fronts[from..to].iter().map(|front| front + offset).collect();
            let outline = bed_outline(&mut dice, &lines[from..=to], &moved, back, shear);
            let slab = Bed {
                outline: &outline,
                base: low,
                top: high,
                tilt,
                lean: Vec2::ZERO,
                bevel: math::min(0.06, 0.3 * (high - low)),
                tone: dice.between(0.86, 1.0),
                foot: 0.92,
            };
            prism(&mut shape, &slab);
            low = high;
        }
    }
    shape
}

fn block(seed: u32) -> Shape {
    let mut dice = Dice::new(seed, BLOCK);
    let mut shape = Shape::default();
    let corners = dice.count(4, 6);
    let long = dice.between(0.42, 0.6);
    let wide = long * dice.between(0.55, 0.95);
    let turn = dice.between(0.0, std::f32::consts::TAU);
    let outline: Vec<Vec2> = (0..corners)
        .map(|corner| {
            let share = (corner as f32 + dice.between(-0.3, 0.3)) / corners as f32;
            let reach = dice.between(0.75, 1.1);
            let (sin, cos) = math::sin_cos(turn + share * std::f32::consts::TAU);
            Vec2::new(long * reach * cos, wide * reach * sin)
        })
        .collect();
    let thick = dice.between(0.25, 0.75) * wide * 2.0;
    let tilt = Vec2::new(dice.between(-0.25, 0.25), dice.between(-0.25, 0.25));
    let lean = Vec2::new(dice.between(-0.2, 0.2), dice.between(-0.2, 0.2)) * thick;
    let chunk = Bed {
        outline: &outline,
        base: -0.5 * thick,
        top: 0.5 * thick,
        tilt,
        lean,
        bevel: 0.035,
        tone: dice.between(0.88, 1.0),
        foot: 1.0,
    };
    prism(&mut shape, &chunk);
    let rest = Quat::from_rotation_x(dice.between(-0.35, 0.35))
        * Quat::from_rotation_z(dice.between(-0.35, 0.35));
    shape.turned(rest)
}

fn cobble(seed: u32) -> Shape {
    let mut dice = Dice::new(seed, 0);
    let faces: Vec<(Vec3, f32)> = (icosphere(0).0.into_iter())
        .map(|corner| {
            let facing = (corner + dice.direction() * 0.45).normalize();
            (facing, dice.between(0.42, 0.56))
        })
        .collect();
    let stretch = Vec3::new(1.0, dice.between(0.55, 0.8), dice.between(0.7, 1.0));
    let turn = Quat::from_rotation_y(dice.between(0.0, std::f32::consts::TAU));
    let offset = Vec3::new(dice.roll(), dice.roll(), dice.roll()) * 37.0;
    let edge = dice.between(5.0, 8.0);
    let surface = |direction: Vec3| {
        let crowded: f32 = faces
            .iter()
            .map(|&(normal, distance)| {
                math::pow(math::max(direction.dot(normal), 0.0) / distance, edge)
            })
            .sum();
        let radius = math::pow(math::max(crowded, 1.0e-12), -1.0 / edge);
        let worn = 1.0 + 0.08 * (noise(direction * 1.9 + offset) - 0.5);
        turn * (direction * radius * worn * stretch)
    };
    let (directions, triangles) = icosphere(LEVELS);
    let mut points: Vec<Vec3> = directions.into_iter().map(surface).collect();
    let lowest = points.iter().map(|point| point.y).fold(0.0, math::min);
    for point in &mut points {
        let settled = math::clamp(point.y / math::min(lowest, -1.0e-3), 0.0, 1.0);
        point.y *= 1.0 - 0.3 * settled * settled;
    }
    let mut normals = vec![Vec3::ZERO; points.len()];
    for &[a, b, c] in &triangles {
        let [pa, pb, pc] = [a, b, c].map(|index| points[index as usize]);
        let face = (pb - pa).cross(pc - pa);
        for index in [a, b, c] {
            normals[index as usize] += face;
        }
    }
    Shape {
        shade: vec![1.0; points.len()],
        normals: normals
            .into_iter()
            .map(|normal| normal.normalize_or(Vec3::Y))
            .collect(),
        points,
        triangles,
    }
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
