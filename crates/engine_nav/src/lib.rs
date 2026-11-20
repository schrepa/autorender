#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod build;
mod funnel;
mod search;
mod walk;

use engine_codec::Error;
use glam::{Vec2, Vec3};

pub use search::{Costs, Path, Scratch};

type PolygonId = u32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spot {
    polygon: PolygonId,
    pub point: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Portal {
    a: Vec3,
    b: Vec3,
    polygons: [PolygonId; 2],
    width: f32,
    walls: [bool; 2],
}

impl Portal {
    fn ends(&self, radius: f32) -> (Vec3, Vec3) {
        let across = if self.width > 0.0 {
            (self.b - self.a) / self.width
        } else {
            Vec3::ZERO
        };
        let room = if self.walls[0] && self.walls[1] {
            engine_math::min(radius, self.width * 0.5)
        } else {
            engine_math::min(radius, self.width)
        };
        let a = if self.walls[0] {
            self.a + across * room
        } else {
            self.a
        };
        let b = if self.walls[1] {
            self.b - across * room
        } else {
            self.b
        };
        (a, b)
    }

    fn toward_line(&self, from: Vec3, to: Vec3, radius: f32) -> Vec3 {
        let (a, b) = self.ends(radius);
        let line = flat(to) - flat(from);
        let side_a = line.perp_dot(flat(a) - flat(from));
        let side_b = line.perp_dot(flat(b) - flat(from));
        if side_a * side_b <= 0.0 && side_a != side_b {
            a + (b - a) * (side_a / (side_a - side_b))
        } else if side_a.abs() <= side_b.abs() {
            a
        } else {
            b
        }
    }

    fn step(&self, step: u32, radius: f32) -> Vec3 {
        let (a, b) = self.ends(radius);
        match step {
            0 => a,
            1 => (self.a + self.b) * 0.5,
            _ => b,
        }
    }

    fn other(&self, polygon: PolygonId) -> PolygonId {
        if self.polygons[0] == polygon {
            self.polygons[1]
        } else {
            self.polygons[0]
        }
    }

    fn passable(&self, radius: f32) -> bool {
        let walls = f32::from(u8::from(self.walls[0]) + u8::from(self.walls[1]));
        self.width >= radius * walls
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BuildReport {
    pub empty_triangles: usize,
    pub portals: usize,
    pub islands: usize,
}

#[derive(Clone, Debug)]
struct Polygon {
    corners: Vec<u32>,
    links: Vec<Option<u32>>,
    normal: Vec3,
    offset: f32,
    min: Vec2,
    max: Vec2,
    low: f32,
    high: f32,
    island: u32,
}

impl Polygon {
    fn height_at(&self, point: Vec2) -> f32 {
        (self.offset - self.normal.x * point.x - self.normal.z * point.y) / self.normal.y
    }

    fn lift(&self, point: Vec2) -> Vec3 {
        Vec3::new(point.x, self.height_at(point), point.y)
    }
}

#[derive(Clone, Debug)]
struct Grid {
    origin: Vec2,
    cell: f32,
    columns: usize,
    rows: usize,
    starts: Vec<u32>,
    items: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct NavMesh {
    vertices: Vec<Vec3>,
    polygons: Vec<Polygon>,
    portals: Vec<Portal>,
    grid: Grid,
    weld: f32,
    report: BuildReport,
}

impl NavMesh {
    pub fn build<P: AsRef<[u32]>>(
        positions: &[Vec3],
        polygons: &[P],
        weld: f32,
    ) -> Result<NavMesh, Error> {
        let lists = polygons.iter().map(|p| p.as_ref().to_vec()).collect();
        build::assemble(positions, lists, weld)
    }

    pub fn build_triangles(
        positions: &[Vec3],
        indices: &[u32],
        weld: f32,
    ) -> Result<NavMesh, Error> {
        build::assemble_triangles(positions, indices, weld)
    }

    pub fn report(&self) -> &BuildReport {
        &self.report
    }

    pub fn polygon_count(&self) -> usize {
        self.polygons.len()
    }

    pub fn weld(&self) -> f32 {
        self.weld
    }

    pub fn joined(&self, a: Spot, b: Spot) -> bool {
        self.polygon(a.polygon).island == self.polygon(b.polygon).island
    }

    fn polygon(&self, polygon: PolygonId) -> &Polygon {
        &self.polygons[polygon as usize]
    }

    fn nearby(&self, min: Vec2, max: Vec2, mut visit: impl FnMut(PolygonId, &Polygon)) {
        let grid = &self.grid;
        let (first_column, first_row) = grid.cell(min);
        let (last_column, last_row) = grid.cell(max);
        for row in first_row..=last_row {
            for column in first_column..=last_column {
                let index = row * grid.columns + column;
                let span = grid.starts[index] as usize..grid.starts[index + 1] as usize;
                for &id in &grid.items[span] {
                    let polygon = self.polygon(id);
                    let (own_column, own_row) = grid.cell(polygon.min);
                    if own_column.max(first_column) == column && own_row.max(first_row) == row {
                        visit(id, polygon);
                    }
                }
            }
        }
    }

    fn edge(&self, polygon: &Polygon, edge: usize) -> (Vec2, Vec2) {
        let a = polygon.corners[edge];
        let b = polygon.corners[(edge + 1) % polygon.corners.len()];
        (
            flat(self.vertices[a as usize]),
            flat(self.vertices[b as usize]),
        )
    }

    fn closest_in(&self, polygon: &Polygon, point: Vec2) -> (Vec2, f32) {
        let mut inside = true;
        let mut best = (point, f32::INFINITY);
        for edge in 0..polygon.corners.len() {
            let (a, b) = self.edge(polygon, edge);
            if turn(a, b, point) < 0.0 {
                inside = false;
            }
            let near = closest_on_segment(point, a, b);
            let distance = near.distance(point);
            if distance < best.1 {
                best = (near, distance);
            }
        }
        if inside { (point, 0.0) } else { best }
    }
}

fn flat(point: Vec3) -> Vec2 {
    Vec2::new(point.x, point.z)
}

fn turn(origin: Vec2, a: Vec2, b: Vec2) -> f32 {
    (a - origin).perp_dot(b - origin)
}

fn closest_on_segment(point: Vec2, a: Vec2, b: Vec2) -> Vec2 {
    let along = b - a;
    let length = along.length_squared();
    if length <= 0.0 {
        return a;
    }
    let t = engine_math::min(engine_math::max((point - a).dot(along) / length, 0.0), 1.0);
    a + along * t
}

fn segment_distance(point: Vec2, a: Vec2, b: Vec2) -> f32 {
    closest_on_segment(point, a, b).distance(point)
}

fn segments_cross(a: Vec2, b: Vec2, p: Vec2, q: Vec2) -> bool {
    let d1 = turn(p, q, a);
    let d2 = turn(p, q, b);
    let d3 = turn(a, b, p);
    let d4 = turn(a, b, q);
    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

fn segments_distance(a: Vec2, b: Vec2, p: Vec2, q: Vec2) -> f32 {
    if segments_cross(a, b, p, q) {
        return 0.0;
    }
    engine_math::min(
        engine_math::min(segment_distance(a, p, q), segment_distance(b, p, q)),
        engine_math::min(segment_distance(p, a, b), segment_distance(q, a, b)),
    )
}
