mod build;
mod floor;
mod funnel;
mod search;
mod tactics;
mod walk;

use glam::{Vec2, Vec3};
use serde::Deserialize;

pub use floor::{Carved, carve};
pub use search::{Costs, Path};
pub use tactics::{Cover, CoverClass, CoverSpot, Eyes, Tactics};

pub type PolygonId = u32;

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NavNumbers {
    pub weld_m: f32,
    pub locate_height_m: f32,
    pub locate_distance_m: f32,
    pub cover_gap_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spot {
    pub polygon: PolygonId,
    pub point: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Portal {
    pub a: Vec3,
    pub b: Vec3,
    pub polygons: [PolygonId; 2],
    pub width: f32,
    pub walls: [bool; 2],
}

impl Portal {
    fn ends(&self, radius: f32) -> (Vec3, Vec3) {
        let across = if self.width > 0.0 {
            (self.b - self.a) / self.width
        } else {
            Vec3::ZERO
        };
        let room = if self.walls[0] && self.walls[1] {
            lesser(radius, self.width * 0.5)
        } else {
            lesser(radius, self.width)
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Island {
    pub polygons: usize,
    pub area: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BuildReport {
    pub vertices_in: usize,
    pub vertices_welded: usize,
    pub t_junctions: usize,
    pub empty_triangles: usize,
    pub portals: usize,
    pub boundary_edges: usize,
    pub crowded_edges: Vec<[u32; 2]>,
    pub islands: Vec<Island>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavBuildError {
    IndexOutOfRange { polygon: usize, index: u32 },
    Degenerate { polygon: usize },
    NotConvex { polygon: usize },
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
    area: f32,
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
    ) -> Result<NavMesh, NavBuildError> {
        let lists = polygons.iter().map(|p| p.as_ref().to_vec()).collect();
        build::assemble(positions, lists, weld)
    }

    pub fn build_triangles(
        positions: &[Vec3],
        indices: &[u32],
        weld: f32,
    ) -> Result<NavMesh, NavBuildError> {
        build::assemble_triangles(positions, indices, weld)
    }

    pub fn report(&self) -> &BuildReport {
        &self.report
    }

    pub fn polygon_count(&self) -> usize {
        self.polygons.len()
    }

    pub fn island(&self, polygon: PolygonId) -> u32 {
        self.polygon(polygon).island
    }

    fn polygon(&self, polygon: PolygonId) -> &Polygon {
        &self.polygons[polygon as usize]
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

fn lesser(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}

fn greater(a: f32, b: f32) -> f32 {
    if a < b { b } else { a }
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
    let t = lesser(greater((point - a).dot(along) / length, 0.0), 1.0);
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
    lesser(
        lesser(segment_distance(a, p, q), segment_distance(b, p, q)),
        lesser(segment_distance(p, a, b), segment_distance(q, a, b)),
    )
}
