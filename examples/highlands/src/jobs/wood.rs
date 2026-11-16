use std::f32::consts::TAU;

use engine_core::{Error, Mesh, MeshVertex, math};
use glam::Vec3;

const BARK_SPAN: f32 = 1.3;

const CUT_SHADE: f32 = 0.55;

#[derive(Clone, Copy)]
pub(crate) struct Ring {
    pub(crate) at: Vec3,
    pub(crate) radius: f32,
    pub(crate) tint: [f32; 3],
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum End {
    Open,
    Broken,
}

#[derive(Default)]
pub(crate) struct Wood {
    vertices: Vec<MeshVertex>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

struct Frame {
    along: Vec3,
    x: Vec3,
    z: Vec3,
}

fn frames(rings: &[Ring]) -> Vec<Frame> {
    let last = rings.len() - 1;
    let mut x = Vec3::ZERO;
    (0..rings.len())
        .map(|ring| {
            let span = rings[(ring + 1).min(last)].at - rings[ring.saturating_sub(1)].at;
            let along = span.normalize_or(Vec3::Y);
            let carried = match ring {
                0 => along.cross(Vec3::Z),
                _ => x,
            };
            let across = carried - along * carried.dot(along);
            x = across.normalize_or(along.cross(Vec3::X).normalize_or(Vec3::X));
            Frame {
                along,
                x,
                z: x.cross(along),
            }
        })
        .collect()
}

impl Wood {
    fn push(&mut self, position: Vec3, normal: Vec3, uv: [f32; 2], tint: [f32; 3]) -> u32 {
        self.vertices.push(MeshVertex {
            position: position.to_array(),
            normal: normal.normalize_or(Vec3::Y).to_array(),
            uv,
        });
        let [red, green, blue] = tint.map(|channel| math::clamp(channel, 0.0, 1.0));
        self.colors.push([red, green, blue, 1.0]);
        self.vertices.len() as u32 - 1
    }

    pub(crate) fn tube(&mut self, rings: &[Ring], sides: u32, ends: [End; 2]) {
        if rings.len() < 2 || sides < 3 {
            return;
        }
        let frames = frames(rings);
        let girth = TAU * rings.iter().fold(0.0, |widest, ring| math::max(widest, ring.radius));
        let wraps = math::max((girth / BARK_SPAN).round(), 1.0);
        let first = self.vertices.len() as u32;
        let mut run = 0.0;
        for (index, (ring, frame)) in rings.iter().zip(&frames).enumerate() {
            if index > 0 {
                run += ring.at.distance(rings[index - 1].at);
            }
            for side in 0..=sides {
                let (sin, cos) = math::sin_cos(TAU * side as f32 / sides as f32);
                let out = frame.x * cos + frame.z * sin;
                self.push(
                    ring.at + out * ring.radius,
                    out,
                    [wraps * side as f32 / sides as f32, run / BARK_SPAN],
                    ring.tint,
                );
            }
        }
        for ring in 0..rings.len() as u32 - 1 {
            for side in 0..sides {
                let a = first + ring * (sides + 1) + side;
                let [b, c, d] = [sides + 1, 1, sides + 2].map(|step| a + step);
                self.indices.extend([a, b, c, b, d, c]);
            }
        }
        let last = rings.len() - 1;
        for (end, at, facing) in [(ends[0], 0, -1.0), (ends[1], last, 1.0)] {
            if end == End::Broken {
                self.cap(&rings[at], &frames[at], sides, facing);
            }
        }
    }

    fn cap(&mut self, ring: &Ring, frame: &Frame, sides: u32, facing: f32) {
        let normal = frame.along * facing;
        let tint = ring.tint.map(|channel| channel * CUT_SHADE);
        let reach = ring.radius / BARK_SPAN;
        let first = self.vertices.len() as u32;
        for side in 0..sides {
            let (sin, cos) = math::sin_cos(TAU * side as f32 / sides as f32);
            let out = frame.x * cos + frame.z * sin;
            self.push(
                ring.at + out * ring.radius,
                normal + out * 0.5,
                [0.5 + cos * reach, 0.5 + sin * reach],
                tint,
            );
        }
        let center = self.push(
            ring.at + normal * (0.6 * ring.radius),
            normal,
            [0.5, 0.5],
            tint,
        );
        for side in 0..sides {
            let [here, next] = [side, (side + 1) % sides].map(|corner| first + corner);
            self.indices.extend(match facing > 0.0 {
                true => [next, here, center],
                false => [here, next, center],
            });
        }
    }

    pub(crate) fn mesh(self) -> Result<Mesh, Error> {
        Mesh::new(self.vertices, self.indices).and_then(|mesh| mesh.with_colors(self.colors))
    }
}
