//! Flat-shaded unit-cube shapes for the city: faces map rectangles of
//! their texture ([`Uv`]); parts that rest on something omit faces nobody
//! sees.

use std::f32::consts::TAU;

use engine::prelude::*;

/// A rectangle of texture coordinates; `v` grows downward.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Uv {
    u0: f32,
    v0: f32,
    u1: f32,
    v1: f32,
}

impl Uv {
    pub(crate) const FULL: Self = Self::new(0.0, 0.0, 1.0, 1.0);

    pub(crate) const fn new(u0: f32, v0: f32, u1: f32, v1: f32) -> Self {
        Self { u0, v0, u1, v1 }
    }

    pub(crate) fn at(self, u: f32, v: f32) -> [f32; 2] {
        [
            self.u0 + (self.u1 - self.u0) * u,
            self.v0 + (self.v1 - self.v0) * v,
        ]
    }

    /// The part of this rectangle between `top` and `bottom` (fractions of
    /// its height, top first).
    pub(crate) fn band(self, top: f32, bottom: f32) -> Self {
        let v = |fraction: f32| self.v0 + (self.v1 - self.v0) * fraction;
        Self::new(self.u0, v(top), self.u1, v(bottom))
    }
}

/// The unit normal of the flat convex polygon `corners` on its `outward`
/// side (Newell's normal: counterclockwise corners seen from its side), and
/// whether its corners turn the other way from there; `None` when it is
/// degenerate. Both mesh builders face polygons with it.
pub(crate) fn polygon_normal(corners: &[Vec3], outward: Vec3) -> Option<(Vec3, bool)> {
    let mut normal = Vec3::ZERO;
    for (index, a) in corners.iter().enumerate() {
        let b = corners[(index + 1) % corners.len()];
        normal += Vec3::new(
            (a.y - b.y) * (a.z + b.z),
            (a.z - b.z) * (a.x + b.x),
            (a.x - b.x) * (a.y + b.y),
        );
    }
    if normal.length_squared() < 1.0e-12 {
        return None;
    }
    let flip = normal.dot(outward) < 0.0;
    Some((if flip { -normal } else { normal }.normalize(), flip))
}

/// The triangles fanning a polygon of `count` corners from vertex `base`,
/// wound the other way when `flip`.
pub(crate) fn fan(indices: &mut Vec<u32>, base: u32, count: u32, flip: bool) {
    for index in 1..count - 1 {
        if flip {
            indices.extend([base, base + index + 1, base + index]);
        } else {
            indices.extend([base, base + index, base + index + 1]);
        }
    }
}

#[derive(Default)]
pub(crate) struct Builder {
    vertices: Vec<MeshVertex>,
    indices: Vec<u32>,
}

impl Builder {
    /// A flat convex polygon fanned from its first corner, facing `outward`.
    pub(crate) fn face(&mut self, corners: &[Vec3], uvs: &[[f32; 2]], outward: Vec3) {
        let Some((normal, flip)) = polygon_normal(corners, outward) else {
            return;
        };
        let base = self.vertices.len() as u32;
        for (corner, uv) in corners.iter().zip(uvs) {
            self.vertices.push(MeshVertex {
                position: corner.to_array(),
                normal: normal.to_array(),
                uv: *uv,
            });
        }
        fan(&mut self.indices, base, corners.len() as u32, flip);
    }

    /// A quad through its bottom-left, bottom-right, top-right and top-left
    /// corners as seen from outside.
    pub(crate) fn quad(&mut self, corners: [Vec3; 4], uv: Uv, outward: Vec3) {
        self.face(
            &corners,
            &[
                uv.at(0.0, 1.0),
                uv.at(1.0, 1.0),
                uv.at(1.0, 0.0),
                uv.at(0.0, 0.0),
            ],
            outward,
        );
    }

    /// An axis-aligned box: its sides map `side`, its top `cap`, and its
    /// bottom `cap` when `bottom`.
    pub(crate) fn cuboid(&mut self, min: Vec3, max: Vec3, side: Uv, cap: Uv, bottom: bool) {
        let [x0, y0, z0] = min.to_array();
        let [x1, y1, z1] = max.to_array();
        let v = Vec3::new;
        let quads = [
            (
                [v(x0, y0, z1), v(x1, y0, z1), v(x1, y1, z1), v(x0, y1, z1)],
                Vec3::Z,
            ),
            (
                [v(x1, y0, z0), v(x0, y0, z0), v(x0, y1, z0), v(x1, y1, z0)],
                Vec3::NEG_Z,
            ),
            (
                [v(x1, y0, z1), v(x1, y0, z0), v(x1, y1, z0), v(x1, y1, z1)],
                Vec3::X,
            ),
            (
                [v(x0, y0, z0), v(x0, y0, z1), v(x0, y1, z1), v(x0, y1, z0)],
                Vec3::NEG_X,
            ),
        ];
        for (corners, outward) in quads {
            self.quad(corners, side, outward);
        }
        self.quad(
            [v(x0, y1, z1), v(x1, y1, z1), v(x1, y1, z0), v(x0, y1, z0)],
            cap,
            Vec3::Y,
        );
        if bottom {
            self.quad(
                [v(x0, y0, z0), v(x1, y0, z0), v(x1, y0, z1), v(x0, y0, z1)],
                cap,
                Vec3::NEG_Y,
            );
        }
    }

    /// A surface of revolution about Y through `profile` points (radius,
    /// height) from bottom to top, with `sides` faces from `angle`. Rings of
    /// radius zero close to a point; other ends get flat caps (the bottom one
    /// only when `bottom`). Each face maps all of `side` across its width.
    pub(crate) fn lathe(
        &mut self,
        profile: &[(f32, f32)],
        sides: u32,
        angle: f32,
        side: Uv,
        cap: Uv,
        bottom: bool,
    ) {
        let low = profile[0].1;
        let height = math::max(profile[profile.len() - 1].1 - low, 1.0e-6);
        let point = |(radius, y): (f32, f32), index: u32| {
            let (sin, cos) = math::sin_cos(angle + TAU * index as f32 / sides as f32);
            Vec3::new(radius * cos, y, radius * sin)
        };
        for pair in profile.windows(2) {
            let ((r0, y0), (r1, y1)) = (pair[0], pair[1]);
            let (v0, v1) = (1.0 - (y0 - low) / height, 1.0 - (y1 - low) / height);
            for index in 0..sides {
                let (sin, cos) = math::sin_cos(angle + TAU * (index as f32 + 0.5) / sides as f32);
                // Outward in the profile plane: up the profile, then turn right.
                let outward = Vec3::new(cos, 0.0, sin) * (y1 - y0) - Vec3::Y * (r1 - r0);
                let (b0, b1) = (point((r0, y0), index), point((r0, y0), index + 1));
                let (t0, t1) = (point((r1, y1), index), point((r1, y1), index + 1));
                // Angles increase leftward seen from outside.
                if r1 == 0.0 {
                    self.face(
                        &[b1, b0, t0],
                        &[side.at(0.0, v0), side.at(1.0, v0), side.at(0.5, v1)],
                        outward,
                    );
                } else if r0 == 0.0 {
                    self.face(
                        &[b0, t0, t1],
                        &[side.at(0.5, v0), side.at(1.0, v1), side.at(0.0, v1)],
                        outward,
                    );
                } else {
                    self.face(
                        &[b1, b0, t0, t1],
                        &[
                            side.at(0.0, v0),
                            side.at(1.0, v0),
                            side.at(1.0, v1),
                            side.at(0.0, v1),
                        ],
                        outward,
                    );
                }
            }
        }
        let mut ring = |(radius, y): (f32, f32), outward: Vec3| {
            let corners: Vec<Vec3> = (0..sides).map(|index| point((radius, y), index)).collect();
            let uvs: Vec<[f32; 2]> = corners
                .iter()
                .map(|corner| {
                    cap.at(
                        corner.x / (2.0 * radius) + 0.5,
                        corner.z / (2.0 * radius) + 0.5,
                    )
                })
                .collect();
            self.face(&corners, &uvs, outward);
        };
        let (first, last) = (profile[0], profile[profile.len() - 1]);
        if bottom && first.0 > 0.0 {
            ring(first, Vec3::NEG_Y);
        }
        if last.0 > 0.0 {
            ring(last, Vec3::Y);
        }
    }

    /// Moves the vertices added since `start` rigidly by `matrix`.
    pub(crate) fn place(&mut self, start: usize, matrix: Mat4) {
        for vertex in &mut self.vertices[start..] {
            vertex.position = matrix
                .transform_point3(Vec3::from(vertex.position))
                .to_array();
            vertex.normal = matrix
                .transform_vector3(Vec3::from(vertex.normal))
                .to_array();
        }
    }

    /// A prism along X from `x0` to `x1` over a triangle `section` of (z, y)
    /// points whose first two lie on the base, which stays open.
    pub(crate) fn ridge(&mut self, x0: f32, x1: f32, section: [(f32, f32); 3], side: Uv) {
        let at = |x: f32, (z, y): (f32, f32)| Vec3::new(x, y, z);
        let uv = [[0.0, 1.0], [1.0, 1.0], [0.5, 0.0]];
        self.face(&section.map(|point| at(x0, point)), &uv, Vec3::NEG_X);
        self.face(&section.map(|point| at(x1, point)), &uv, Vec3::X);
        let centroid = (
            (section[0].0 + section[1].0 + section[2].0) / 3.0,
            (section[0].1 + section[1].1 + section[2].1) / 3.0,
        );
        for (a, b) in [(section[1], section[2]), (section[2], section[0])] {
            let outward = Vec3::new(
                0.0,
                (a.1 + b.1) * 0.5 - centroid.1,
                (a.0 + b.0) * 0.5 - centroid.0,
            );
            self.quad([at(x0, a), at(x1, a), at(x1, b), at(x0, b)], side, outward);
        }
    }

    pub(crate) fn mark(&self) -> usize {
        self.vertices.len()
    }

    pub(crate) fn finish(self) -> Result<Mesh, String> {
        Mesh::new(self.vertices, self.indices).map_err(|error| error.to_string())
    }
}
