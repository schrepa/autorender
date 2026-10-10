//! Flat-shaded mesh building for the metro. Faces take texture coordinates
//! in texture repeats (`Map::Tile`), so tiling materials keep their scale on
//! meshes built at their nominal world size; vertices can carry colors, a
//! second coordinate set and skin weights.

use std::f32::consts::TAU;

use engine::prelude::*;

use crate::shapes::{fan, polygon_normal};

/// How a face's texture coordinates follow its size.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Map {
    /// One repeat per `u` × `v` world units. Upright faces run `u` along
    /// their bottom edge and `v` down from the builder's reference height
    /// (or their own top); flat faces use world x and z.
    Tile(f32, f32),
    /// A fixed rectangle `(u0, v0, u1, v1)` stretched across each face.
    Rect(f32, f32, f32, f32),
}

impl Map {
    /// A patch so small that the face shows one texel region's color.
    pub(crate) const fn patch(u: f32, v: f32) -> Self {
        Map::Rect(u, v, u + 0.004, v + 0.004)
    }

    fn rect(self, u: f32, v: f32) -> [f32; 2] {
        match self {
            Map::Rect(u0, v0, u1, v1) => [u0 + (u1 - u0) * u, v0 + (v1 - v0) * v],
            Map::Tile(..) => [u, v],
        }
    }
}

/// The second coordinates a mesh derives at the end: none, or its height
/// fraction (0 at the top) and position around its footprint, for maps
/// spanning a whole building such as grime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Uv1 {
    None,
    Span,
}

pub(crate) struct Builder {
    vertices: Vec<MeshVertex>,
    indices: Vec<u32>,
    colors: Vec<[f32; 4]>,
    skin: Vec<([u16; 4], [f32; 4])>,
    /// Color of the vertices added next.
    pub color: [f32; 4],
    /// Joints and weights of the vertices added next.
    pub bind: ([u16; 4], [f32; 4]),
    /// Height `Map::Tile` counts `v` down from; the face's own top when unset.
    pub top: Option<f32>,
    pub uv1: Uv1,
    colored: bool,
    skinned: bool,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            colors: Vec::new(),
            skin: Vec::new(),
            color: [1.0; 4],
            bind: ([0; 4], [1.0, 0.0, 0.0, 0.0]),
            top: None,
            uv1: Uv1::None,
            colored: false,
            skinned: false,
        }
    }
}

impl Builder {
    /// Sets the color of the vertices added next; any color but white makes
    /// the mesh carry vertex colors.
    pub(crate) fn paint(&mut self, color: [f32; 3]) {
        self.color = [color[0], color[1], color[2], 1.0];
        self.colored |= color != [1.0; 3];
    }

    /// Binds the vertices added next to one joint.
    pub(crate) fn joint(&mut self, joint: u16) {
        self.bind = ([joint, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]);
        self.skinned = true;
    }

    pub(crate) fn mark(&self) -> usize {
        self.vertices.len()
    }

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
            self.colors.push(self.color);
            self.skin.push(self.bind);
        }
        fan(&mut self.indices, base, corners.len() as u32, flip);
    }

    /// Texture coordinates of a polygon under `map`.
    fn coordinates(&self, corners: &[Vec3], map: Map, outward: Vec3) -> Vec<[f32; 2]> {
        match map {
            Map::Rect(..) => {
                // Stretched over the polygon's own extent in its plane.
                let (u_axis, v_axis) = plane_axes(outward);
                let us: Vec<f32> = corners.iter().map(|corner| corner.dot(u_axis)).collect();
                let vs: Vec<f32> = corners.iter().map(|corner| corner.dot(v_axis)).collect();
                let (u0, u1) = span(&us);
                let (v0, v1) = span(&vs);
                us.iter()
                    .zip(&vs)
                    .map(|(u, v)| {
                        map.rect(
                            (u - u0) / math::max(u1 - u0, 1.0e-6),
                            (v1 - v) / math::max(v1 - v0, 1.0e-6),
                        )
                    })
                    .collect()
            }
            Map::Tile(scale_u, scale_v) => {
                let normal = outward.normalize_or_zero();
                if normal.y.abs() > 0.7 {
                    return corners
                        .iter()
                        .map(|corner| [corner.x / scale_u, corner.z / scale_v])
                        .collect();
                }
                let (u_axis, _) = plane_axes(outward);
                let top = self.top.unwrap_or_else(|| {
                    corners
                        .iter()
                        .fold(f32::NEG_INFINITY, |top, corner| math::max(top, corner.y))
                });
                let start = corners.iter().fold(f32::INFINITY, |low, corner| {
                    math::min(low, corner.dot(u_axis))
                });
                corners
                    .iter()
                    .map(|corner| {
                        [
                            (corner.dot(u_axis) - start) / scale_u,
                            (top - corner.y) / scale_v,
                        ]
                    })
                    .collect()
            }
        }
    }

    /// A polygon mapped by `map`.
    pub(crate) fn poly(&mut self, corners: &[Vec3], map: Map, outward: Vec3) {
        let uvs = self.coordinates(corners, map, outward);
        self.face(corners, &uvs, outward);
    }

    /// A quad through its bottom-left, bottom-right, top-right and top-left
    /// corners as seen from outside.
    pub(crate) fn quad(&mut self, corners: [Vec3; 4], map: Map, outward: Vec3) {
        match map {
            Map::Rect(..) => {
                let uvs = [
                    map.rect(0.0, 1.0),
                    map.rect(1.0, 1.0),
                    map.rect(1.0, 0.0),
                    map.rect(0.0, 0.0),
                ];
                self.face(&corners, &uvs, outward);
            }
            Map::Tile(..) => self.poly(&corners, map, outward),
        }
    }

    /// An axis-aligned box: sides map `side`, the top `cap`, the bottom `cap`
    /// when `bottom`.
    pub(crate) fn cuboid(&mut self, min: Vec3, max: Vec3, side: Map, cap: Map, bottom: bool) {
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

    /// A box with every face mapped by `map`.
    pub(crate) fn block(&mut self, min: Vec3, max: Vec3, map: Map) {
        self.cuboid(min, max, map, map, true);
    }

    /// A surface of revolution about Y through `profile` points (radius,
    /// height) from bottom to top, with `sides` faces from `angle`. Rings of
    /// radius zero close to a point; other ends get flat caps (the bottom one
    /// only when `bottom`).
    pub(crate) fn lathe(
        &mut self,
        profile: &[(f32, f32)],
        sides: u32,
        angle: f32,
        side: Map,
        cap: Map,
        bottom: bool,
    ) {
        let point = |(radius, y): (f32, f32), index: u32| {
            let (sin, cos) = math::sin_cos(angle + TAU * index as f32 / sides as f32);
            Vec3::new(radius * cos, y, radius * sin)
        };
        let top = self.top.unwrap_or(profile[profile.len() - 1].1);
        for pair in profile.windows(2) {
            let ((r0, y0), (r1, y1)) = (pair[0], pair[1]);
            for index in 0..sides {
                let (sin, cos) = math::sin_cos(angle + TAU * (index as f32 + 0.5) / sides as f32);
                let outward = Vec3::new(cos, 0.0, sin) * (y1 - y0) - Vec3::Y * (r1 - r0);
                let (b0, b1) = (point((r0, y0), index), point((r0, y0), index + 1));
                let (t0, t1) = (point((r1, y1), index), point((r1, y1), index + 1));
                // Coordinates around the ring and down from the top.
                let (u0, u1, v0, v1) = match side {
                    Map::Tile(scale_u, scale_v) => {
                        let arc = |radius: f32| TAU * radius / sides as f32 / scale_u;
                        let width = math::max(arc(r0), arc(r1));
                        (
                            index as f32 * width,
                            (index + 1) as f32 * width,
                            (top - y0) / scale_v,
                            (top - y1) / scale_v,
                        )
                    }
                    Map::Rect(..) => {
                        let low = profile[0].1;
                        let height = math::max(profile[profile.len() - 1].1 - low, 1.0e-6);
                        let a = side.rect(0.0, 1.0 - (y0 - low) / height);
                        let b = side.rect(1.0, 1.0 - (y1 - low) / height);
                        (a[0], b[0], a[1], b[1])
                    }
                };
                // Angles increase leftward seen from outside.
                if r1 == 0.0 {
                    self.face(
                        &[b1, b0, t0],
                        &[[u0, v0], [u1, v0], [(u0 + u1) * 0.5, v1]],
                        outward,
                    );
                } else if r0 == 0.0 {
                    self.face(
                        &[b0, t0, t1],
                        &[[(u0 + u1) * 0.5, v0], [u1, v1], [u0, v1]],
                        outward,
                    );
                } else {
                    self.face(
                        &[b1, b0, t0, t1],
                        &[[u0, v0], [u1, v0], [u1, v1], [u0, v1]],
                        outward,
                    );
                }
            }
        }
        let mut ring = |(radius, y): (f32, f32), outward: Vec3| {
            let corners: Vec<Vec3> = (0..sides).map(|index| point((radius, y), index)).collect();
            self.poly(&corners, cap, outward);
        };
        let (first, last) = (profile[0], profile[profile.len() - 1]);
        if bottom && first.0 > 0.0 {
            ring(first, Vec3::NEG_Y);
        }
        if last.0 > 0.0 {
            ring(last, Vec3::Y);
        }
    }

    /// A round bar from `a` to `b`, closed at both ends.
    pub(crate) fn bar(&mut self, a: Vec3, b: Vec3, radius: f32, sides: u32, map: Map) {
        let axis = b - a;
        let length = axis.length();
        if length < 1.0e-5 {
            return;
        }
        let start = self.mark();
        let top = self.top.take();
        self.lathe(
            &[(radius, 0.0), (radius, length)],
            sides,
            0.0,
            map,
            map,
            true,
        );
        self.top = top;
        let rotation = Quat::from_rotation_arc(Vec3::Y, axis / length);
        self.place(start, Mat4::from_rotation_translation(rotation, a));
    }

    /// A prism along X from `x0` to `x1` over a triangle `section` of (z, y)
    /// points whose first two lie on the base, which stays open.
    pub(crate) fn ridge(
        &mut self,
        x0: f32,
        x1: f32,
        section: [(f32, f32); 3],
        side: Map,
        end: Map,
    ) {
        let at = |x: f32, (z, y): (f32, f32)| Vec3::new(x, y, z);
        self.poly(&section.map(|point| at(x0, point)), end, Vec3::NEG_X);
        self.poly(&section.map(|point| at(x1, point)), end, Vec3::X);
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

    /// Two crossed upright quads of `width` × `height` from `base`, turned
    /// by `yaw`: leaf and frond cards, drawn from both sides.
    pub(crate) fn cards(&mut self, base: Vec3, width: f32, height: f32, yaw: f32, map: Map) {
        for turn in [0.0, std::f32::consts::FRAC_PI_2] {
            let (sin, cos) = math::sin_cos(yaw + turn);
            let across = Vec3::new(cos, 0.0, sin) * (width * 0.5);
            let outward = Vec3::new(-sin, 0.0, cos);
            let up = Vec3::Y * height;
            self.quad(
                [
                    base - across,
                    base + across,
                    base + across + up,
                    base - across + up,
                ],
                map,
                outward,
            );
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

    /// The mesh, moved so its bounds center on the origin when `center`
    /// (renderables are placed by their centers) and with the attributes the
    /// build used, and the build-space point that became its origin.
    pub(crate) fn finish(mut self, center: bool) -> Result<(Mesh, Vec3), String> {
        let (min, max) = self.vertices.iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(min, max), vertex| {
                let at = Vec3::from(vertex.position);
                (min.min(at), max.max(at))
            },
        );
        let middle = if center {
            (min + max) * 0.5
        } else {
            Vec3::ZERO
        };
        if center {
            for vertex in &mut self.vertices {
                vertex.position = (Vec3::from(vertex.position) - middle).to_array();
            }
        }
        let uv1 = (self.uv1 == Uv1::Span).then(|| {
            let size = math::max((max.x - min.x) + (max.z - min.z), 1.0e-6);
            let height = math::max(max.y - min.y, 1.0e-6);
            let low = if center {
                (min + max) * 0.5
            } else {
                Vec3::ZERO
            };
            self.vertices
                .iter()
                .map(|vertex| {
                    let at = Vec3::from(vertex.position) + low;
                    [
                        ((at.x - min.x) + (at.z - min.z)) / size,
                        (max.y - at.y) / height,
                    ]
                })
                .collect::<Vec<_>>()
        });
        let mut mesh = Mesh::new(self.vertices, self.indices).map_err(|error| error.to_string())?;
        if self.colored {
            mesh = mesh
                .with_colors(self.colors)
                .map_err(|error| error.to_string())?;
        }
        if let Some(uv1) = uv1 {
            mesh = mesh.with_uv1(uv1).map_err(|error| error.to_string())?;
        }
        if self.skinned {
            let (joints, weights) = self.skin.into_iter().unzip();
            mesh = mesh
                .with_skin(joints, weights)
                .map_err(|error| error.to_string())?;
        }
        Ok((mesh, middle))
    }
}

/// Two axes spanning the plane facing `outward`: `u` runs horizontally to
/// the right seen from outside (along X for flat faces), `v` up the plane.
fn plane_axes(outward: Vec3) -> (Vec3, Vec3) {
    let normal = outward.normalize_or_zero();
    if normal.y.abs() > 0.99 {
        return (Vec3::X, Vec3::NEG_Z * normal.y.signum());
    }
    let u = Vec3::Y.cross(normal).normalize();
    (u, normal.cross(u))
}

fn span(values: &[f32]) -> (f32, f32) {
    values
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &value| {
            (math::min(low, value), math::max(high, value))
        })
}
