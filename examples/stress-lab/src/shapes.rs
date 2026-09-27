//! Low-poly shapes and textures of the city scene, built through the public
//! asset API. Every mesh fits the unit cube centered at its origin, so an
//! instance's scale is its size and a unit cuboid collider bounds it. Faces
//! are flat-shaded; parts that rest on something omit faces nobody sees.

use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

use engine::prelude::*;

use crate::scene::hash;

/// A rectangle of texture coordinates; `v` grows downward.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Uv {
    u0: f32,
    v0: f32,
    u1: f32,
    v1: f32,
}

impl Uv {
    const FULL: Self = Self::new(0.0, 0.0, 1.0, 1.0);

    const fn new(u0: f32, v0: f32, u1: f32, v1: f32) -> Self {
        Self { u0, v0, u1, v1 }
    }

    fn at(self, u: f32, v: f32) -> [f32; 2] {
        [
            self.u0 + (self.u1 - self.u0) * u,
            self.v0 + (self.v1 - self.v0) * v,
        ]
    }

    /// The part of this rectangle between `top` and `bottom` (fractions of
    /// its height, top first).
    fn band(self, top: f32, bottom: f32) -> Self {
        let v = |fraction: f32| self.v0 + (self.v1 - self.v0) * fraction;
        Self::new(self.u0, v(top), self.u1, v(bottom))
    }
}

// Façade textures: 30 floors of windows above a 16-texel roof band.
const FLOORS: f32 = 30.0;
const WALL: f32 = 240.0 / 256.0;
/// The top `floors` of a façade's thirty.
const fn facade(floors: f32) -> Uv {
    Uv::new(0.0, 0.0, 1.0, WALL * floors / FLOORS)
}
const ROOF: Uv = Uv::new(0.1, 0.955, 0.9, 0.995);
// Foliage textures: leaves above, bark below.
const LEAVES: Uv = Uv::new(0.0, 0.0, 1.0, 0.7);
const BARK: Uv = Uv::new(0.0, 0.8, 1.0, 1.0);
// Vehicle textures: a window band across the upper half.
const BODY: Uv = Uv::new(0.0, 0.55, 1.0, 1.0);
const WINDOWS: Uv = Uv::new(0.0, 0.1, 1.0, 0.5);

#[derive(Default)]
struct Builder {
    vertices: Vec<MeshVertex>,
    indices: Vec<u32>,
}

impl Builder {
    /// A flat convex polygon fanned from its first corner, facing `outward`.
    fn face(&mut self, corners: &[Vec3], uvs: &[[f32; 2]], outward: Vec3) {
        // Newell's normal: counterclockwise corners seen from its side.
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
            return;
        }
        let flip = normal.dot(outward) < 0.0;
        let normal = if flip { -normal } else { normal }.normalize();
        let base = self.vertices.len() as u32;
        for (corner, uv) in corners.iter().zip(uvs) {
            self.vertices.push(MeshVertex {
                position: corner.to_array(),
                normal: normal.to_array(),
                uv: *uv,
            });
        }
        for index in 1..corners.len() as u32 - 1 {
            if flip {
                self.indices.extend([base, base + index + 1, base + index]);
            } else {
                self.indices.extend([base, base + index, base + index + 1]);
            }
        }
    }

    /// A quad through its bottom-left, bottom-right, top-right and top-left
    /// corners as seen from outside.
    fn quad(&mut self, corners: [Vec3; 4], uv: Uv, outward: Vec3) {
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
    fn cuboid(&mut self, min: Vec3, max: Vec3, side: Uv, cap: Uv, bottom: bool) {
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
    fn lathe(
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
    fn place(&mut self, start: usize, matrix: Mat4) {
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
    fn ridge(&mut self, x0: f32, x1: f32, section: [(f32, f32); 3], side: Uv) {
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

    fn mark(&self) -> usize {
        self.vertices.len()
    }

    fn finish(self) -> Result<Mesh, String> {
        Mesh::new(self.vertices, self.indices).map_err(|error| error.to_string())
    }
}

/// Every shape of the city; the first eight survive at `mesh_variety` 8,
/// the first 32 at 32, and all at 64. Others fall back to a coarser shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Shape {
    Block,
    Tower,
    Round,
    Roof,
    Setback,
    Tree,
    Car,
    Dome,
    House,
    Midrise,
    Setback2,
    Setback4,
    Twisted,
    Hex,
    Tapered,
    Spire,
    Ell,
    Pine,
    Bus,
    Wagon,
    Boat,
    Drone,
    Jib,
    Rim,
    Cabin,
    Lamp,
    Antenna,
    WaterTower,
    Fountain,
    Wedge,
    Obelisk,
    Pillar,
    TwistedTall,
    Octa,
    Crowned,
    Ziggurat,
    Tee,
    Court,
    Poplar,
    Bush,
    Helicopter,
    Rotor,
    Beacon,
    Billboard,
    Chimney,
    Tank,
    Sawtooth,
    TvTower,
    Stadium,
    Pagoda,
    Arch,
    Deck,
}

impl Shape {
    pub(crate) const ALL: [Shape; 52] = [
        Shape::Block,
        Shape::Tower,
        Shape::Round,
        Shape::Roof,
        Shape::Setback,
        Shape::Tree,
        Shape::Car,
        Shape::Dome,
        Shape::House,
        Shape::Midrise,
        Shape::Setback2,
        Shape::Setback4,
        Shape::Twisted,
        Shape::Hex,
        Shape::Tapered,
        Shape::Spire,
        Shape::Ell,
        Shape::Pine,
        Shape::Bus,
        Shape::Wagon,
        Shape::Boat,
        Shape::Drone,
        Shape::Jib,
        Shape::Rim,
        Shape::Cabin,
        Shape::Lamp,
        Shape::Antenna,
        Shape::WaterTower,
        Shape::Fountain,
        Shape::Wedge,
        Shape::Obelisk,
        Shape::Pillar,
        Shape::TwistedTall,
        Shape::Octa,
        Shape::Crowned,
        Shape::Ziggurat,
        Shape::Tee,
        Shape::Court,
        Shape::Poplar,
        Shape::Bush,
        Shape::Helicopter,
        Shape::Rotor,
        Shape::Beacon,
        Shape::Billboard,
        Shape::Chimney,
        Shape::Tank,
        Shape::Sawtooth,
        Shape::TvTower,
        Shape::Stadium,
        Shape::Pagoda,
        Shape::Arch,
        Shape::Deck,
    ];

    /// The coarser shape standing in for this one when fewer meshes exist.
    fn fallback(self) -> Shape {
        use Shape::*;
        match self {
            Block | Tower | Roof | Tree | Car | Jib | Rim | Cabin | Lamp | Pillar => Block,
            Round | Setback | House | Midrise | Twisted | Tapered => Tower,
            Dome => Round,
            Setback2 | Setback4 => Setback,
            Hex | WaterTower => Round,
            Spire | Wedge => Roof,
            Ell => Midrise,
            Pine => Tree,
            Bus | Boat | Drone => Car,
            Wagon => Bus,
            Antenna | Obelisk => Spire,
            Fountain => Dome,
            TwistedTall => Twisted,
            Octa | Crowned | Chimney => Round,
            Ziggurat | Pagoda => Setback4,
            Tee | Court => Ell,
            Poplar | Bush => Tree,
            Helicopter => Drone,
            Rotor => Jib,
            Beacon => Antenna,
            Billboard | Arch | Deck => Block,
            Tank => WaterTower,
            Sawtooth => Wedge,
            TvTower => Obelisk,
            Stadium => Dome,
        }
    }

    /// The shape drawn when `variety` meshes exist (1, 8, 32 or 64).
    pub(crate) fn resolve(self, variety: u32) -> Shape {
        let mut shape = self;
        while shape as usize >= variety as usize {
            shape = shape.fallback();
        }
        shape
    }

    /// Whether a unit sphere collider fits the shape better than a cuboid.
    pub(crate) fn round(self) -> bool {
        use Shape::*;
        matches!(
            self,
            Round
                | Dome
                | Tree
                | Hex
                | Pine
                | Rim
                | Fountain
                | Octa
                | Crowned
                | Poplar
                | Bush
                | Tank
                | Stadium
        )
    }

    fn build(self) -> Result<Mesh, String> {
        use Shape::*;
        let mut b = Builder::default();
        let v = Vec3::new;
        let box_ = |b: &mut Builder, min: Vec3, max: Vec3, side: Uv, cap: Uv| {
            b.cuboid(min, max, side, cap, true)
        };
        match self {
            Block => return Ok(Mesh::cube()),
            Tower => b.cuboid(
                v(-0.5, -0.5, -0.5),
                v(0.5, 0.5, 0.5),
                facade(30.0),
                ROOF,
                false,
            ),
            House => b.cuboid(
                v(-0.5, -0.5, -0.5),
                v(0.5, 0.5, 0.5),
                facade(3.0),
                ROOF,
                false,
            ),
            Midrise => b.cuboid(
                v(-0.5, -0.5, -0.5),
                v(0.5, 0.5, 0.5),
                facade(12.0),
                ROOF,
                false,
            ),
            Round => b.lathe(
                &[(0.5, -0.5), (0.5, 0.5)],
                12,
                0.0,
                facade(30.0),
                ROOF,
                false,
            ),
            Hex => b.lathe(
                &[(0.5, -0.5), (0.5, 0.5)],
                6,
                0.0,
                facade(24.0),
                ROOF,
                false,
            ),
            Octa => b.lathe(
                &[(0.5, -0.5), (0.5, 0.5)],
                8,
                PI / 8.0,
                facade(30.0),
                ROOF,
                false,
            ),
            Crowned => b.lathe(
                &[
                    (0.42, -0.5),
                    (0.42, 0.28),
                    (0.5, 0.3),
                    (0.5, 0.34),
                    (0.3, 0.36),
                    (0.3, 0.42),
                    (0.0, 0.5),
                ],
                12,
                0.0,
                facade(30.0),
                ROOF,
                false,
            ),
            Roof => b.lathe(
                &[(0.707, -0.5), (0.0, 0.5)],
                4,
                FRAC_PI_4,
                Uv::FULL,
                Uv::FULL,
                false,
            ),
            Spire => b.lathe(
                &[(0.5, -0.5), (0.0, 0.5)],
                8,
                0.0,
                Uv::FULL,
                Uv::FULL,
                false,
            ),
            Obelisk => b.lathe(
                &[(0.42, -0.5), (0.3, 0.3), (0.0, 0.5)],
                4,
                FRAC_PI_4,
                Uv::FULL,
                ROOF,
                false,
            ),
            Tapered => b.lathe(
                &[(0.707, -0.5), (0.42, 0.5)],
                4,
                FRAC_PI_4,
                facade(30.0),
                ROOF,
                false,
            ),
            Dome => b.lathe(
                &[
                    (0.5, -0.5),
                    (0.462, -0.117),
                    (0.354, 0.207),
                    (0.191, 0.424),
                    (0.0, 0.5),
                ],
                12,
                0.0,
                Uv::FULL,
                Uv::FULL,
                false,
            ),
            Setback | Setback2 | Setback4 | Ziggurat => {
                let tiers: &[(f32, f32, f32)] = match self {
                    Setback => &[(0.5, -0.5, 0.0), (0.36, 0.0, 0.3), (0.22, 0.3, 0.5)],
                    Setback2 => &[(0.5, -0.5, 0.2), (0.3, 0.2, 0.5)],
                    Setback4 => &[
                        (0.5, -0.5, -0.05),
                        (0.38, -0.05, 0.2),
                        (0.26, 0.2, 0.38),
                        (0.14, 0.38, 0.5),
                    ],
                    _ => &[
                        (0.5, -0.5, -0.3),
                        (0.42, -0.3, -0.1),
                        (0.34, -0.1, 0.1),
                        (0.26, 0.1, 0.3),
                        (0.18, 0.3, 0.5),
                    ],
                };
                for &(half, y0, y1) in tiers {
                    let floors = math::max((y1 - y0) * FLOORS, 1.0);
                    b.cuboid(
                        v(-half, y0, -half),
                        v(half, y1, half),
                        facade(floors),
                        ROOF,
                        false,
                    );
                }
            }
            Twisted | TwistedTall => {
                let (slabs, twist) = if self == Twisted {
                    (8, FRAC_PI_4)
                } else {
                    (12, FRAC_PI_2)
                };
                for slab in 0..slabs {
                    let (y0, y1) = (
                        slab as f32 / slabs as f32 - 0.5,
                        (slab + 1) as f32 / slabs as f32 - 0.5,
                    );
                    let band = facade(30.0).band(
                        1.0 - (slab + 1) as f32 / slabs as f32,
                        1.0 - slab as f32 / slabs as f32,
                    );
                    let start = b.mark();
                    b.cuboid(v(-0.35, y0, -0.35), v(0.35, y1, 0.35), band, ROOF, false);
                    b.place(
                        start,
                        Mat4::from_rotation_y(twist * slab as f32 / (slabs - 1) as f32),
                    );
                }
            }
            Ell => {
                b.cuboid(
                    v(-0.5, -0.5, -0.5),
                    v(0.5, 0.5, 0.0),
                    facade(12.0),
                    ROOF,
                    false,
                );
                b.cuboid(
                    v(-0.5, -0.5, 0.0),
                    v(0.0, 0.2, 0.5),
                    facade(7.0),
                    ROOF,
                    false,
                );
            }
            Tee => {
                b.cuboid(
                    v(-0.5, -0.5, -0.5),
                    v(0.5, 0.0, -0.1),
                    facade(5.0),
                    ROOF,
                    false,
                );
                b.cuboid(
                    v(-0.2, -0.5, -0.1),
                    v(0.2, 0.5, 0.5),
                    facade(20.0),
                    ROOF,
                    false,
                );
            }
            Court => {
                b.cuboid(
                    v(-0.5, -0.5, -0.5),
                    v(-0.2, 0.5, 0.5),
                    facade(12.0),
                    ROOF,
                    false,
                );
                b.cuboid(
                    v(0.2, -0.5, -0.5),
                    v(0.5, 0.5, 0.5),
                    facade(12.0),
                    ROOF,
                    false,
                );
                b.cuboid(
                    v(-0.2, -0.5, -0.5),
                    v(0.2, 0.3, -0.2),
                    facade(9.0),
                    ROOF,
                    false,
                );
            }
            Pagoda => b.lathe(
                &[
                    (0.354, -0.5),
                    (0.354, -0.25),
                    (0.707, -0.25),
                    (0.283, -0.12),
                    (0.283, 0.02),
                    (0.566, 0.02),
                    (0.212, 0.13),
                    (0.212, 0.24),
                    (0.424, 0.24),
                    (0.1, 0.36),
                    (0.0, 0.5),
                ],
                4,
                FRAC_PI_4,
                facade(8.0),
                ROOF,
                false,
            ),
            TvTower => b.lathe(
                &[
                    (0.12, -0.5),
                    (0.08, 0.1),
                    (0.3, 0.13),
                    (0.4, 0.18),
                    (0.32, 0.25),
                    (0.06, 0.28),
                    (0.03, 0.45),
                    (0.0, 0.5),
                ],
                12,
                0.0,
                Uv::FULL,
                Uv::FULL,
                false,
            ),
            Tree | Pine | Poplar | Bush => {
                if self != Bush {
                    b.lathe(&[(0.1, -0.5), (0.07, -0.1)], 6, 0.0, BARK, BARK, false);
                }
                match self {
                    Tree => b.lathe(
                        &[(0.0, -0.2), (0.45, 0.05), (0.35, 0.3), (0.0, 0.5)],
                        7,
                        0.0,
                        LEAVES,
                        LEAVES,
                        false,
                    ),
                    Pine => {
                        b.lathe(&[(0.5, -0.3), (0.05, 0.1)], 8, 0.0, LEAVES, LEAVES, true);
                        b.lathe(&[(0.4, -0.05), (0.03, 0.3)], 8, 0.3, LEAVES, LEAVES, true);
                        b.lathe(&[(0.3, 0.15), (0.0, 0.5)], 8, 0.6, LEAVES, LEAVES, true);
                    }
                    Poplar => b.lathe(
                        &[
                            (0.0, -0.3),
                            (0.3, -0.1),
                            (0.35, 0.2),
                            (0.2, 0.4),
                            (0.0, 0.5),
                        ],
                        6,
                        0.0,
                        LEAVES,
                        LEAVES,
                        false,
                    ),
                    _ => b.lathe(
                        &[(0.0, -0.5), (0.5, -0.2), (0.4, 0.2), (0.0, 0.5)],
                        6,
                        0.0,
                        LEAVES,
                        LEAVES,
                        false,
                    ),
                }
            }
            Car => {
                box_(&mut b, v(-0.5, -0.5, -0.5), v(0.5, 0.0, 0.5), BODY, BODY);
                b.cuboid(
                    v(-0.25, 0.0, -0.42),
                    v(0.25, 0.45, 0.42),
                    WINDOWS,
                    BODY,
                    false,
                );
            }
            Bus => {
                box_(
                    &mut b,
                    v(-0.5, -0.5, -0.5),
                    v(0.5, 0.35, 0.5),
                    WINDOWS,
                    BODY,
                );
                b.cuboid(v(-0.48, 0.35, -0.4), v(0.48, 0.5, 0.4), BODY, BODY, false);
            }
            Wagon => {
                box_(&mut b, v(-0.5, -0.5, -0.5), v(0.5, 0.3, 0.5), WINDOWS, BODY);
                b.cuboid(v(-0.48, 0.3, -0.35), v(0.48, 0.5, 0.35), BODY, BODY, false);
            }
            Boat => {
                // Hull with a pointed bow toward +x.
                let (y0, y1) = (-0.5, -0.1);
                box_(&mut b, v(-0.5, y0, -0.35), v(0.25, y1, 0.35), BODY, BODY);
                let bow = [v(0.25, y0, -0.35), v(0.25, y0, 0.35), v(0.5, y0, 0.0)];
                let top = [v(0.25, y1, -0.35), v(0.25, y1, 0.35), v(0.5, y1, 0.0)];
                let uv = [[0.0, 1.0], [1.0, 1.0], [0.5, 0.6]];
                b.face(&top, &uv, Vec3::Y);
                b.face(&bow, &uv, Vec3::NEG_Y);
                b.quad([bow[1], bow[2], top[2], top[1]], BODY, v(1.0, 0.0, 1.0));
                b.quad([bow[2], bow[0], top[0], top[2]], BODY, v(1.0, 0.0, -1.0));
                b.cuboid(v(-0.3, y1, -0.22), v(0.1, 0.3, 0.22), WINDOWS, BODY, false);
                b.cuboid(v(-0.2, 0.3, -0.02), v(-0.16, 0.5, 0.02), BODY, BODY, false);
            }
            Drone => {
                box_(
                    &mut b,
                    v(-0.18, -0.12, -0.18),
                    v(0.18, 0.08, 0.18),
                    BODY,
                    BODY,
                );
                box_(
                    &mut b,
                    v(-0.36, -0.02, -0.03),
                    v(0.36, 0.04, 0.03),
                    BODY,
                    BODY,
                );
                box_(
                    &mut b,
                    v(-0.03, -0.01, -0.36),
                    v(0.03, 0.05, 0.36),
                    BODY,
                    BODY,
                );
                for (x, z) in [(0.35, 0.0), (-0.35, 0.0), (0.0, 0.35), (0.0, -0.35)] {
                    let start = b.mark();
                    b.lathe(
                        &[(0.15, 0.05), (0.15, 0.08)],
                        8,
                        0.0,
                        WINDOWS,
                        WINDOWS,
                        true,
                    );
                    b.place(start, Mat4::from_translation(v(x, 0.0, z)));
                }
            }
            Helicopter => {
                box_(
                    &mut b,
                    v(-0.2, -0.3, -0.18),
                    v(0.25, 0.15, 0.18),
                    WINDOWS,
                    BODY,
                );
                box_(
                    &mut b,
                    v(-0.5, -0.05, -0.04),
                    v(-0.2, 0.05, 0.04),
                    BODY,
                    BODY,
                );
                box_(
                    &mut b,
                    v(-0.5, 0.05, -0.02),
                    v(-0.42, 0.25, 0.02),
                    BODY,
                    BODY,
                );
                box_(
                    &mut b,
                    v(-0.2, -0.5, -0.2),
                    v(0.25, -0.45, -0.16),
                    BODY,
                    BODY,
                );
                box_(&mut b, v(-0.2, -0.5, 0.16), v(0.25, -0.45, 0.2), BODY, BODY);
                box_(
                    &mut b,
                    v(-0.03, 0.15, -0.03),
                    v(0.03, 0.22, 0.03),
                    BODY,
                    BODY,
                );
            }
            Rotor => {
                // Three flat blades about a shaft along Y.
                b.lathe(&[(0.06, -0.5), (0.06, 0.5)], 8, 0.0, BODY, BODY, true);
                for blade in 0..3 {
                    let start = b.mark();
                    box_(&mut b, v(0.04, 0.3, -0.07), v(0.49, 0.38, 0.07), BODY, BODY);
                    b.place(start, Mat4::from_rotation_y(TAU * blade as f32 / 3.0));
                }
            }
            Jib => {
                // Arm and counter-arm about the pivot at the origin.
                box_(
                    &mut b,
                    v(-0.5, 0.3, -0.06),
                    v(0.5, 0.45, 0.06),
                    Uv::FULL,
                    Uv::FULL,
                );
                box_(
                    &mut b,
                    v(-0.48, 0.05, -0.12),
                    v(-0.3, 0.3, 0.12),
                    Uv::FULL,
                    Uv::FULL,
                );
                box_(&mut b, v(-0.1, -0.5, -0.1), v(0.1, 0.3, 0.1), WINDOWS, BODY);
                box_(
                    &mut b,
                    v(0.3, 0.2, -0.05),
                    v(0.4, 0.3, 0.05),
                    Uv::FULL,
                    Uv::FULL,
                );
                b.lathe(
                    &[(0.05, 0.45), (0.0, 0.5)],
                    4,
                    FRAC_PI_4,
                    Uv::FULL,
                    Uv::FULL,
                    false,
                );
            }
            Rim => {
                // A wheel in the XY plane: rim segments, spokes and a hub.
                let segments = 16;
                for segment in 0..segments {
                    let start = b.mark();
                    box_(
                        &mut b,
                        v(0.42, -0.09, -0.05),
                        v(0.48, 0.09, 0.05),
                        Uv::FULL,
                        Uv::FULL,
                    );
                    b.place(
                        start,
                        Mat4::from_rotation_z(TAU * segment as f32 / segments as f32),
                    );
                }
                for spoke in 0..8 {
                    let start = b.mark();
                    box_(
                        &mut b,
                        v(0.0, -0.01, -0.01),
                        v(0.45, 0.01, 0.01),
                        Uv::FULL,
                        Uv::FULL,
                    );
                    b.place(start, Mat4::from_rotation_z(TAU * spoke as f32 / 8.0));
                }
                let start = b.mark();
                b.lathe(
                    &[(0.05, -0.08), (0.05, 0.08)],
                    8,
                    0.0,
                    Uv::FULL,
                    Uv::FULL,
                    true,
                );
                b.place(start, Mat4::from_rotation_x(FRAC_PI_2));
            }
            Cabin => {
                box_(
                    &mut b,
                    v(-0.5, -0.5, -0.5),
                    v(0.5, 0.25, 0.5),
                    WINDOWS,
                    BODY,
                );
                b.lathe(
                    &[(0.707, 0.25), (0.0, 0.5)],
                    4,
                    FRAC_PI_4,
                    BODY,
                    BODY,
                    false,
                );
            }
            Lamp => {
                b.lathe(&[(0.06, -0.5), (0.04, 0.42)], 6, 0.0, BODY, BODY, false);
                box_(&mut b, v(0.0, 0.38, -0.03), v(0.4, 0.42, 0.03), BODY, BODY);
                box_(
                    &mut b,
                    v(0.3, 0.33, -0.08),
                    v(0.5, 0.39, 0.08),
                    WINDOWS,
                    WINDOWS,
                );
            }
            Antenna => {
                b.lathe(
                    &[(0.25, -0.5), (0.05, 0.35)],
                    4,
                    FRAC_PI_4,
                    Uv::FULL,
                    Uv::FULL,
                    false,
                );
                box_(
                    &mut b,
                    v(-0.3, -0.1, -0.02),
                    v(0.3, -0.06, 0.02),
                    Uv::FULL,
                    Uv::FULL,
                );
                box_(
                    &mut b,
                    v(-0.02, 0.1, -0.22),
                    v(0.02, 0.14, 0.22),
                    Uv::FULL,
                    Uv::FULL,
                );
                b.lathe(
                    &[(0.0, 0.33), (0.1, 0.42), (0.0, 0.5)],
                    6,
                    0.0,
                    WINDOWS,
                    WINDOWS,
                    false,
                );
            }
            Beacon => {
                b.lathe(&[(0.08, -0.5), (0.08, -0.05)], 6, 0.0, BODY, BODY, false);
                box_(
                    &mut b,
                    v(-0.5, -0.05, -0.08),
                    v(0.5, 0.05, 0.08),
                    WINDOWS,
                    WINDOWS,
                );
                box_(
                    &mut b,
                    v(-0.08, -0.04, -0.5),
                    v(0.08, 0.06, 0.5),
                    WINDOWS,
                    WINDOWS,
                );
                b.lathe(
                    &[(0.0, 0.05), (0.2, 0.25), (0.0, 0.5)],
                    8,
                    0.0,
                    WINDOWS,
                    WINDOWS,
                    false,
                );
            }
            Billboard => {
                box_(
                    &mut b,
                    v(-0.5, 0.0, -0.04),
                    v(0.5, 0.5, 0.04),
                    Uv::FULL,
                    BODY,
                );
                box_(
                    &mut b,
                    v(-0.33, -0.5, -0.03),
                    v(-0.27, 0.0, 0.03),
                    BODY,
                    BODY,
                );
                box_(&mut b, v(0.27, -0.5, -0.03), v(0.33, 0.0, 0.03), BODY, BODY);
            }
            WaterTower => {
                for (x, z) in [(0.25, 0.25), (-0.25, 0.25), (-0.25, -0.25), (0.25, -0.25)] {
                    b.cuboid(
                        v(x - 0.04, -0.5, z - 0.04),
                        v(x + 0.04, 0.0, z + 0.04),
                        BARK,
                        BARK,
                        false,
                    );
                }
                b.lathe(
                    &[(0.4, 0.0), (0.4, 0.3), (0.0, 0.5)],
                    10,
                    0.0,
                    Uv::FULL,
                    Uv::FULL,
                    true,
                );
            }
            Tank => b.lathe(
                &[(0.5, -0.5), (0.5, 0.2), (0.35, 0.4), (0.0, 0.5)],
                12,
                0.0,
                Uv::FULL,
                Uv::FULL,
                false,
            ),
            Chimney => b.lathe(
                &[(0.5, -0.5), (0.35, 0.4), (0.5, 0.42), (0.5, 0.5)],
                8,
                0.0,
                Uv::FULL,
                ROOF,
                false,
            ),
            Fountain => b.lathe(
                &[
                    (0.5, -0.5),
                    (0.5, -0.4),
                    (0.2, -0.4),
                    (0.2, -0.2),
                    (0.3, -0.2),
                    (0.3, -0.15),
                    (0.1, -0.15),
                    (0.08, 0.2),
                    (0.2, 0.25),
                    (0.0, 0.3),
                ],
                10,
                0.0,
                Uv::FULL,
                Uv::FULL,
                false,
            ),
            // Roofs along X: one ridge, or steep teeth repeating along Z.
            Wedge => b.ridge(-0.5, 0.5, [(-0.5, -0.5), (0.5, -0.5), (0.0, 0.5)], Uv::FULL),
            Sawtooth => {
                for tooth in 0..4 {
                    let (z0, z1) = (-0.5 + tooth as f32 * 0.25, -0.25 + tooth as f32 * 0.25);
                    b.ridge(-0.5, 0.5, [(z0, -0.5), (z1, -0.5), (z0, 0.5)], Uv::FULL);
                }
            }
            Pillar => {
                b.cuboid(
                    v(-0.15, -0.5, -0.15),
                    v(0.15, 0.35, 0.15),
                    Uv::FULL,
                    Uv::FULL,
                    false,
                );
                box_(
                    &mut b,
                    v(-0.5, 0.35, -0.25),
                    v(0.5, 0.5, 0.25),
                    Uv::FULL,
                    Uv::FULL,
                );
            }
            Deck => {
                box_(
                    &mut b,
                    v(-0.5, -0.5, -0.5),
                    v(0.5, 0.0, 0.5),
                    Uv::FULL,
                    Uv::FULL,
                );
                b.cuboid(v(-0.5, 0.0, 0.45), v(0.5, 0.5, 0.5), BODY, BODY, false);
                b.cuboid(v(-0.5, 0.0, -0.5), v(0.5, 0.5, -0.45), BODY, BODY, false);
            }
            Arch => {
                b.cuboid(
                    v(-0.5, -0.5, -0.3),
                    v(-0.3, 0.3, 0.3),
                    facade(4.0),
                    ROOF,
                    false,
                );
                b.cuboid(
                    v(0.3, -0.5, -0.3),
                    v(0.5, 0.3, 0.3),
                    facade(4.0),
                    ROOF,
                    false,
                );
                box_(
                    &mut b,
                    v(-0.5, 0.3, -0.3),
                    v(0.5, 0.5, 0.3),
                    facade(2.0),
                    ROOF,
                );
            }
            Stadium => {
                // Outer wall, a rim and seating sloping down to the field.
                let sides = 16;
                for index in 0..sides {
                    let angle = |step: f32| TAU * (index as f32 + step) / sides as f32;
                    let at = |radius: f32, y: f32, step: f32| {
                        let (sin, cos) = math::sin_cos(angle(step));
                        v(radius * cos, y, radius * sin)
                    };
                    let (sin, cos) = math::sin_cos(angle(0.5));
                    let radial = v(cos, 0.0, sin);
                    b.quad(
                        [
                            at(0.5, -0.5, 1.0),
                            at(0.5, -0.5, 0.0),
                            at(0.5, 0.1, 0.0),
                            at(0.5, 0.1, 1.0),
                        ],
                        facade(4.0),
                        radial,
                    );
                    b.quad(
                        [
                            at(0.45, 0.1, 1.0),
                            at(0.45, 0.1, 0.0),
                            at(0.5, 0.1, 0.0),
                            at(0.5, 0.1, 1.0),
                        ],
                        ROOF,
                        Vec3::Y,
                    );
                    b.quad(
                        [
                            at(0.3, -0.4, 0.0),
                            at(0.3, -0.4, 1.0),
                            at(0.45, 0.1, 1.0),
                            at(0.45, 0.1, 0.0),
                        ],
                        Uv::FULL,
                        Vec3::Y - radial,
                    );
                    b.face(
                        &[v(0.0, -0.4, 0.0), at(0.3, -0.4, 1.0), at(0.3, -0.4, 0.0)],
                        &[[0.5, 0.5], [0.5, 0.9], [0.9, 0.9]],
                        Vec3::Y,
                    );
                }
            }
        }
        b.finish()
    }
}

/// Procedural textures; building façades are 64x256 with a roof band. Alpha
/// masks emission (decision 019): lit windows, neon and lamp heads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Tex {
    Office,
    Glass,
    Residential,
    Industrial,
    Road,
    Crossing,
    Paving,
    Grass,
    Water,
    Metal,
    Vehicle,
    Foliage,
    Tiles,
    Neon,
    Lamp,
}

impl Tex {
    const ALL: [Tex; 15] = [
        Tex::Office,
        Tex::Glass,
        Tex::Residential,
        Tex::Industrial,
        Tex::Road,
        Tex::Crossing,
        Tex::Paving,
        Tex::Grass,
        Tex::Water,
        Tex::Metal,
        Tex::Vehicle,
        Tex::Foliage,
        Tex::Tiles,
        Tex::Neon,
        Tex::Lamp,
    ];

    /// How the surface responds to light (decision 019): glass and metal
    /// reflect the sky, water ripples, and lit windows, neon and lamp heads
    /// emit where the texture's alpha lets them.
    pub(crate) fn surface(self) -> Surface {
        let surface = |roughness, metalness, emission, ripples| Surface {
            roughness,
            metalness,
            emission,
            ripples,
        };
        match self {
            Tex::Office => surface(0.6, 0.0, 2.5, 0.0),
            Tex::Glass => surface(0.12, 0.6, 2.0, 0.0),
            Tex::Residential => surface(0.8, 0.0, 2.0, 0.0),
            Tex::Industrial => surface(0.7, 0.2, 1.5, 0.0),
            Tex::Road | Tex::Crossing => surface(0.85, 0.0, 0.0, 0.0),
            Tex::Paving => surface(0.8, 0.0, 0.0, 0.0),
            Tex::Tiles => surface(0.6, 0.0, 0.0, 0.0),
            Tex::Grass | Tex::Foliage => surface(0.95, 0.0, 0.0, 0.0),
            Tex::Water => surface(0.05, 0.0, 0.0, 1.0),
            Tex::Metal => surface(0.45, 0.5, 0.0, 0.0),
            Tex::Vehicle => surface(0.3, 0.5, 0.0, 0.0),
            Tex::Neon => surface(0.4, 0.0, 3.0, 0.0),
            Tex::Lamp => surface(0.5, 0.6, 6.0, 0.0),
        }
    }

    fn build(self) -> Result<Texture, String> {
        let facade = matches!(
            self,
            Tex::Office | Tex::Glass | Tex::Residential | Tex::Industrial
        );
        let (width, height) = if facade { (64, 256) } else { (64, 64) };
        let mut pixels = Vec::with_capacity(width * height * 4);
        let seed = (self as u32).wrapping_mul(0x9e37_79b9);
        for y in 0..height {
            for x in 0..width {
                let noise = (hash(seed ^ (y as u32 * 64 + x as u32)) % 24) as i32 - 12;
                let (rgb, emits) = self.texel(x, y, noise, seed);
                pixels.extend_from_slice(&[rgb[0], rgb[1], rgb[2], if emits { 255 } else { 0 }]);
            }
        }
        Texture::with_emission_mask(width as u32, height as u32, pixels)
            .map_err(|error| error.to_string())
    }

    /// A texel's color and whether it emits.
    fn texel(self, x: usize, y: usize, noise: i32, seed: u32) -> ([u8; 3], bool) {
        let shade = |value: i32| (value + noise).clamp(0, 255) as u8;
        let gray = |value: i32| [shade(value); 3];
        let color = match self {
            Tex::Office | Tex::Glass | Tex::Residential | Tex::Industrial => {
                return self.facade(x, y, noise, seed);
            }
            Tex::Neon => {
                let band = (x / 8 + y / 8) % 3;
                return (
                    gray(match band {
                        0 => 255,
                        1 => 190,
                        _ => 120,
                    }),
                    true,
                );
            }
            // The lamp head's texture rows (the shapes' window band) glow
            // warm white; the rest is dark painted metal.
            Tex::Lamp => {
                return if (6..32).contains(&y) {
                    ([shade(255), shade(232), shade(176)], true)
                } else {
                    (gray(70), false)
                };
            }
            _ => self.plain(x, y, noise, seed),
        };
        (color, false)
    }

    /// Façade texels: windows, some lit, between walls, under a roof band.
    fn facade(self, x: usize, y: usize, noise: i32, seed: u32) -> ([u8; 3], bool) {
        let shade = |value: i32| (value + noise).clamp(0, 255) as u8;
        let gray = |value: i32| [shade(value); 3];
        let lit = |cell: u32| hash(seed ^ cell.wrapping_mul(0x2c1b_3c6d)) % 100;
        if y >= 240 {
            return (gray(128), false);
        }
        let (cx, cy) = (x % 8, y % 8);
        let cell = (y / 8 * 8 + x / 8) as u32;
        let window = match self {
            Tex::Office => (1..7).contains(&cx) && (1..6).contains(&cy),
            Tex::Glass => cx != 0 && cy != 0,
            Tex::Residential => (2..6).contains(&cx) && (2..6).contains(&cy),
            _ => (1..7).contains(&cx) && cy == 3,
        };
        if !window {
            return (
                gray(match self {
                    Tex::Glass => 110,
                    Tex::Residential => 205,
                    Tex::Industrial => 160,
                    _ => 175,
                }),
                false,
            );
        }
        let lit_share = match self {
            Tex::Glass => 18,
            Tex::Residential => 30,
            _ => 35,
        };
        if lit(cell) < lit_share {
            ([shade(255), shade(236), shade(190)], true)
        } else if self == Tex::Glass {
            (gray(62 + (y % 8) as i32 * 4), false)
        } else {
            (gray(48), false)
        }
    }

    fn plain(self, x: usize, y: usize, noise: i32, seed: u32) -> [u8; 3] {
        let shade = |value: i32| (value + noise).clamp(0, 255) as u8;
        let gray = |value: i32| [shade(value); 3];
        match self {
            Tex::Road => {
                let center = (30..34).contains(&y) && x % 16 < 9;
                let edge = y == 3 || y == 60;
                if center {
                    [shade(230), shade(196), shade(70)]
                } else if edge {
                    gray(200)
                } else {
                    gray(58)
                }
            }
            Tex::Crossing => {
                let stripe = (x % 12 < 6) && !(10..54).contains(&y)
                    || (y % 12 < 6) && !(10..54).contains(&x);
                gray(if stripe { 205 } else { 62 })
            }
            Tex::Paving => gray(if x.is_multiple_of(16) || y.is_multiple_of(16) {
                120
            } else {
                168
            }),
            Tex::Grass => [shade(66), shade(132), shade(60)],
            Tex::Water => {
                let ripple = (x + y * 3) % 23 < 2;
                if ripple {
                    [shade(110), shade(170), shade(215)]
                } else {
                    [shade(26), shade(86), shade(150)]
                }
            }
            Tex::Metal => {
                let brace = (x + y) % 16 < 2 || (x + 64 - y % 64) % 16 < 2 || x % 32 < 2;
                gray(if brace { 225 } else { 150 })
            }
            Tex::Vehicle => {
                if (6..32).contains(&y) {
                    if x.is_multiple_of(16) {
                        gray(200)
                    } else {
                        gray(40)
                    }
                } else {
                    gray(232)
                }
            }
            Tex::Foliage => {
                if y >= 48 {
                    [shade(92), shade(64), shade(40)]
                } else {
                    let bright = hash(seed ^ (y as u32 / 3 * 64 + x as u32 / 3)).is_multiple_of(4);
                    if bright {
                        [shade(96), shade(176), shade(70)]
                    } else {
                        [shade(42), shade(118), shade(48)]
                    }
                }
            }
            Tex::Tiles => gray(
                if y.is_multiple_of(8) || (x + y / 8 * 4).is_multiple_of(16) {
                    90
                } else {
                    170
                },
            ),
            // Built by `texel` and `facade`.
            Tex::Office
            | Tex::Glass
            | Tex::Residential
            | Tex::Industrial
            | Tex::Neon
            | Tex::Lamp => gray(0),
        }
    }
}

/// A texture and a linear color: one material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Paint {
    pub tex: Tex,
    pub color: [f32; 3],
}

/// Meshes and textures of one city, and its materials on demand.
pub(crate) struct Library {
    variety: u32,
    meshes: HashMap<Shape, MeshHandle>,
    textures: HashMap<Tex, TextureHandle>,
    materials: HashMap<(Tex, [u32; 3]), MaterialHandle>,
}

impl Library {
    /// Inserts the meshes `variety` keeps (1, 8, 32 or all) and, when
    /// `textured`, the textures, whose alpha masks emission.
    pub(crate) fn new(
        assets: &mut AssetLibrary,
        variety: u32,
        textured: bool,
    ) -> Result<Self, String> {
        let mut meshes = HashMap::new();
        for shape in Shape::ALL {
            let shape = shape.resolve(variety);
            if let std::collections::hash_map::Entry::Vacant(entry) = meshes.entry(shape) {
                entry.insert(
                    assets
                        .insert_mesh(shape.build()?)
                        .map_err(|error| error.to_string())?,
                );
            }
        }
        let mut textures = HashMap::new();
        if textured {
            for tex in Tex::ALL {
                textures.insert(
                    tex,
                    assets
                        .insert_texture(tex.build()?)
                        .map_err(|error| error.to_string())?,
                );
            }
        }
        Ok(Self {
            variety,
            meshes,
            textures,
            materials: HashMap::new(),
        })
    }

    /// `tex`'s surface; without textures only neon emits, as no mask
    /// limits the others' emission to their lit texels.
    fn surface(&self, tex: Tex) -> Surface {
        let surface = tex.surface();
        if self.textures.is_empty() && tex != Tex::Neon {
            Surface {
                emission: 0.0,
                ..surface
            }
        } else {
            surface
        }
    }

    /// The mesh and material drawing `shape` with `paint`.
    pub(crate) fn get(
        &mut self,
        assets: &mut AssetLibrary,
        shape: Shape,
        paint: Paint,
    ) -> Result<Mesh3d, String> {
        let mesh = self.meshes[&shape.resolve(self.variety)];
        let key = (paint.tex, paint.color.map(f32::to_bits));
        let material = match self.materials.get(&key) {
            Some(&material) => material,
            None => {
                let color = [paint.color[0], paint.color[1], paint.color[2], 1.0];
                let material = match self.textures.get(&paint.tex) {
                    Some(&texture) => Material::textured(color, texture),
                    None => Material::new(color),
                }
                .and_then(|material| material.with_surface(self.surface(paint.tex)))
                .map_err(|error| error.to_string())?;
                let handle = assets
                    .insert_material(material)
                    .map_err(|error| error.to_string())?;
                self.materials.insert(key, handle);
                handle
            }
        };
        Ok(Mesh3d { mesh, material })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_fit_the_unit_cube_and_face_their_normals() {
        for shape in Shape::ALL {
            let mesh = shape.build().unwrap();
            let bounds = mesh.bounds();
            assert!(
                bounds.min().cmpge(Vec3::splat(-0.5 - 1.0e-5)).all(),
                "{shape:?} {:?}",
                bounds.min()
            );
            assert!(
                bounds.max().cmple(Vec3::splat(0.5 + 1.0e-5)).all(),
                "{shape:?} {:?}",
                bounds.max()
            );
            // Counterclockwise triangles seen from the side their normal points to.
            let vertices = mesh.vertices();
            for triangle in mesh.indices().chunks(3) {
                let [a, b, c] = [0, 1, 2]
                    .map(|corner| Vec3::from(vertices[triangle[corner] as usize].position));
                let winding = (b - a).cross(c - a);
                let normal = Vec3::from(vertices[triangle[0] as usize].normal);
                assert!(winding.dot(normal) > 0.0, "{shape:?} triangle {triangle:?}");
            }
        }
    }

    #[test]
    fn convex_shapes_face_away_from_their_axis() {
        use Shape::*;
        for shape in [
            Block, Tower, Round, Roof, Dome, Hex, Octa, Spire, Tapered, Obelisk, Tank,
        ] {
            let mesh = shape.build().unwrap();
            for vertex in mesh.vertices() {
                let position = Vec3::from(vertex.position);
                let normal = Vec3::from(vertex.normal);
                // Faces point away from the vertical axis or up/down.
                let radial = Vec3::new(position.x, 0.0, position.z);
                let outward = radial.dot(normal) >= -1.0e-5 || normal.y.abs() > 0.99;
                assert!(outward, "{shape:?} {position} {normal}");
            }
        }
    }

    #[test]
    fn variety_keeps_the_first_shapes_and_falls_back_to_them() {
        assert_eq!(Shape::ALL.len(), 52);
        for (index, shape) in Shape::ALL.iter().enumerate() {
            assert_eq!(*shape as usize, index);
            assert!((shape.fallback() as usize) < index.max(1), "{shape:?}");
        }
        for variety in [1, 8, 32, 64] {
            let kept: std::collections::HashSet<Shape> = Shape::ALL
                .iter()
                .map(|shape| shape.resolve(variety))
                .collect();
            assert_eq!(
                kept.len(),
                Shape::ALL.len().min(variety as usize),
                "variety {variety}"
            );
        }
        assert_eq!(Shape::Twisted.resolve(8), Shape::Tower);
        assert_eq!(Shape::Wagon.resolve(8), Shape::Car);
        assert_eq!(Shape::Pagoda.resolve(32), Shape::Setback4);
        assert_eq!(Shape::Stadium.resolve(1), Shape::Block);
    }

    #[test]
    fn textures_build_with_a_roof_band_on_facades() {
        for tex in Tex::ALL {
            let texture = tex.build().unwrap();
            assert_eq!(texture.width(), 64);
            assert!(texture.height() == 64 || texture.height() == 256);
        }
    }

    #[test]
    fn only_lit_windows_neon_and_lamp_heads_emit() {
        for tex in Tex::ALL {
            let surface = tex.surface();
            assert!(
                Material::new([1.0; 4])
                    .unwrap()
                    .with_surface(surface)
                    .is_ok()
            );
            let texture = tex.build().unwrap();
            let emitting = texture
                .pixels()
                .chunks(4)
                .filter(|texel| texel[3] == 255)
                .count();
            let total = texture.pixels().len() / 4;
            match tex {
                Tex::Neon => assert_eq!(emitting, total),
                Tex::Office | Tex::Glass | Tex::Residential | Tex::Industrial | Tex::Lamp => {
                    assert!(surface.emission > 0.0);
                    assert!(emitting > 0 && emitting < total / 2, "{tex:?} {emitting}");
                }
                _ => assert_eq!(emitting, 0, "{tex:?}"),
            }
            // Masked or not, texels are either off or fully on.
            assert!(
                texture
                    .pixels()
                    .chunks(4)
                    .all(|texel| matches!(texel[3], 0 | 255))
            );
        }
        assert_eq!(Tex::Water.surface().ripples, 1.0);
    }
}
