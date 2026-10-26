//! Common junipers (Juniperus communis) for the highlands, the scene's own
//! content: low, dark bushes spreading wider than they stand tall, their
//! limbs rising from a short base and arching out to a dome over it, each
//! limb a flat card of needles across it and an upright card through it
//! over the spruces' bark and needle atlas ([`super::trees`]), the needle
//! spray on the cards and the bark on the stems. A card's normal is mostly
//! the dome's, so the bush lights as one mound, and partly its face's own:
//! cards carry each face with vertices of its own. Vertex colors darken the
//! bush toward its inside and its foot, and grey its needles' green toward
//! the blue of a juniper's, whose upper faces bear a broad white band of
//! stomata. Coarser levels drop whole limbs, the last turned round the bush
//! first, and the stems; every level's cards are cut to the needles' alpha
//! ([`trim_cards`]). Shadow maps draw each bush as a closed solid dome
//! within its cards' outline.

use std::f32::consts::{FRAC_PI_2, TAU};

use engine::prelude::*;
use engine::terrain::{PlacementAsset, PlacementLod};

use super::trees::{ATLAS, BARK, BARK_MIDDLE, CUTOFF, NEEDLES, texels};
use crate::kit::random::{golden_hash, unit};
use crate::kit::records;

/// Juniper `seed` as a placement asset drawn with `material`, the spruces'
/// needles ([`super::trees::material`]): its levels, finest first,
/// switching at [`JUNIPER_ERRORS`], and its shadow proxy, each a record
/// ([`records::mesh`]) whose handle is reserved until the record is here.
/// It stands rigid and collides with nothing: the scene sets its sway and
/// collider.
pub(crate) fn juniper(
    world: &mut World,
    seed: u32,
    material: MaterialHandle,
) -> engine::Result<PlacementAsset> {
    let mut levels = (0..DETAILS.len())
        .map(|detail| {
            let mut identity = seed.to_le_bytes().to_vec();
            identity.extend_from_slice(&(detail as u32).to_le_bytes());
            records::mesh(world, "highlands_juniper", &identity, move || {
                bush(seed, detail)
            })
        })
        .collect::<engine::Result<Vec<_>>>()?;
    let shadow = records::mesh(
        world,
        "highlands_juniper_shadow",
        &seed.to_le_bytes(),
        move || shadow_proxy(seed),
    )?;
    let mesh = levels.remove(0);
    Ok(PlacementAsset {
        mesh,
        material,
        lods: levels
            .into_iter()
            .zip(JUNIPER_ERRORS)
            .map(|(mesh, error)| PlacementLod { mesh, error })
            .collect(),
        parts: Vec::new(),
        sway: 0.0,
        collider: None,
        shadow: Some(shadow),
    })
}

/// How finely a level is built: its cards' segments, the share of the
/// bush's limbs it keeps, four at least, the first turned round it, which
/// spread round it at any count ([`GOLDEN_ANGLE`]), and whether it builds
/// their stems.
struct Detail {
    bends: usize,
    keep: f32,
    stems: bool,
}

/// The full bush, then each coarser level.
const DETAILS: [Detail; 3] = [
    Detail {
        bends: 2,
        keep: 1.0,
        stems: true,
    },
    Detail {
        bends: 1,
        keep: 0.7,
        stems: false,
    },
    Detail {
        bends: 1,
        keep: 0.45,
        stems: false,
    },
];

/// The errors, in meters, the coarser levels show: about the gaps the limbs
/// they drop open between their neighbors' overlapping cards, a tenth and
/// then a fifth of a card's width of some 0.7 meters, rather than any
/// geometric deviation, which cards make meaningless.
const JUNIPER_ERRORS: [f32; 2] = [0.07, 0.14];

/// A card segment's corners as its triangles, its first two vertices
/// across its base and the next two across its tip: its front's, then its
/// back's.
const FRONT: [u32; 6] = [0, 1, 2, 1, 3, 2];
const BACK: [u32; 6] = [0, 2, 1, 1, 2, 3];

/// The golden angle (radians): each limb turns this much further round the
/// bush than the one before, so the first of them, however many a level
/// keeps, spread evenly round it.
const GOLDEN_ANGLE: f32 = 2.399_963;

/// How far out along a limb its cards begin: within that it is bare stem,
/// hidden inside the bush.
const BARE: f32 = 0.3;

/// A limb's cards' half widths as shares of its length: the flat card
/// across it and the upright card through it, wide enough that neighboring
/// limbs' cards overlap and close the dome.
const SPREAD: f32 = 0.34;
const UPRIGHT: f32 = 0.24;

/// How far a limb's tip arches down below its line, as a share of its
/// length, where it spreads level: juniper limbs rise and nod outward, the
/// flatter the more.
const ARCH: f32 = 0.16;

/// How much of a card's normal is the dome's rather than its face's own:
/// the bush lights as one mound, each card still by the way it faces.
const DOME_NORMAL: f32 = 0.75;

/// The share of the spray's red, green and blue a juniper's needles keep:
/// the white band of stomata along each needle greys their green toward
/// blue.
const GLAUCOUS: [f32; 3] = [0.82, 0.88, 0.97];

/// A stem's sides, and its radius at its foot (meters), halving out to its
/// cards.
const STEM_SIDES: u32 = 4;
const STEM_RADIUS: f32 = 0.035;

/// A shadow proxy's sides, its rings' rises up the dome (shares of a
/// quarter turn), and how far out toward the dome over the cards it
/// reaches: within their outline.
const PROXY_SIDES: u32 = 8;
const PROXY_RINGS: [f32; 3] = [0.0, 0.45, 0.8];
const PROXY: f32 = 0.8;

/// Juniper `seed` built at `detail` (0 the full bush, up to 2): about a
/// meter tall and two across before placement scales it, its stems' foot a
/// little into the ground, its cards cut to the needles' alpha.
fn bush(seed: u32, detail: usize) -> engine::Result<Mesh> {
    let build = &DETAILS[detail];
    let shape = Shape::of(seed);
    let mut builder = Builder::default();
    let count = shape.limbs.len();
    let kept = ((count as f32 * build.keep).round() as usize).clamp(4, count);
    for limb in &shape.limbs[..kept] {
        if build.stems {
            builder.stem(limb);
        }
        builder.limb(&shape, limb, build.bends);
    }
    let atlas = AlphaImage {
        width: ATLAS,
        height: ATLAS,
        rgba: texels(),
    };
    builder
        .mesh()
        .and_then(|mesh| trim_cards(&mesh, atlas, CUTOFF))
        .map_err(|error| {
            engine::Error::from(error).context(format!("juniper {seed} at detail {detail}"))
        })
}

/// Juniper `seed`'s shadow proxy, which shadow maps draw in place of its
/// levels: a closed dome [`PROXY`] of the way out to the dome over its
/// cards, a ring on the ground and two up its side under its top, closed
/// underneath, a few dozen triangles. Its vertices carry the streams its
/// levels' do, white and amid the bark.
fn shadow_proxy(seed: u32) -> engine::Result<Mesh> {
    let shape = Shape::of(seed);
    let mut builder = Builder::default();
    let top = Vec3::new(0.0, PROXY * shape.height, 0.0);
    let apex = builder.push(top, Vec3::Y, BARK_MIDDLE, [1.0; 3]);
    let under = builder.push(
        Vec3::new(0.0, -0.05, 0.0),
        Vec3::NEG_Y,
        BARK_MIDDLE,
        [1.0; 3],
    );
    let first = builder.vertices.len() as u32;
    for rise in PROXY_RINGS {
        let (up, out) = math::sin_cos(rise * FRAC_PI_2);
        for side in 0..PROXY_SIDES {
            let (sin, cos) = math::sin_cos(TAU * side as f32 / PROXY_SIDES as f32);
            let point = Vec3::new(
                cos * out * shape.radius,
                up * shape.height,
                sin * out * shape.radius,
            ) * PROXY;
            builder.push(point, shape.normal(point), BARK_MIDDLE, [1.0; 3]);
        }
    }
    let rings = PROXY_RINGS.len() as u32;
    let at = |ring: u32, side: u32| first + ring * PROXY_SIDES + side % PROXY_SIDES;
    for side in 0..PROXY_SIDES {
        for ring in 0..rings - 1 {
            let [a, b, c, d] = [
                at(ring, side),
                at(ring + 1, side),
                at(ring, side + 1),
                at(ring + 1, side + 1),
            ];
            builder.indices.extend([a, b, c, b, d, c]);
        }
        builder
            .indices
            .extend([at(rings - 1, side), apex, at(rings - 1, side + 1)]);
        builder
            .indices
            .extend([at(0, side), at(0, side + 1), under]);
    }
    builder.mesh().map_err(|error| {
        engine::Error::from(error).context(format!("juniper {seed}'s shadow proxy"))
    })
}

/// A juniper's shape, which every level and its shadow proxy keep: its
/// height and the radius of the dome over it, and its limbs, each turned
/// the golden angle further round it than the one before.
struct Shape {
    height: f32,
    radius: f32,
    limbs: Vec<Limb>,
}

/// A limb: where it leaves the base, the way it points out from it and the
/// way it rises along, its length, the share of it its tip arches down
/// ([`ARCH`]) and the shade of its needles.
struct Limb {
    start: Vec3,
    outward: Vec3,
    direction: Vec3,
    length: f32,
    droop: f32,
    shade: f32,
}

impl Limb {
    /// The point `t` of its length out (1 its tip), arching down toward its
    /// tip.
    fn at(&self, t: f32) -> Vec3 {
        self.start + self.direction * (self.length * t)
            - Vec3::Y * (self.droop * self.length * t * t)
    }
}

impl Shape {
    /// The shape of seed `seed`: a dome 0.85 to 1.15 meters tall and 0.9 to
    /// 1.25 in radius over 10 to 13 limbs from its base, rising from 12 to 78
    /// degrees, spread over that by the golden ratio's own sequence so every
    /// first few span it, their tips 85 to 97 percent of the way out to the
    /// dome.
    fn of(seed: u32) -> Self {
        let random = |index: u32| unit(golden_hash(seed.wrapping_mul(0x2c1b_3c6d) ^ index));
        let height = 0.85 + 0.3 * random(1);
        let radius = 0.9 + 0.35 * random(2);
        let count = 10 + golden_hash(seed ^ 0x51ed_270b) % 4;
        let twist = random(3) * TAU;
        let rising = random(4);
        let limbs = (0..count)
            .map(|limb| {
                let index = 100 + limb * 8;
                let (sin, cos) = math::sin_cos(twist + GOLDEN_ANGLE * limb as f32);
                let outward = Vec3::new(cos, 0.0, sin);
                let spread = (rising + 0.618_034 * limb as f32).fract();
                let (rise_sin, rise_cos) = math::sin_cos(0.21 + 1.15 * spread);
                // How far the dome over the bush lies along the limb's way.
                let across = rise_cos / radius;
                let up = rise_sin / height;
                let dome = 1.0 / (across * across + up * up).sqrt();
                Limb {
                    start: outward * 0.06 - Vec3::Y * 0.04,
                    outward,
                    direction: outward * rise_cos + Vec3::Y * rise_sin,
                    length: dome * (0.85 + 0.12 * random(index)),
                    droop: ARCH * rise_cos,
                    shade: 0.85 + 0.3 * random(index + 1),
                }
            })
            .collect();
        Self {
            height,
            radius,
            limbs,
        }
    }

    /// The dome's outward normal at `point`, as if from a middle a quarter
    /// of its height under the ground: the bush's sides light as a mound's
    /// do and its top as the sky does.
    fn normal(&self, point: Vec3) -> Vec3 {
        let across = self.radius * self.radius;
        let up = self.height * self.height;
        Vec3::new(
            point.x / across,
            (point.y + 0.25 * self.height) / up,
            point.z / across,
        )
        .normalize_or(Vec3::Y)
    }

    /// The color of a limb of shade `shade` at `point`: darker toward the
    /// bush's inside and its foot, which its own needles shade, and greyed
    /// as juniper needles are ([`GLAUCOUS`]).
    fn color(&self, point: Vec3, shade: f32) -> [f32; 3] {
        let across = (point.x * point.x + point.z * point.z) / (self.radius * self.radius);
        let up = point.y / self.height;
        let out = (across + up * up).sqrt();
        let inside = 0.45 + 0.55 * math::clamp(out, 0.0, 1.0);
        let foot = 0.7 + 0.3 * math::clamp(point.y / (0.4 * self.height), 0.0, 1.0);
        GLAUCOUS.map(|channel| channel * inside * foot * shade)
    }
}

/// A mesh being built: its vertices, their colors and its triangles.
#[derive(Default)]
struct Builder {
    vertices: Vec<MeshVertex>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

impl Builder {
    /// Adds a vertex; returns its index.
    fn push(&mut self, position: Vec3, normal: Vec3, uv: [f32; 2], color: [f32; 3]) -> u32 {
        self.vertices.push(MeshVertex {
            position: position.to_array(),
            normal: normal.normalize_or(Vec3::Y).to_array(),
            uv,
        });
        let [red, green, blue] = color.map(|channel| math::clamp(channel, 0.0, 1.0));
        self.colors.push([red, green, blue, 1.0]);
        self.vertices.len() as u32 - 1
    }

    /// The mesh built.
    fn mesh(self) -> Result<Mesh, AssetError> {
        let Self {
            vertices,
            colors,
            indices,
        } = self;
        Mesh::new(vertices, indices).and_then(|mesh| mesh.with_colors(colors))
    }

    /// `limb`'s stem from its foot to a little past where its cards begin:
    /// a tube of [`STEM_SIDES`] sides in the bark, narrowing by half, darker
    /// at its foot.
    fn stem(&mut self, limb: &Limb) {
        let (from, to) = (limb.at(0.0), limb.at(BARE + 0.1));
        let along = (to - from).normalize_or(Vec3::Y);
        let side_a = along.cross(Vec3::Y).normalize_or(Vec3::X);
        let side_b = side_a.cross(along);
        let first = self.vertices.len() as u32;
        for side in 0..=STEM_SIDES {
            let (sin, cos) = math::sin_cos(TAU * side as f32 / STEM_SIDES as f32);
            let out = side_a * cos + side_b * sin;
            let u = BARK[0] + (BARK[1] - BARK[0]) * side as f32 / STEM_SIDES as f32;
            for (point, radius, v, shade) in [
                (from, STEM_RADIUS, 0.99, 0.5),
                (to, 0.5 * STEM_RADIUS, 0.8, 0.8),
            ] {
                self.push(point + out * radius, out, [u, v], [shade; 3]);
            }
        }
        for side in 0..STEM_SIDES {
            let [a, b, c, d] = [0, 1, 2, 3].map(|corner| first + side * 2 + corner);
            self.indices.extend([a, b, c, b, d, c]);
        }
    }

    /// `limb`'s cards over its outer part, from [`BARE`] of its length out
    /// to its tip in `bends` segments: a flat one across it and an upright
    /// one through it, lit and colored as the bush is about them.
    fn limb(&mut self, shape: &Shape, limb: &Limb, bends: usize) {
        let spine: Vec<Vec3> = (0..=bends)
            .map(|step| limb.at(BARE + (1.0 - BARE) * step as f32 / bends as f32))
            .collect();
        let across = Vec3::new(-limb.outward.z, 0.0, limb.outward.x);
        let upright = limb.direction.cross(across).normalize_or(Vec3::Y);
        self.card(shape, &spine, across * (SPREAD * limb.length), limb.shade);
        self.card(shape, &spine, upright * (UPRIGHT * limb.length), limb.shade);
    }

    /// A card along `spine` (its inner end first), reaching `half` to each
    /// side and mapping the spray from its base to its tip, of a limb of
    /// shade `shade`: each vertex lit as [`DOME_NORMAL`] of the dome's normal
    /// there and the rest of the face's own, and colored as the bush is
    /// there ([`Shape::color`]). Each face has vertices of its own.
    fn card(&mut self, shape: &Shape, spine: &[Vec3], half: Vec3, shade: f32) {
        let last = spine.len() - 1;
        for (facing, corners) in [(1.0_f32, FRONT), (-1.0, BACK)] {
            let first = self.vertices.len() as u32;
            for (step, &center) in spine.iter().enumerate() {
                let t = step as f32 / last as f32;
                let v = NEEDLES.base + (NEEDLES.tip - NEEDLES.base) * t;
                // The face's own normal, across the spine's way here.
                let way = spine[(step + 1).min(last)] - spine[step.saturating_sub(1)];
                let face = half.cross(way).normalize_or(Vec3::Y) * facing;
                for side in [-1.0_f32, 1.0] {
                    let point = center + half * side;
                    let normal = shape.normal(point) * DOME_NORMAL + face * (1.0 - DOME_NORMAL);
                    let uv = [NEEDLES.center + NEEDLES.half * side, v];
                    self.push(point, normal, uv, shape.color(point, shade));
                }
            }
            for step in 0..last as u32 {
                let base = first + step * 2;
                self.indices.extend(corners.map(|corner| base + corner));
            }
        }
    }
}
