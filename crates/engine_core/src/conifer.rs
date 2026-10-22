//! Conifers: spruces built in code at every level of detail a placement
//! layer draws, over one atlas of bark, needle mass and needle spray, for
//! the forests any game's terrains place ([`Conifer::asset`]).
//!
//! A tapering trunk carries whorls of drooping branches. Each whorl is a
//! tier of the crown: a closed, opaque cone skirt of needle mass from the
//! stem out to its branches' tips and back in under them, its normals
//! leaning out from the trunk so the crown lights as one volume, its vertex
//! colors darkening the crown's inside and its lower whorls. A thin fringe
//! of masked needle cards along the branches' outer reach draws the
//! crown's silhouette, the asset's part ([`PlacementPart`]). Coarser levels
//! keep the tree's height and crown: they shed fringe cards, straighten
//! their tiers and simplify the trunk, each covering less alpha-tested area
//! than the level before; the coarsest is the solid crown alone. Every
//! level holds the coarsest one within its own surfaces, so shadow maps,
//! which cast that one, never shadow from outside what the camera draws.

use std::f32::consts::{FRAC_PI_2, TAU};

use glam::{Vec2, Vec3};

use crate::math;
use crate::{
    AlphaMode, AssetError, Material, MaterialHandle, Mesh, MeshHandle, MeshVertex,
    PlacementAsset, PlacementLod, PlacementPart, Surface, TextureHandle, TextureRef,
    TextureSource, TextureUsage,
};

/// A conifer of the shape `seed` picks: 13 to 17 meters tall before
/// placement scales it, its base half a meter into the ground.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conifer {
    pub seed: u32,
}

/// What of a conifer a mesh holds: its opaque trunk and crown, or the
/// fringe of masked needle cards along its crown's silhouette.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConiferPart {
    Crown,
    Fringe,
}

/// The atlas's side in texels.
const ATLAS: u32 = 512;

/// The atlas's regions as texture coordinates: the bark across the trunk's
/// sides and the needle mass across a tier's sector, each a quarter of its
/// width less a margin, so filtering never reaches the next; the needle
/// spray's horizontal center and half width in its right half, and its
/// vertical span (base at the bottom).
const BARK: [f32; 2] = [0.01, 0.24];
const NEEDLES: [f32; 2] = [0.26, 0.49];
const SPRAY: [f32; 2] = [0.75, 0.235];
const SPRAY_SPAN: [f32; 2] = [0.99, 0.01];

/// How far out a fringe card reaches along its branch: a little past the
/// tier's rim, so the silhouette is the needles'.
const REACH: f32 = 1.12;

/// The coarsest trunk's radius against the others': within the circle a
/// four-sided trunk's faces touch, so it lies inside every finer trunk.
const INSCRIBED: f32 = 0.707;

/// How finely a level is built: its trunk's sides and segments, whether
/// its tiers bend with their branches (bulging above the straight tiers of
/// coarser levels, as the branches arch down toward their tips), and its
/// fringe, if any.
struct Detail {
    sides: u32,
    segments: u32,
    bent: bool,
    fringe: Option<Fringe>,
}

/// A level's fringe: where along each branch its card starts, the card's
/// segments, whether a hanging card droops under it, on which of a whorl's
/// branches cards stand (every first, every second, staggered from whorl
/// to whorl), and the crossed cards topping the leader.
struct Fringe {
    from: f32,
    bends: usize,
    hanging: bool,
    every: u32,
    leader: u32,
}

/// The full tree, then each coarser level.
const DETAILS: [Detail; Conifer::LEVELS] = [
    Detail {
        sides: 7,
        segments: 8,
        bent: true,
        fringe: Some(Fringe {
            from: 0.45,
            bends: 2,
            hanging: true,
            every: 1,
            leader: 2,
        }),
    },
    Detail {
        sides: 5,
        segments: 4,
        bent: true,
        fringe: Some(Fringe {
            from: 0.55,
            bends: 1,
            hanging: false,
            every: 1,
            leader: 2,
        }),
    },
    Detail {
        sides: 4,
        segments: 2,
        bent: false,
        fringe: Some(Fringe {
            from: 0.65,
            bends: 1,
            hanging: false,
            every: 2,
            leader: 1,
        }),
    },
    Detail {
        sides: 3,
        segments: 1,
        bent: false,
        fringe: None,
    },
];

impl Conifer {
    /// The levels a conifer is built at, the full tree first.
    pub const LEVELS: usize = 4;
    /// The levels holding a fringe: every one but the coarsest.
    pub const FRINGED: usize = 3;
    /// The errors (meters, at scale 1) its coarser levels show, which
    /// switch them against the pixel error: as the fringe's needles blur
    /// into the crown rather than by geometric deviation, which cards make
    /// meaningless.
    pub const ERRORS: [f32; 3] = [0.03, 0.07, 0.16];

    /// Its mesh of `part` at `level` (0 the full tree): the trunk, the
    /// crown's tiers and the leader's shoot, or the fringe's cards, which
    /// only the fringed levels hold.
    pub fn mesh(self, level: usize, part: ConiferPart) -> Result<Mesh, AssetError> {
        let Some(detail) = DETAILS.get(level) else {
            return Err(AssetError::InvalidLevels {
                reason: "conifers have four levels",
            });
        };
        let shape = Shape::of(self.seed);
        let mut builder = Builder::default();
        match (part, &detail.fringe) {
            (ConiferPart::Crown, _) => {
                let coarsest = level + 1 == Self::LEVELS;
                let scale = if coarsest { INSCRIBED } else { 1.0 };
                builder.trunk(&shape, (detail.sides, detail.segments), scale);
                for whorl in &shape.whorls {
                    builder.tier(whorl, detail.bent);
                }
                builder.leader(shape.height);
            }
            (ConiferPart::Fringe, Some(fringe)) => builder.fringe(&shape, fringe),
            (ConiferPart::Fringe, None) => {
                return Err(AssetError::InvalidLevels {
                    reason: "a conifer's coarsest level has no fringe",
                });
            }
        }
        builder.mesh()
    }

    /// The bark, needle mass and needle spray atlas conifers draw with, an
    /// RGBA8 color texture of sRGB color with alpha coverage.
    pub fn atlas() -> TextureSource {
        TextureSource {
            width: ATLAS,
            height: ATLAS,
            pixels: atlas(),
            usage: TextureUsage::Color,
        }
    }

    /// The materials conifers draw with over the atlas `atlas`: the trunk's
    /// and crown's, opaque, and the fringe's, masked by the spray's
    /// coverage; both matte foliage, colored by the atlas and their
    /// vertices.
    pub fn materials(atlas: TextureHandle) -> Result<(Material, Material), AssetError> {
        let crown = Material::new([1.0; 4])?
            .with_surface(Surface {
                roughness: 0.9,
                ..Surface::MATTE
            })?
            .with_base_color_texture(TextureRef::new(atlas))?
            .with_foliage(true);
        let fringe = crown.with_alpha_mode(AlphaMode::Mask { cutoff: 0.5 })?;
        Ok((crown, fringe))
    }

    /// Its placement asset, drawn with `materials` (the crown's, then the
    /// fringe's), `mesh` naming the mesh of each level and part: its crown
    /// at every level, finest first, switching at [`Self::ERRORS`], and its
    /// fringe at the fringed levels, as its part. It stands rigid and
    /// collides with nothing: callers set its sway and collider.
    pub fn asset<E>(
        self,
        (crown, fringe): (MaterialHandle, MaterialHandle),
        mut mesh: impl FnMut(Self, usize, ConiferPart) -> Result<MeshHandle, E>,
    ) -> Result<PlacementAsset, E> {
        let mut crowns = Vec::with_capacity(Self::LEVELS);
        for level in 0..Self::LEVELS {
            crowns.push(mesh(self, level, ConiferPart::Crown)?);
        }
        let mut fringes = Vec::with_capacity(Self::FRINGED);
        for level in 0..Self::FRINGED {
            fringes.push(mesh(self, level, ConiferPart::Fringe)?);
        }
        let finest = crowns.remove(0);
        Ok(PlacementAsset {
            mesh: finest,
            material: crown,
            lods: crowns
                .into_iter()
                .zip(Self::ERRORS)
                .map(|(mesh, error)| PlacementLod { mesh, error })
                .collect(),
            parts: vec![PlacementPart {
                material: fringe,
                meshes: fringes,
            }],
            sway: 0.0,
            collider: None,
        })
    }
}

/// A branch: where it leaves the stem, the way it points out from it,
/// that way dipped by its droop, its length and its shade.
struct Branch {
    start: Vec3,
    outward: Vec3,
    direction: Vec3,
    length: f32,
    color: [f32; 3],
}

impl Branch {
    /// The point `t` of its length out (1 its tip), bending further down
    /// toward its tip.
    fn at(&self, t: f32) -> Vec3 {
        self.start + self.direction * (self.length * t) - Vec3::Y * (0.18 * self.length * t * t)
    }
}

/// A whorl: its height, its branches in order around the stem, and the
/// height its tier's underside closes at on the stem.
struct Whorl {
    y: f32,
    branches: Vec<Branch>,
    under: f32,
}

/// A conifer's shape, which every level keeps: its height, its trunk's
/// base radius and its whorls from the lowest up.
struct Shape {
    height: f32,
    base_radius: f32,
    whorls: Vec<Whorl>,
}

impl Shape {
    /// The shape of seed `seed`: whorls of six to eight branches every 0.42
    /// meters or so up the stem from about two meters, shorter toward the
    /// top, the upper ones reaching up a little and the lower ones drooping.
    fn of(seed: u32) -> Self {
        let random = |index: u32| unit(golden_hash(seed.wrapping_mul(0x9e37_79b9) ^ index));
        let height = 13.0 + 4.0 * random(1);
        let base_radius = 0.2 + 0.06 * random(2);
        let crown_base = 1.6 + 0.8 * random(3);
        let mut whorls = Vec::new();
        let mut y = crown_base;
        let mut whorl = 0_u32;
        while y < height - 0.7 {
            let below_top = height - y;
            let count = 6 + golden_hash(seed ^ whorl.wrapping_mul(0x632b_e5ab)) % 3;
            let twist = random(100 + whorl) * TAU;
            let droop = -0.25 + 0.6 * math::clamp(below_top / height, 0.0, 1.0);
            let (dip_sin, dip_cos) = math::sin_cos(droop);
            let shade = 0.6 + 0.4 * math::clamp((y - crown_base) / (0.6 * height), 0.0, 1.0);
            let branches: Vec<Branch> = (0..count)
                .map(|branch| {
                    let index = 1000 + whorl * 16 + branch;
                    let azimuth = twist
                        + TAU * (branch as f32 + 0.35 * (random(index) - 0.5)) / count as f32;
                    let (sin, cos) = math::sin_cos(azimuth);
                    let outward = Vec3::new(cos, 0.0, sin);
                    Branch {
                        start: Vec3::new(0.0, y, 0.0),
                        outward,
                        direction: outward * dip_cos - Vec3::Y * dip_sin,
                        length: (0.45 + below_top * 0.19) * (0.85 + 0.3 * random(index + 7)),
                        color: [shade * (0.94 + 0.12 * random(index + 9)), shade, shade],
                    }
                })
                .collect();
            let spacing = 0.42 * (0.85 + 0.3 * random(200 + whorl));
            // The underside closes on the stem below the whorl and below
            // its lowest tip, inside the tier under it.
            let (lowest, longest) = branches.iter().fold((y, 0.0_f32), |(low, long), branch| {
                (math::min(low, branch.at(1.0).y), math::max(long, branch.length))
            });
            let under = math::min(y - 0.5 * spacing, lowest - 0.15 * longest);
            whorls.push(Whorl { y, branches, under });
            y += spacing;
            whorl += 1;
        }
        Self {
            height,
            base_radius,
            whorls,
        }
    }
}

/// The normal crown surfaces facing `outward` from the stem take: leaning
/// out and up, the same for a tier's top and underside, so the crown lights
/// as one volume.
fn volume_normal(outward: Vec3) -> Vec3 {
    outward + Vec3::Y * 0.55
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

    /// The trunk, narrowing to a point at the top, of `sides` sides and
    /// `segments` segments, each mapping the bark once, its radius scaled
    /// by `scale`; darker toward its foot.
    fn trunk(&mut self, shape: &Shape, (sides, segments): (u32, u32), scale: f32) {
        let radius = |y: f32| {
            scale
                * (shape.base_radius * math::pow(math::max(1.0 - y / shape.height, 0.0), 0.85)
                    + 0.012)
        };
        for segment in 0..segments {
            let [low, high] = [segment, segment + 1]
                .map(|ring| -0.5 + (shape.height + 0.5) * ring as f32 / segments as f32);
            let first = self.vertices.len() as u32;
            for side in 0..=sides {
                let angle = TAU * side as f32 / sides as f32;
                let (sin, cos) = math::sin_cos(angle);
                let u = BARK[0] + (BARK[1] - BARK[0]) * side as f32 / sides as f32;
                for (y, v) in [(low, 0.99), (high, 0.01)] {
                    let shade = 0.55 + 0.45 * math::clamp(y / (0.35 * shape.height), 0.0, 1.0);
                    self.push(
                        Vec3::new(cos * radius(y), y, sin * radius(y)),
                        Vec3::new(cos, 0.15, sin),
                        [u, v],
                        [shade; 3],
                    );
                }
            }
            for side in 0..sides {
                let [a, b, c, d] = [0, 1, 2, 3].map(|corner| first + side * 2 + corner);
                self.indices.extend([a, b, c, b, d, c]);
            }
        }
    }

    /// `whorl`'s tier: a cone skirt from the stem out to its branches'
    /// tips and back in under them to the stem, sector by sector between
    /// neighboring branches, each mapping the needle mass once, darkest at
    /// the stem and under; through each branch's middle where `bent`, which
    /// lies above the straight skirt from stem to tip.
    fn tier(&mut self, whorl: &Whorl, bent: bool) {
        let count = whorl.branches.len();
        let middle_u = 0.5 * (NEEDLES[0] + NEEDLES[1]);
        let shaded = |color: [f32; 3], by: f32| color.map(|channel| channel * by);
        for index in 0..count {
            let left = &whorl.branches[index];
            let right = &whorl.branches[(index + 1) % count];
            let between = (left.outward + right.outward).normalize_or(left.outward);
            let color =
                [0_usize, 1, 2].map(|channel| 0.5 * (left.color[channel] + right.color[channel]));
            let stem = self.push(
                Vec3::new(0.0, whorl.y, 0.0),
                volume_normal(between),
                [middle_u, 0.01],
                shaded(color, 0.62),
            );
            let tip = |builder: &mut Self, branch: &Branch, u: f32, by: f32| {
                builder.push(
                    branch.at(1.0),
                    volume_normal(branch.outward),
                    [u, 0.99],
                    shaded(branch.color, by),
                )
            };
            let tip_left = tip(self, left, NEEDLES[0], 1.0);
            let tip_right = tip(self, right, NEEDLES[1], 1.0);
            if bent {
                let middle = |builder: &mut Self, branch: &Branch, u: f32| {
                    builder.push(
                        branch.at(0.5),
                        volume_normal(branch.outward),
                        [u, 0.5],
                        shaded(branch.color, 0.81),
                    )
                };
                let middle_left = middle(self, left, NEEDLES[0]);
                let middle_right = middle(self, right, NEEDLES[1]);
                self.indices.extend([
                    stem,
                    middle_right,
                    middle_left,
                    middle_left,
                    tip_right,
                    tip_left,
                    middle_left,
                    middle_right,
                    tip_right,
                ]);
            } else {
                self.indices.extend([stem, tip_right, tip_left]);
            }
            // Under: from the tips in to the stem below.
            let under = self.push(
                Vec3::new(0.0, whorl.under, 0.0),
                volume_normal(between),
                [middle_u, 0.01],
                shaded(color, 0.45),
            );
            let under_left = tip(self, left, NEEDLES[0], 0.75);
            let under_right = tip(self, right, NEEDLES[1], 0.75);
            self.indices.extend([under, under_left, under_right]);
        }
    }

    /// The leader's shoot: a narrow cone of needle mass topping the stem.
    fn leader(&mut self, height: f32) {
        let (base, top) = (height - 1.0, height + 0.15);
        let first = self.vertices.len() as u32;
        for side in 0..=3_u32 {
            let (sin, cos) = math::sin_cos(TAU * side as f32 / 3.0);
            let u = NEEDLES[0] + (NEEDLES[1] - NEEDLES[0]) * side as f32 / 3.0;
            self.push(
                Vec3::new(cos * 0.07, base, sin * 0.07),
                Vec3::new(cos, 0.3, sin),
                [u, 0.99],
                [0.9; 3],
            );
        }
        let tip = self.push(
            Vec3::new(0.0, top, 0.0),
            Vec3::Y,
            [0.5 * (NEEDLES[0] + NEEDLES[1]), 0.01],
            [1.0; 3],
        );
        for side in 0..3 {
            self.indices.extend([first + side + 1, first + side, tip]);
        }
    }

    /// `fringe`'s cards: along the outer reach of its whorls' branches,
    /// flat across them and, where it hangs, drooping under them, and the
    /// crossed cards topping the leader.
    fn fringe(&mut self, shape: &Shape, fringe: &Fringe) {
        let mut spine = Vec::with_capacity(fringe.bends + 1);
        for (number, whorl) in shape.whorls.iter().enumerate() {
            for (index, branch) in whorl.branches.iter().enumerate() {
                if !(index + number).is_multiple_of(fringe.every as usize) {
                    continue;
                }
                spine.clear();
                spine.extend((0..=fringe.bends).map(|step| {
                    let t = fringe.from + (REACH - fringe.from) * step as f32 / fringe.bends as f32;
                    (branch.at(t), t)
                }));
                let across = Vec3::new(-branch.outward.z, 0.0, branch.outward.x);
                let normal = volume_normal(branch.outward);
                self.card(
                    &spine,
                    across * (0.32 * branch.length),
                    normal,
                    branch.color,
                );
                if fringe.hanging {
                    let hanging = branch.direction.cross(across).normalize_or(Vec3::NEG_Y);
                    self.card(
                        &spine,
                        hanging * (0.22 * branch.length),
                        normal,
                        branch.color,
                    );
                }
            }
        }
        for card in 0..fringe.leader {
            let (sin, cos) = math::sin_cos(card as f32 * FRAC_PI_2);
            self.card(
                &[
                    (Vec3::new(0.0, shape.height - 1.0, 0.0), 0.0),
                    (Vec3::new(0.0, shape.height + 0.15, 0.0), 1.0),
                ],
                Vec3::new(cos, 0.0, sin) * 0.25,
                Vec3::Y,
                [1.0; 3],
            );
        }
    }

    /// A card along `spine` (each point with how far out its branch it
    /// lies, the card's base first), reaching `half` to each side, mapping
    /// the spray from its base to its tip, both faces lit as `normal`;
    /// darker nearer the stem.
    fn card(&mut self, spine: &[(Vec3, f32)], half: Vec3, normal: Vec3, color: [f32; 3]) {
        let first = self.vertices.len() as u32;
        let last = spine.len() - 1;
        for (step, &(center, along)) in spine.iter().enumerate() {
            let t = step as f32 / last as f32;
            let v = SPRAY_SPAN[0] + (SPRAY_SPAN[1] - SPRAY_SPAN[0]) * t;
            let inner = 0.62 + 0.38 * math::min(along, 1.0);
            let shade = color.map(|channel| channel * inner);
            for side in [-1.0_f32, 1.0] {
                self.push(
                    center + half * side,
                    normal,
                    [SPRAY[0] + SPRAY[1] * side, v],
                    shade,
                );
            }
        }
        for step in 0..last as u32 {
            let [a, b, c, d] = [0, 1, 2, 3].map(|corner| first + step * 2 + corner);
            // Front and back faces, both lit by the shared normal.
            self.indices.extend([a, b, c, b, d, c, a, c, b, b, c, d]);
        }
    }
}

/// The atlas's RGBA8 texels, sRGB color with alpha coverage: bark furrowed
/// along its height in its first quarter, needle mass in its second, and a
/// needle spray on its right half, the spray's uncovered texels keeping the
/// needles' mean color, so filtering never darkens their edges.
fn atlas() -> Vec<u8> {
    let size = ATLAS as usize;
    let (quarter, half) = (size / 4, size / 2);
    let mut color = vec![[0.06, 0.13, 0.06]; size * size];
    let mut alpha = vec![0.0_f32; size * size];
    // Bark: furrows along the height, broken by knots.
    for y in 0..size {
        for x in 0..quarter {
            let point = Vec2::new(x as f32 / 9.0, y as f32 / 70.0);
            let furrow = 1.0 - (noise(point) * 2.0 - 1.0).abs();
            let grain = noise(Vec2::new(x as f32 / 3.0, y as f32 / 14.0));
            let tone = 0.55 + 0.35 * furrow * furrow + 0.15 * grain;
            color[y * size + x] = [0.15 * tone, 0.115 * tone, 0.09 * tone];
            alpha[y * size + x] = 1.0;
        }
    }
    let mut canvas = Canvas {
        color: &mut color,
        alpha: &mut alpha,
        size,
    };
    canvas.needle_mass([quarter, half]);
    canvas.spray([half, size]);
    let mut texels = Vec::with_capacity(size * size * 4);
    for (rgb, coverage) in color.iter().zip(&alpha) {
        texels.extend(rgb.map(srgb));
        texels.push((coverage * 255.0).round() as u8);
    }
    texels
}

/// The atlas being drawn: linear colors and coverage, `size` texels a side.
struct Canvas<'a> {
    color: &'a mut [[f32; 3]],
    alpha: &'a mut [f32],
    size: usize,
}

impl Canvas<'_> {
    /// Needle mass over the atlas's `columns`: needles' shaded green
    /// beneath, then needles laid over it in rows down its height, each
    /// pointing down and out at its own angle, colored as the spray's are
    /// and greener toward the bottom, a tier's rim; opaque throughout.
    fn needle_mass(&mut self, columns: [usize; 2]) {
        for y in 0..self.size {
            for x in columns[0]..columns[1] {
                self.color[y * self.size + x] = [0.035, 0.08, 0.035];
                self.alpha[y * self.size + x] = 1.0;
            }
        }
        let spacing = 3.0;
        let rows = (self.size as f32 / spacing) as u32;
        let across = ((columns[1] - columns[0]) as f32 / spacing) as u32;
        let mut needles = 0_u32;
        for row in 0..rows {
            for column in 0..across {
                needles += 1;
                let draw = |salt: u32| unit(golden_hash(needles.wrapping_mul(0x2c1b_3c6d) ^ salt));
                let root = Vec2::new(
                    columns[0] as f32 + (column as f32 + draw(1)) * spacing,
                    (row as f32 + draw(2)) * spacing,
                );
                let (sin, cos) = math::sin_cos((draw(3) - 0.5) * 1.4);
                let reach = 12.0 + 10.0 * draw(4);
                let down = root.y / self.size as f32;
                let shade = 0.8 + 0.35 * draw(5);
                let tint = [0.055 * shade, (0.12 + 0.04 * down) * shade, 0.055 * shade];
                self.capsule(root, root + Vec2::new(sin, cos) * reach, 0.95, tint, columns);
            }
        }
    }

    /// A needle spray over the atlas's `columns`: its main twig rising from
    /// the bottom edge with side shoots alternating up it, longest at the
    /// base, every twig fringed with needles.
    fn spray(&mut self, columns: [usize; 2]) {
        let width = (columns[1] - columns[0]) as f32;
        let center = columns[0] as f32 + 0.5 * width;
        let (base, tip) = (self.size as f32 - 6.0, 10.0);
        let twig = [0.24, 0.17, 0.11];
        let mut shoots = Vec::new();
        let mut along = 0.06;
        let mut side = 1.0;
        while along < 0.9 {
            let start = Vec2::new(center, base + (tip - base) * along);
            let spread = unit(golden_hash(shoots.len() as u32 + 5));
            let reach = 0.42 * width * (1.0 - along) * (0.8 + 0.4 * spread);
            let end = start + Vec2::new(side * 0.72, -0.7).normalize() * reach;
            shoots.push((start, end));
            along += 0.058;
            side = -side;
        }
        let mut needles = 0_u32;
        let mut fringe = |canvas: &mut Self, from: Vec2, to: Vec2, length: f32| {
            let span = to - from;
            let steps = (span.length() / 1.7) as u32;
            let direction = span.normalize_or(Vec2::Y);
            for step in 0..steps {
                let t = step as f32 / steps.max(1) as f32;
                let root = from + span * t;
                for side in [-1.0_f32, 1.0] {
                    needles += 1;
                    let jitter = unit(golden_hash(needles));
                    let (sin, cos) = math::sin_cos(side * (0.85 + 0.35 * jitter));
                    let pointing = Vec2::new(
                        direction.x * cos - direction.y * sin,
                        direction.x * sin + direction.y * cos,
                    );
                    let spread = unit(golden_hash(needles ^ 0x5bd1));
                    let reach = length * (1.0 - 0.45 * t) * (0.8 + 0.4 * spread);
                    let shade = 0.8 + 0.35 * unit(golden_hash(needles ^ 0x27d4));
                    let tint = [0.055 * shade, (0.12 + 0.04 * t) * shade, 0.055 * shade];
                    canvas.capsule(root, root + pointing * reach, 0.95, tint, columns);
                }
            }
        };
        for &(start, end) in &shoots {
            fringe(self, start, end, 21.0);
        }
        fringe(self, Vec2::new(center, base), Vec2::new(center, tip), 24.0);
        for &(start, end) in &shoots {
            self.capsule(start, end, 1.1, twig, columns);
        }
        self.capsule(
            Vec2::new(center, base),
            Vec2::new(center, tip),
            2.2,
            twig,
            columns,
        );
    }

    /// A stroke from `a` to `b`, `radius` texels thick with antialiased
    /// edges, of linear color `tint`, laid over what is drawn within the
    /// atlas's `columns`.
    fn capsule(&mut self, a: Vec2, b: Vec2, radius: f32, tint: [f32; 3], columns: [usize; 2]) {
        let min = a.min(b) - Vec2::splat(radius + 1.0);
        let max = a.max(b) + Vec2::splat(radius + 1.0);
        let x0 = math::max(min.x.floor(), columns[0] as f32) as usize;
        let x1 = math::min(max.x.ceil(), columns[1] as f32 - 1.0) as usize;
        let y0 = math::max(min.y.floor(), 0.0) as usize;
        let y1 = math::min(max.y.ceil(), self.size as f32 - 1.0) as usize;
        let span = b - a;
        let length2 = math::max(span.length_squared(), 1.0e-6);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let point = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let t = math::clamp((point - a).dot(span) / length2, 0.0, 1.0);
                let distance = (point - (a + span * t)).length();
                let cover = math::clamp(radius + 0.5 - distance, 0.0, 1.0);
                if cover <= 0.0 {
                    continue;
                }
                let at = y * self.size + x;
                let old = self.alpha[at];
                let alpha = cover + old * (1.0 - cover);
                let laid = |channel: usize| {
                    (tint[channel] * cover + self.color[at][channel] * old * (1.0 - cover)) / alpha
                };
                self.color[at] = [laid(0), laid(1), laid(2)];
                self.alpha[at] = alpha;
            }
        }
    }
}

/// `value` with its bits mixed after a multiply by the golden ratio.
fn golden_hash(value: u32) -> u32 {
    let mut h = value.wrapping_mul(0x9e37_79b1);
    h = (h ^ (h >> 16)).wrapping_mul(0x7feb_352d);
    h = (h ^ (h >> 15)).wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

/// A draw in [0, 1) from `value`'s top 24 bits.
fn unit(value: u32) -> f32 {
    (value >> 8) as f32 / 16_777_216.0
}

/// Linear `value` as an sRGB byte.
fn srgb(value: f32) -> u8 {
    let value = math::clamp(value, 0.0, 1.0);
    let encoded = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * math::pow(value, 1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round() as u8
}

/// Value noise from 0 to 1 at `point`, smooth between lattice points.
fn noise(point: Vec2) -> f32 {
    let cell = point.floor();
    let f = point - cell;
    let u = f * f * (Vec2::splat(3.0) - 2.0 * f);
    let corner = |dx: i32, dy: i32| {
        let key =
            (cell.x as i32 + dx) as u32 ^ ((cell.y as i32 + dy) as u32).wrapping_mul(0x8da6_b343);
        unit(golden_hash(key))
    };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    lerp(
        lerp(corner(0, 0), corner(1, 0), u.x),
        lerp(corner(0, 1), corner(1, 1), u.x),
        u.y,
    )
}
