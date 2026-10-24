//! Conifers for the highlands, the scene's own content: a tapering trunk
//! and whorls of drooping branches, each a flat card of needles across it
//! and a hanging card through it, over one atlas whose left half is bark
//! and right half a needle spray, drawn with one masked material shaded as
//! foliage. A card's normal is mostly its branch's, leaning out from the
//! trunk so the crown lights as one volume, and partly its face's own:
//! cards carry each face with vertices of its own, and vertex colors darken
//! the crown toward its trunk and its lower whorls. Coarser levels keep
//! every card's size and drop whole branches in the order their whorls
//! rank them: the lower whorls thin first, then about every other branch
//! of each whorl goes, and the crown's top and each whorl's longest
//! branches, the tips of its outline, stay longest. Every level's cards are
//! cut to the needles' alpha ([`trim_cards`]). Shadow maps draw each tree
//! as a closed solid crown within its cards' outline: its trunk and
//! drooping cone tiers.

use std::f32::consts::{FRAC_PI_2, TAU};
use std::sync::OnceLock;

use engine::prelude::*;
use engine::terrain::{PlacementAsset, PlacementLod};

use crate::kit::random::{golden_hash, unit};
use crate::kit::records;

/// Spruce `seed` as a placement asset drawn with `material` ([`material`]):
/// its levels, finest first, switching at [`CONIFER_ERRORS`], and its
/// shadow proxy, each a record ([`records::mesh`]) whose handle is reserved
/// until the record is here. It stands rigid and collides with nothing:
/// the scene sets its sway and collider.
pub(crate) fn spruce(
    world: &mut World,
    seed: u32,
    material: MaterialHandle,
) -> engine::Result<PlacementAsset> {
    let mut levels = (0..DETAILS.len())
        .map(|detail| conifer_record(world, seed, detail))
        .collect::<engine::Result<Vec<_>>>()?;
    let shadow = shadow_record(world, seed)?;
    let mesh = levels.remove(0);
    Ok(PlacementAsset {
        mesh,
        material,
        lods: levels
            .into_iter()
            .zip(CONIFER_ERRORS)
            .map(|(mesh, error)| PlacementLod { mesh, error })
            .collect(),
        parts: Vec::new(),
        sway: 0.0,
        collider: None,
        shadow: Some(shadow),
    })
}

/// Conifer `seed` at `detail` as a record ([`records::mesh`]): its handle,
/// reserved until the record is here.
fn conifer_record(world: &mut World, seed: u32, detail: usize) -> engine::Result<MeshHandle> {
    let mut identity = seed.to_le_bytes().to_vec();
    identity.extend_from_slice(&(detail as u32).to_le_bytes());
    records::mesh(world, "highlands_conifer", &identity, move || {
        conifer(seed, detail)
    })
}

/// Conifer `seed`'s shadow proxy as a record ([`records::mesh`]): its
/// handle, reserved until the record is here.
fn shadow_record(world: &mut World, seed: u32) -> engine::Result<MeshHandle> {
    records::mesh(
        world,
        "highlands_conifer_shadow",
        &seed.to_le_bytes(),
        move || shadow_proxy(seed),
    )
}

/// The bark and needle atlas as a texture record ([`records::textures`]),
/// cooked as every generated texture is: its handle, which materials name
/// at once.
pub(crate) fn atlas_record(world: &mut World) -> engine::Result<TextureHandle> {
    let handles = world.resource_scope(|world, mut stored: Mut<Records>| {
        let mut assets = world.resource_mut::<AssetLibrary>();
        records::textures(
            (&mut stored, &mut assets),
            "highlands/bark_atlas".into(),
            &[("atlas", TextureUsage::Color)],
            || {
                Ok(vec![TextureSource {
                    width: ATLAS,
                    height: ATLAS,
                    pixels: texels().to_vec(),
                    usage: TextureUsage::Color,
                }])
            },
        )
    })?;
    Ok(handles[0])
}

/// The material conifers draw with over their atlas `atlas`: matte,
/// colored by the atlas and their vertices, cut out by the atlas's alpha
/// at [`CUTOFF`] and shaded as foliage.
pub(crate) fn material(atlas: TextureHandle) -> Result<Material, AssetError> {
    Material::new([1.0; 4])?
        .with_surface(Surface {
            roughness: 0.9,
            ..Surface::MATTE
        })?
        .with_base_color_texture(TextureRef::new(atlas))?
        .with_alpha_mode(AlphaMode::Mask { cutoff: CUTOFF })
        .map(|material| material.with_foliage(true))
}

/// The atlas's texels ([`atlas`]), drawn once: what its texture record
/// cooks, and what every level's cards are cut to.
fn texels() -> &'static [u8] {
    static TEXELS: OnceLock<Vec<u8>> = OnceLock::new();
    TEXELS.get_or_init(atlas)
}

/// The atlas's side in texels.
const ATLAS: u32 = 512;

/// The alpha below which the atlas's texels are cut out: the material's
/// mask, and the needles every level's cards are cut to.
const CUTOFF: f32 = 0.5;

/// Bark texture coordinates span this much of the atlas's width: its left
/// half less a margin, so filtering never reaches the needles.
const BARK: [f32; 2] = [0.01, 0.49];
/// Texture coordinates amid the bark, which covers the atlas there
/// throughout: those of every vertex of a shadow proxy.
const BARK_MIDDLE: [f32; 2] = [0.5 * (BARK[0] + BARK[1]), 0.5];
/// The needle spray's horizontal center and half width in the atlas, and
/// its vertical span (base at the bottom).
const SPRAY: [f32; 2] = [0.75, 0.235];
const SPRAY_SPAN: [f32; 2] = [0.99, 0.01];

/// How much of a card's normal is its branch's, leaning out from the trunk
/// and [`LEAN`] up, rather than its face's own: the crown lights as one
/// volume, each card still by the way it faces.
const CROWN_NORMAL: f32 = 0.7;
const LEAN: f32 = 0.55;

/// A card segment's corners as its triangles, its first two vertices
/// across its base and the next two across its tip: its front's, then its
/// back's.
const FRONT: [u32; 6] = [0, 1, 2, 1, 3, 2];
const BACK: [u32; 6] = [0, 2, 1, 1, 2, 3];

/// The meters between whorls, each spacing varying about it.
const SPACING: f32 = 0.42;

/// A shadow proxy's tiers: the whorls each spans, the sides of their rims,
/// and how far out a tier's rim lies along its lowest whorl's branches, as
/// a share of the length they vary about: within their needles.
const TIER: usize = 2;
const RIM_SIDES: u32 = 8;
const RIM: f32 = 0.8;
/// A shadow proxy's trunk's radius at its foot against the trunk's: its
/// corners within the faces of the finer levels' trunks.
const SPIKE: f32 = 0.7;

/// How finely a level is built: its trunk's sides and segments (each
/// mapping the bark once), its cards' segments, and the share of each
/// whorl's branches it keeps, at the crown's foot and at its top (between
/// them by the whorl's height), two at least. It drops the rest whole, the
/// last its whorl ranks ([`rank`]), and every card keeps its size: coarser
/// levels thin the lower whorls first, whose long branches overlap the
/// most, then keep about every other branch of each whorl, the crown's top
/// and each whorl's longest branches staying longest.
struct Detail {
    sides: u32,
    segments: u32,
    bends: usize,
    keep: [f32; 2],
}

/// The full tree, then each coarser level.
const DETAILS: [Detail; 4] = [
    Detail {
        sides: 7,
        segments: 8,
        bends: 2,
        keep: [1.0, 1.0],
    },
    Detail {
        sides: 5,
        segments: 4,
        bends: 2,
        keep: [0.55, 1.0],
    },
    Detail {
        sides: 4,
        segments: 2,
        bends: 1,
        keep: [0.45, 0.6],
    },
    Detail {
        sides: 3,
        segments: 1,
        bends: 1,
        keep: [0.3, 0.5],
    },
];

/// The errors, in meters, the coarser levels show: chosen so a tree drops
/// branches where the gaps they leave blur into its crown rather than by
/// geometric deviation, which cards make meaningless.
const CONIFER_ERRORS: [f32; 3] = [0.04, 0.09, 0.16];

/// Conifer `seed` built at `detail` (0 the full tree, up to 3): 13 to 17
/// meters tall before placement scales it, its base half a meter into the
/// ground, its cards cut to the needles' alpha.
fn conifer(seed: u32, detail: usize) -> engine::Result<Mesh> {
    let build = &DETAILS[detail];
    let shape = Shape::of(seed);
    let mut builder = Builder::default();
    builder.trunk(&shape, (build.sides, build.segments));
    // Each whorl's best ranked branches, as many as its share keeps.
    for whorl in &shape.whorls {
        let count = whorl.branches.len();
        let share = build.keep[0] + (build.keep[1] - build.keep[0]) * whorl.rise;
        let kept = ((count as f32 * share).round() as usize).clamp(2, count);
        for branch in whorl.branches.iter().filter(|branch| branch.rank < kept) {
            builder.branch(branch, build.bends);
        }
    }
    builder.leader(shape.height);
    let atlas = AlphaImage {
        width: ATLAS,
        height: ATLAS,
        rgba: texels(),
    };
    builder
        .mesh()
        .and_then(|mesh| trim_cards(&mesh, atlas, CUTOFF))
        .map_err(|error| {
            engine::Error::from(error).context(format!("conifer {seed} at detail {detail}"))
        })
}

/// Conifer `seed`'s shadow proxy, which shadow maps draw in place of its
/// levels: a closed solid crown within its cards' outline, a few hundred
/// triangles. Its trunk is a four-sided spike from its foot below the
/// ground to the top; its crown a drooping cone tier for every [`TIER`]
/// whorls, from the stem at the tier's highest whorl (the top, for the
/// highest tier) down and out to a rim along its lowest whorl's branches,
/// and back in under them. Its vertices carry the streams its levels' do,
/// white and amid the bark.
fn shadow_proxy(seed: u32) -> engine::Result<Mesh> {
    let shape = Shape::of(seed);
    let mut builder = Builder::default();
    let foot = SPIKE * shape.radius(-0.5);
    for side in 0..4 {
        let (sin, cos) = math::sin_cos(FRAC_PI_2 * side as f32);
        let corner = Vec3::new(cos * foot, -0.5, sin * foot);
        builder.push(corner, Vec3::new(cos, 0.0, sin), BARK_MIDDLE, [1.0; 3]);
    }
    let tip = Vec3::new(0.0, shape.height, 0.0);
    let peak = builder.push(tip, Vec3::Y, BARK_MIDDLE, [1.0; 3]);
    for side in 0..4 {
        builder.indices.extend([side, peak, (side + 1) % 4]);
    }
    let tiers = shape.whorls.len().div_ceil(TIER);
    for (number, whorls) in shape.whorls.chunks(TIER).enumerate() {
        let top = if number + 1 == tiers {
            shape.height
        } else {
            whorls[whorls.len() - 1].y
        };
        builder.tier(&whorls[0], top);
    }
    builder.mesh().map_err(|error| {
        engine::Error::from(error).context(format!("conifer {seed}'s shadow proxy"))
    })
}

/// A conifer's shape, which every level and its shadow proxy keep: its
/// height, its trunk's base radius and its whorls from the lowest up.
struct Shape {
    height: f32,
    base_radius: f32,
    whorls: Vec<Whorl>,
}

/// A whorl: its height on the stem, the length its branches vary about,
/// the sine and cosine of their droop, the turn about the stem they start
/// from, how far up the crown it stands (0 at its foot, toward 1 at its
/// top), and its branches in order round the stem.
struct Whorl {
    y: f32,
    length: f32,
    dip: (f32, f32),
    twist: f32,
    rise: f32,
    branches: Vec<Branch>,
}

/// A branch: where it leaves the stem, the way it points out from it, that
/// way dipped by its whorl's droop, its length and its shade; where it
/// stands round the stem, in spacings between its whorl's branches; the
/// draw that set its length (toward 1 the longest); and its rank, how many
/// of its whorl's branches coarser levels keep before it.
struct Branch {
    start: Vec3,
    outward: Vec3,
    direction: Vec3,
    length: f32,
    color: [f32; 3],
    slot: f32,
    reach: f32,
    rank: usize,
}

impl Branch {
    /// The point `t` of its length out (1 its tip).
    fn at(&self, t: f32) -> Vec3 {
        point_along(self.start, self.direction, self.length, t)
    }
}

/// The point `t` of the way along a branch from `start` pointing
/// `direction`, `length` meters long, bending further down toward its tip.
fn point_along(start: Vec3, direction: Vec3, length: f32, t: f32) -> Vec3 {
    start + direction * (length * t) - Vec3::Y * (0.18 * length * t * t)
}

impl Shape {
    /// The shape of seed `seed`: 13 to 17 meters tall, with whorls of six
    /// to eight branches every [`SPACING`] meters or so up the stem from
    /// about two meters, shorter toward the top, the upper ones reaching up
    /// a little and the lower ones drooping and darker, each whorl's
    /// branches ranked ([`rank`]).
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
            let length = 0.45 + below_top * 0.19;
            let shade = 0.5 + 0.5 * math::clamp((y - crown_base) / (0.6 * height), 0.0, 1.0);
            let mut branches: Vec<Branch> = (0..count)
                .map(|branch| {
                    let index = 1000 + whorl * 16 + branch;
                    let slot = branch as f32 + 0.35 * (random(index) - 0.5);
                    let (sin, cos) = math::sin_cos(twist + TAU * slot / count as f32);
                    let outward = Vec3::new(cos, 0.0, sin);
                    let reach = random(index + 7);
                    Branch {
                        start: Vec3::new(0.0, y, 0.0),
                        outward,
                        direction: outward * dip_cos - Vec3::Y * dip_sin,
                        length: length * (0.85 + 0.3 * reach),
                        color: [shade * (0.94 + 0.12 * random(index + 9)), shade, shade],
                        slot,
                        reach,
                        rank: 0,
                    }
                })
                .collect();
            rank(&mut branches);
            whorls.push(Whorl {
                y,
                length,
                dip: (dip_sin, dip_cos),
                twist,
                rise: math::clamp((y - crown_base) / (height - 0.7 - crown_base), 0.0, 1.0),
                branches,
            });
            y += SPACING * (0.85 + 0.3 * random(200 + whorl));
            whorl += 1;
        }
        Self {
            height,
            base_radius,
            whorls,
        }
    }

    /// Its trunk's radius `y` meters up, narrowing to a point at the top.
    fn radius(&self, y: f32) -> f32 {
        self.base_radius * math::pow(math::max(1.0 - y / self.height, 0.0), 0.85) + 0.012
    }
}

/// Ranks one whorl's `branches` by how long coarser levels keep them: its
/// longest first, then each time the one farthest round the stem from
/// those ranked, the longer of two about as far. A level keeping any number
/// spreads them round the stem, and drops the shorter, inner ones crowded
/// between others first.
fn rank(branches: &mut [Branch]) {
    let count = branches.len() as f32;
    let mut ranked: Vec<usize> = Vec::with_capacity(branches.len());
    while ranked.len() < branches.len() {
        let mut best: Option<(usize, f32)> = None;
        for (index, branch) in branches.iter().enumerate() {
            if ranked.contains(&index) {
                continue;
            }
            // Its room: the spacings round the stem to the nearest ranked.
            let room = ranked.iter().fold(count, |room, &other| {
                let gap = (branch.slot - branches[other].slot).abs();
                math::min(room, math::min(gap, count - gap))
            });
            let score = room + 0.5 * branch.reach;
            if best.is_none_or(|(_, top)| score > top) {
                best = Some((index, score));
            }
        }
        let Some((index, _)) = best else { break };
        branches[index].rank = ranked.len();
        ranked.push(index);
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

    /// The trunk, narrowing to a point at the top, of `sides` sides and
    /// `segments` segments, each mapping the bark once; darker toward its
    /// foot.
    fn trunk(&mut self, shape: &Shape, (sides, segments): (u32, u32)) {
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
                    let radius = shape.radius(y);
                    self.push(
                        Vec3::new(cos * radius, y, sin * radius),
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

    /// `branch`'s cards along it in `bends` segments, bending further down
    /// toward its tip: a flat one across it and a hanging one through it,
    /// lit as its normal leaning out from the stem.
    fn branch(&mut self, branch: &Branch, bends: usize) {
        let spine: Vec<Vec3> = (0..=bends)
            .map(|step| branch.at(step as f32 / bends as f32))
            .collect();
        let across = Vec3::new(-branch.outward.z, 0.0, branch.outward.x);
        let hanging = branch.direction.cross(across).normalize_or(Vec3::Y);
        let crown = (branch.outward + Vec3::Y * LEAN).normalize_or(Vec3::Y);
        self.card(&spine, across * (0.45 * branch.length), crown, branch.color);
        self.card(&spine, hanging * (0.3 * branch.length), crown, branch.color);
    }

    /// The leader: two crossed upright cards topping the stem, lit as the
    /// crown's top.
    fn leader(&mut self, height: f32) {
        for turn in [0.0, FRAC_PI_2] {
            let (sin, cos) = math::sin_cos(turn);
            self.card(
                &[
                    Vec3::new(0.0, height - 1.0, 0.0),
                    Vec3::new(0.0, height + 0.15, 0.0),
                ],
                Vec3::new(cos, 0.0, sin) * 0.25,
                Vec3::Y,
                [1.0; 3],
            );
        }
    }

    /// A card along `spine` (its base first), reaching `half` to each side
    /// and mapping the spray from its base to its tip; its base, nearest
    /// the stem, is darkest. Each face has vertices of its own, lit as
    /// [`CROWN_NORMAL`] of `crown` (a unit normal) and the rest of the
    /// face's own normal.
    fn card(&mut self, spine: &[Vec3], half: Vec3, crown: Vec3, color: [f32; 3]) {
        let last = spine.len() - 1;
        for (facing, corners) in [(1.0_f32, FRONT), (-1.0, BACK)] {
            let first = self.vertices.len() as u32;
            for (step, &center) in spine.iter().enumerate() {
                let t = step as f32 / last as f32;
                let v = SPRAY_SPAN[0] + (SPRAY_SPAN[1] - SPRAY_SPAN[0]) * t;
                let inner = 0.5 + 0.5 * t;
                let shade = color.map(|channel| channel * inner);
                // The face's own normal, across the spine's way here.
                let way = spine[(step + 1).min(last)] - spine[step.saturating_sub(1)];
                let face = half.cross(way).normalize_or(crown) * facing;
                let normal = crown * CROWN_NORMAL + face * (1.0 - CROWN_NORMAL);
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
                let base = first + step * 2;
                self.indices.extend(corners.map(|corner| base + corner));
            }
        }
    }

    /// A shadow proxy's tier, closed: from the stem at `top` down and out
    /// to a rim [`RIM`] of the way along the branches of `whorl`, its
    /// lowest, and back in under them to the stem a whorl's spacing below
    /// `whorl`.
    fn tier(&mut self, whorl: &Whorl, top: f32) {
        let apex = self.push(Vec3::new(0.0, top, 0.0), Vec3::Y, BARK_MIDDLE, [1.0; 3]);
        let below = Vec3::new(0.0, whorl.y - SPACING, 0.0);
        let under = self.push(below, Vec3::NEG_Y, BARK_MIDDLE, [1.0; 3]);
        let start = Vec3::new(0.0, whorl.y, 0.0);
        let (dip_sin, dip_cos) = whorl.dip;
        let first = self.vertices.len() as u32;
        for side in 0..RIM_SIDES {
            let turn = whorl.twist + TAU * side as f32 / RIM_SIDES as f32;
            let (sin, cos) = math::sin_cos(turn);
            let outward = Vec3::new(cos, 0.0, sin);
            let direction = outward * dip_cos - Vec3::Y * dip_sin;
            let rim = point_along(start, direction, whorl.length, RIM);
            self.push(rim, outward, BARK_MIDDLE, [1.0; 3]);
        }
        for side in 0..RIM_SIDES {
            let [left, right] = [side, (side + 1) % RIM_SIDES].map(|corner| first + corner);
            self.indices.extend([apex, right, left, under, left, right]);
        }
    }
}

/// The atlas's RGBA8 texels, sRGB color with alpha coverage: bark furrowed
/// along its height on the left half, a needle spray on the right, its
/// main twig rising from the bottom edge with side shoots, every twig
/// fringed with needles. Uncovered texels keep the needles' mean color, so
/// filtering never darkens their edges.
fn atlas() -> Vec<u8> {
    let size = ATLAS as usize;
    let half = size / 2;
    let needle = [0.06, 0.13, 0.06];
    let mut color = vec![needle; size * size];
    let mut alpha = vec![0.0_f32; size * size];
    // Bark: furrows along the height, broken by knots.
    for y in 0..size {
        for x in 0..half {
            let point = Vec2::new(x as f32 / 9.0, y as f32 / 70.0);
            let furrow = 1.0 - (noise(point) * 2.0 - 1.0).abs();
            let grain = noise(Vec2::new(x as f32 / 3.0, y as f32 / 14.0));
            let tone = 0.55 + 0.35 * furrow * furrow + 0.15 * grain;
            color[y * size + x] = [0.15 * tone, 0.115 * tone, 0.09 * tone];
            alpha[y * size + x] = 1.0;
        }
    }
    let mut spray = Spray {
        color: &mut color,
        alpha: &mut alpha,
        size,
    };
    let center = (half + half / 2) as f32;
    let (base, tip) = (size as f32 - 6.0, 10.0);
    let twig = [0.24, 0.17, 0.11];
    // Side shoots alternate up the main twig, longest at the base.
    let mut shoots = Vec::new();
    let mut along = 0.06;
    let mut side = 1.0;
    while along < 0.9 {
        let start = Vec2::new(center, base + (tip - base) * along);
        let reach = 0.42
            * (half as f32)
            * (1.0 - along)
            * (0.8 + 0.4 * unit(golden_hash(shoots.len() as u32 + 5)));
        let end = start + Vec2::new(side * 0.72, -0.7).normalize() * reach;
        shoots.push((start, end));
        along += 0.058;
        side = -side;
    }
    let mut needles = 0_u32;
    let mut fringe = |spray: &mut Spray<'_>, from: Vec2, to: Vec2, length: f32| {
        let span = to - from;
        let steps = (span.length() / 1.7) as u32;
        let direction = span.normalize_or(Vec2::Y);
        for step in 0..steps {
            let t = step as f32 / steps.max(1) as f32;
            let root = from + span * t;
            for side in [-1.0_f32, 1.0] {
                needles += 1;
                let jitter = unit(golden_hash(needles));
                let angle = side * (0.85 + 0.35 * jitter);
                let (sin, cos) = math::sin_cos(angle);
                let pointing = Vec2::new(
                    direction.x * cos - direction.y * sin,
                    direction.x * sin + direction.y * cos,
                );
                let reach =
                    length * (1.0 - 0.45 * t) * (0.8 + 0.4 * unit(golden_hash(needles ^ 0x5bd1)));
                let shade = 0.8 + 0.35 * unit(golden_hash(needles ^ 0x27d4));
                let tint = [0.055 * shade, (0.12 + 0.04 * t) * shade, 0.055 * shade];
                spray.capsule(root, root + pointing * reach, 0.95, tint);
            }
        }
    };
    for &(start, end) in &shoots {
        fringe(&mut spray, start, end, 21.0);
    }
    fringe(
        &mut spray,
        Vec2::new(center, base),
        Vec2::new(center, tip),
        24.0,
    );
    for &(start, end) in &shoots {
        spray.capsule(start, end, 1.1, twig);
    }
    spray.capsule(Vec2::new(center, base), Vec2::new(center, tip), 2.2, twig);
    let mut texels = Vec::with_capacity(size * size * 4);
    for (rgb, coverage) in color.iter().zip(&alpha) {
        texels.extend(rgb.map(srgb));
        texels.push((coverage * 255.0).round() as u8);
    }
    texels
}

/// The needle spray being drawn: linear colors and coverage of the atlas.
struct Spray<'a> {
    color: &'a mut [[f32; 3]],
    alpha: &'a mut [f32],
    size: usize,
}

impl Spray<'_> {
    /// A stroke from `a` to `b`, `radius` texels thick with antialiased
    /// edges, of linear color `tint`, laid over what is drawn.
    fn capsule(&mut self, a: Vec2, b: Vec2, radius: f32, tint: [f32; 3]) {
        let min = a.min(b) - Vec2::splat(radius + 1.0);
        let max = a.max(b) + Vec2::splat(radius + 1.0);
        let half = (self.size / 2) as f32;
        let x0 = math::max(min.x.floor(), half) as usize;
        let x1 = math::min(max.x.ceil(), self.size as f32 - 1.0) as usize;
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
