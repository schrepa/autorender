//! Deadfall for the highlands, the scene's own content: fallen spruce logs
//! and the stumps of spruces that broke or were cut, in the spruces' bark
//! ([`super::trees`]'s atlas). A log lies along its x axis on the ground,
//! its butt toward -x, tapering to where its stem broke, out of round and a
//! little bowed as rot settles it, the stubs of its branches standing out
//! of its top and sides; a stump is a trunk's foot flaring into its roots
//! under the ground, sawn level or broken off jagged. Each maps the bark
//! once around it and once every [`BARK_REPEAT`] meters along it; vertex
//! colors darken their undersides and their roots, where they meet the
//! damp ground. Their coarser levels are their meshes' own, simplified
//! offline.

use std::f32::consts::{PI, TAU};

use engine::prelude::*;
use engine::terrain::{PlacedCollider, PlacedShape, PlacementAsset};

use super::trees::{BARK, BARK_MIDDLE};
use crate::kit::random::{golden_hash, noise, unit};
use crate::kit::records;

/// Log `seed` as a placement asset drawn with `material`: its mesh, a record
/// ([`records::mesh`]) whose handle is reserved until the record is here,
/// colliding as a box about its middle girth; rigid.
pub(crate) fn log(
    world: &mut World,
    seed: u32,
    material: MaterialHandle,
) -> engine::Result<PlacementAsset> {
    let size = Log::of(seed);
    let girth = 0.5 * (size.butt + size.tip) * (1.0 - SAG);
    let mesh = records::mesh(world, "highlands_log", &seed.to_le_bytes(), move || {
        log_mesh(seed)
    })?;
    Ok(PlacementAsset {
        mesh,
        material,
        lods: Vec::new(),
        parts: Vec::new(),
        shadow: None,
        sway: 0.0,
        collider: Some(PlacedCollider::new(
            PlacedShape::Cuboid {
                half_extents: [0.5 * size.length, girth, girth],
            },
            [0.0, girth, 0.0],
        )),
    })
}

/// Stump `seed` as a placement asset drawn with `material`: its mesh, a
/// record ([`records::mesh`]) whose handle is reserved until the record is
/// here, colliding as a box about its trunk; rigid.
pub(crate) fn stump(
    world: &mut World,
    seed: u32,
    material: MaterialHandle,
) -> engine::Result<PlacementAsset> {
    let size = Stump::of(seed);
    let mesh = records::mesh(world, "highlands_stump", &seed.to_le_bytes(), move || {
        stump_mesh(seed)
    })?;
    Ok(PlacementAsset {
        mesh,
        material,
        lods: Vec::new(),
        parts: Vec::new(),
        shadow: None,
        sway: 0.0,
        collider: Some(PlacedCollider::new(
            PlacedShape::Cuboid {
                half_extents: [size.radius, 0.5 * size.height, size.radius],
            },
            [0.0, 0.5 * size.height, 0.0],
        )),
    })
}

/// Meters of stem one repeat of the bark covers along it: about a segment
/// of a spruce's trunk, each of which maps it once ([`super::trees`]).
const BARK_REPEAT: f32 = 2.0;

/// A log's tube: its rings along it and its sides around.
const LOG_RINGS: u32 = 5;
const LOG_SIDES: u32 = 8;

/// How much rot flattens a log's section where it lies, and how lumpy it
/// leaves its surface: shares of its radius.
const SAG: f32 = 0.1;
const LUMPS: f32 = 0.08;

/// How far a broken end's splinters reach either way along a log, as a
/// share of its radius there.
const SPLINTERS: f32 = 0.6;

/// The fewest branch stubs a log keeps and how many more it may, and their
/// length, shortest and longest, and radius at their foot (meters): a
/// spruce's dead branches snap off a hand or two out from the stem.
const STUBS: [u32; 2] = [4, 3];
const STUB_LENGTH: [f32; 2] = [0.1, 0.3];
const STUB_RADIUS: f32 = 0.025;

/// A log's size: its length, its radius at its butt and where it broke,
/// and how far it bows sideways at its middle (meters).
struct Log {
    length: f32,
    butt: f32,
    tip: f32,
    bow: f32,
}

impl Log {
    /// The log of seed `seed`: 3.5 to 7 meters of a spruce's stem, 0.13 to
    /// 0.22 meters in radius at its butt and tapering by a fifth to two
    /// fifths to where it broke, as a spruce's stem does over its lower
    /// half, bowed by up to two percent of its length.
    fn of(seed: u32) -> Self {
        let random = |index: u32| unit(golden_hash(seed.wrapping_mul(0x7f4a_7c15) ^ index));
        let length = 3.5 + 3.5 * random(1);
        let butt = 0.13 + 0.09 * random(2);
        Self {
            length,
            butt,
            tip: butt * (0.6 + 0.2 * random(3)),
            bow: 0.02 * length * (2.0 * random(4) - 1.0),
        }
    }

    /// Its radius `t` of the way from its butt to its broken end.
    fn radius(&self, t: f32) -> f32 {
        self.butt + (self.tip - self.butt) * t
    }

    /// Its axis `t` of the way from its butt to its broken end: resting on
    /// the ground under its origin, bowed.
    fn axis(&self, t: f32) -> Vec3 {
        Vec3::new(
            (t - 0.5) * self.length,
            self.radius(t) * (1.0 - SAG),
            self.bow * math::sin(PI * t),
        )
    }
}

/// Log `seed`'s mesh: its tube along x from its butt at -length/2 to its
/// broken end at length/2, resting on the ground at its origin's height,
/// its broken end splintered and its butt sawn or snapped by its seed,
/// both ends closed; and its stubs, standing out of its top and sides
/// toward its broken end, where its crown was.
fn log_mesh(seed: u32) -> engine::Result<Mesh> {
    let size = Log::of(seed);
    let random = |index: u32| unit(golden_hash(seed.wrapping_mul(0x7f4a_7c15) ^ index));
    let offset = Vec3::new(random(5), random(6), random(7)) * 23.0;
    let snapped = !seed.is_multiple_of(2);
    // A point of its surface: at ring `ring` along it and side `side` round
    // it, and the outward way there, across its flattened section. Its
    // ends' rings splinter along it.
    let surface = |ring: u32, side: u32| {
        let side = side % LOG_SIDES;
        let t = ring as f32 / (LOG_RINGS - 1) as f32;
        let (sin, cos) = math::sin_cos(TAU * side as f32 / LOG_SIDES as f32);
        let out = Vec3::new(0.0, sin / (1.0 - SAG), cos).normalize_or(Vec3::Z);
        let lump = noise(Vec3::new(t * size.length * 1.5, sin * 2.0, cos * 2.0) + offset);
        let radius = size.radius(t) * (1.0 + LUMPS * (2.0 * lump - 1.0));
        let broken = ring + 1 == LOG_RINGS || (ring == 0 && snapped);
        let splinter = if broken {
            SPLINTERS * size.radius(t) * (2.0 * random(200 + ring * LOG_SIDES + side) - 1.0)
        } else {
            0.0
        };
        let point = size.axis(t) + Vec3::new(splinter, sin * radius * (1.0 - SAG), cos * radius);
        (point, out)
    };
    let mut builder = Builder::default();
    // The tube, each segment mapping its length's share of the bark.
    let segment = size.length / (LOG_RINGS - 1) as f32;
    let span = 0.98 * math::min(segment / BARK_REPEAT, 1.0);
    for ring in 0..LOG_RINGS - 1 {
        let first = builder.vertices.len() as u32;
        for side in 0..=LOG_SIDES {
            let u = BARK[0] + (BARK[1] - BARK[0]) * side as f32 / LOG_SIDES as f32;
            for (end, v) in [(ring, 0.99), (ring + 1, 0.99 - span)] {
                let (point, out) = surface(end, side);
                builder.push(point, out, [u, v], underside(out));
            }
        }
        for side in 0..LOG_SIDES {
            let [a, b, c, d] = [0, 1, 2, 3].map(|corner| first + side * 2 + corner);
            builder.indices.extend([a, b, c, b, d, c]);
        }
    }
    // Its ends, each a fan from its middle facing along it.
    for (ring, facing) in [(0, -1.0_f32), (LOG_RINGS - 1, 1.0)] {
        let t = ring as f32 / (LOG_RINGS - 1) as f32;
        let middle = size.axis(t);
        let normal = Vec3::X * facing;
        let center = builder.push(middle, normal, BARK_MIDDLE, [0.75; 3]);
        let first = builder.vertices.len() as u32;
        for side in 0..LOG_SIDES {
            let u = BARK[0] + (BARK[1] - BARK[0]) * side as f32 / LOG_SIDES as f32;
            builder.push(surface(ring, side).0, normal, [u, 0.4], [0.75; 3]);
        }
        for side in 0..LOG_SIDES {
            let [left, right] = [side, (side + 1) % LOG_SIDES].map(|corner| first + corner);
            if facing > 0.0 {
                builder.indices.extend([center, right, left]);
            } else {
                builder.indices.extend([center, left, right]);
            }
        }
    }
    // Its stubs, over its top and sides, toward where its crown was.
    let stubs = STUBS[0] + golden_hash(seed ^ 0x2545_f491) % (STUBS[1] + 1);
    for stub in 0..stubs {
        let index = 60 + stub * 4;
        let t = 0.25 + 0.7 * random(index);
        let (sin, cos) = math::sin_cos(-0.2 + (PI + 0.4) * random(index + 1));
        let out = Vec3::new(0.0, sin, cos);
        let foot = size.axis(t) + out * (0.9 * size.radius(t));
        let length = STUB_LENGTH[0] + (STUB_LENGTH[1] - STUB_LENGTH[0]) * random(index + 2);
        // Branches point up their tree, toward its top.
        builder.spike(foot, (out + Vec3::X * 0.7).normalize_or(out) * length);
    }
    builder
        .mesh()
        .map_err(|error| engine::Error::from(error).context(format!("log {seed}")))
}

/// A stump's rings up it, as shares of its height (the first under the
/// ground, so its foot never shows on a slope), and its sides round it.
const STUMP_RINGS: [f32; 5] = [-0.4, 0.0, 0.12, 0.35, 1.0];
const STUMP_SIDES: u32 = 10;

/// How far a stump's roots flare out at the ground, as a share of its
/// radius, and up what share of its height they flare from; and how far its
/// broken top's splinters reach above and below its mean, as shares of its
/// height.
const FLARE: f32 = 0.6;
const FLARE_RISE: f32 = 0.45;
const JAGGED: f32 = 0.25;

/// A stump's size: its height over the ground, its trunk's radius, its
/// roots and the turn of the first, and whether it was sawn or broke.
struct Stump {
    height: f32,
    radius: f32,
    roots: u32,
    turn: f32,
    sawn: bool,
}

impl Stump {
    /// The stump of seed `seed`: 0.32 to 0.5 meters tall and 0.17 to 0.25
    /// in radius, flaring into four or five roots, every other one sawn
    /// level and the rest broken off jagged.
    fn of(seed: u32) -> Self {
        let random = |index: u32| unit(golden_hash(seed.wrapping_mul(0x1b87_3593) ^ index));
        Self {
            height: 0.32 + 0.18 * random(1),
            radius: 0.17 + 0.08 * random(2),
            roots: 4 + golden_hash(seed ^ 0x3c6e_f372) % 2,
            turn: random(3) * TAU,
            sawn: seed.is_multiple_of(2),
        }
    }

    /// Its radius turned `angle` round it, `y` meters up: its trunk's, and
    /// below its flare's rise wider, the more toward each root.
    fn reach(&self, angle: f32, y: f32) -> f32 {
        let flare = math::clamp(1.0 - y / (FLARE_RISE * self.height), 0.0, 1.0);
        let root = math::max(math::cos(self.roots as f32 * (angle - self.turn)), 0.0);
        self.radius * (1.0 + FLARE * flare * flare * (0.3 + 0.7 * root * root))
    }
}

/// Stump `seed`'s mesh: its trunk's foot round its origin, flaring into its
/// roots toward the ground and on under it, mapping the bark up it, darker
/// toward its roots; and its top, sawn level or broken jagged, closed.
fn stump_mesh(seed: u32) -> engine::Result<Mesh> {
    let size = Stump::of(seed);
    let random = |index: u32| unit(golden_hash(seed.wrapping_mul(0x1b87_3593) ^ index));
    let rings = STUMP_RINGS.len() as u32;
    // Each side's height at the top: level where sawn, else splintered.
    let top = |side: u32| {
        if size.sawn {
            size.height
        } else {
            size.height * (1.0 + JAGGED * (2.0 * random(20 + side % STUMP_SIDES) - 1.0))
        }
    };
    let foot = STUMP_RINGS[0] * size.height;
    let mut builder = Builder::default();
    for side in 0..=STUMP_SIDES {
        let angle = TAU * (side % STUMP_SIDES) as f32 / STUMP_SIDES as f32;
        let (sin, cos) = math::sin_cos(angle);
        let u = BARK[0] + (BARK[1] - BARK[0]) * side as f32 / STUMP_SIDES as f32;
        for (ring, &share) in STUMP_RINGS.iter().enumerate() {
            let y = if ring as u32 + 1 == rings {
                top(side)
            } else {
                share * size.height
            };
            let radius = size.reach(angle, y);
            // The flare's slope up the stump tilts its normal up.
            let flaring = (size.reach(angle, y - 0.02) - size.reach(angle, y + 0.02)) / 0.04;
            let normal = Vec3::new(cos, flaring, sin);
            let v = 0.99 - 0.98 * (y - foot) / BARK_REPEAT;
            let shade = 0.55 + 0.45 * math::clamp(y / size.height, 0.0, 1.0);
            builder.push(
                Vec3::new(cos * radius, y, sin * radius),
                normal,
                [u, v],
                [shade; 3],
            );
        }
    }
    let at = |side: u32, ring: u32| side * rings + ring;
    for side in 0..STUMP_SIDES {
        for ring in 0..rings - 1 {
            let [a, b, c, d] = [
                at(side, ring),
                at(side, ring + 1),
                at(side + 1, ring),
                at(side + 1, ring + 1),
            ];
            builder.indices.extend([a, b, c, b, d, c]);
        }
    }
    // Its top, a fan from its middle, a little sunk where it broke.
    let middle = if size.sawn {
        size.height
    } else {
        size.height * (1.0 - 0.3 * JAGGED)
    };
    let center = builder.push(Vec3::new(0.0, middle, 0.0), Vec3::Y, BARK_MIDDLE, [0.8; 3]);
    let first = builder.vertices.len() as u32;
    for side in 0..STUMP_SIDES {
        let angle = TAU * side as f32 / STUMP_SIDES as f32;
        let (sin, cos) = math::sin_cos(angle);
        let y = top(side);
        let radius = size.reach(angle, y);
        let u = BARK[0] + (BARK[1] - BARK[0]) * side as f32 / STUMP_SIDES as f32;
        builder.push(
            Vec3::new(cos * radius, y, sin * radius),
            Vec3::Y,
            [u, 0.4],
            [0.8; 3],
        );
    }
    for side in 0..STUMP_SIDES {
        let [left, right] = [side, (side + 1) % STUMP_SIDES].map(|corner| first + corner);
        builder.indices.extend([center, right, left]);
    }
    builder
        .mesh()
        .map_err(|error| engine::Error::from(error).context(format!("stump {seed}")))
}

/// The shade of bark facing `out`: darker underneath, where it lies on the
/// damp ground and its own girth shades it.
fn underside(out: Vec3) -> [f32; 3] {
    [0.55 + 0.45 * math::clamp(0.5 + 0.5 * out.y, 0.0, 1.0); 3]
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

    /// A three-sided spike in the bark from `foot` out along `reach`,
    /// [`STUB_RADIUS`] thick at its foot: a branch's stub.
    fn spike(&mut self, foot: Vec3, reach: Vec3) {
        let along = reach.normalize_or(Vec3::Y);
        let side_a = along.cross(Vec3::Y).normalize_or(Vec3::Z);
        let side_b = side_a.cross(along);
        let tip = self.push(foot + reach, along, [BARK[0] + 0.1, 0.3], [0.7; 3]);
        let first = self.vertices.len() as u32;
        for side in 0..3 {
            let (sin, cos) = math::sin_cos(TAU * side as f32 / 3.0);
            let out = side_a * cos + side_b * sin;
            let u = BARK[0] + 0.05 * side as f32;
            self.push(foot + out * STUB_RADIUS, out, [u, 0.35], [0.7; 3]);
        }
        for side in 0..3 {
            self.indices
                .extend([first + side, tip, first + (side + 1) % 3]);
        }
    }
}
