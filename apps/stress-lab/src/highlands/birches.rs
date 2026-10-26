//! Downy birches (Betula pubescens) for the highlands, the scene's own
//! content, built as the spruces are ([`super::trees`]): a white stem
//! sweeping a little aside as it rises, ascending limbs reaching out to a
//! rounded crown, and along each limb sprays of leaves, each a flat card
//! across it and a hanging card through it, over an atlas of their own
//! whose left eighth is bark and the rest a leafy twig, drawn with the
//! card trees' masked material shaded as foliage. A card's normal is mostly
//! the crown's, out of its rounded outline, so the crown lights as one
//! light, rounded volume against the spruces' dark cones; vertex colors
//! darken its inner and lower sprays. How a birch grows ([`Growth`]) sets
//! its height, its crown and its limbs: those of the forest and the stands
//! beyond it ([`BIRCH`]) and young ones at its edges ([`YOUNG_BIRCH`]).
//! Coarser levels keep every card's size and drop each limb's inner
//! sprays first, the one at its tip last, so the crown keeps its outline;
//! every level's cards are cut to the leaves' alpha ([`trim_cards`]).
//! Shadow maps draw each birch as a closed solid crown within its cards'
//! outline, over a spike of a stem.

use std::f32::consts::TAU;
use std::sync::OnceLock;

use engine::prelude::*;
use engine::terrain::{PlacementAsset, PlacementLod};

use super::trees::{
    ATLAS, Builder, CUTOFF, Canvas, ERRORS, RIM_SIDES, SPIKE, SprayMap, encoded, noise, sway,
    trunk_collider,
};
use crate::kit::random::{golden_hash, unit};
use crate::kit::records;

/// Birch `seed` grown as `growth` says, as a placement asset drawn with
/// `material`, the card trees' material over this atlas
/// ([`atlas_record`]): its levels, finest first, switching at [`ERRORS`]
/// of its sprays' spacing, and its shadow proxy, each a record
/// ([`records::mesh`]) whose handle is reserved until the record is here.
/// It sways and its stem collides as its size asks.
pub(crate) fn birch(
    world: &mut World,
    seed: u32,
    growth: &'static Growth,
    material: MaterialHandle,
) -> engine::Result<PlacementAsset> {
    let mut levels = (0..DETAILS.len())
        .map(|detail| {
            let mut identity = seed.to_le_bytes().to_vec();
            identity.extend_from_slice(&(detail as u32).to_le_bytes());
            records::mesh(world, growth.kind, &identity, move || {
                birch_mesh(seed, growth, detail)
            })
        })
        .collect::<engine::Result<Vec<_>>>()?;
    let shadow = records::mesh(
        world,
        &format!("{}_shadow", growth.kind),
        &seed.to_le_bytes(),
        move || shadow_proxy(seed, growth),
    )?;
    let mesh = levels.remove(0);
    let shape = Shape::of(seed, growth);
    Ok(PlacementAsset {
        mesh,
        material,
        lods: levels
            .into_iter()
            .zip(ERRORS)
            .map(|(mesh, error)| PlacementLod {
                mesh,
                error: error * growth.spray_spacing,
            })
            .collect(),
        parts: Vec::new(),
        sway: sway(shape.height),
        collider: Some(trunk_collider(
            shape.radius(0.0),
            shape.foot,
            shape.crown.foot,
        )),
        shadow: Some(shadow),
    })
}

/// The bark and leaf atlas as a texture record ([`records::textures`]),
/// cooked as every generated texture is: its handle, which materials name
/// at once.
pub(crate) fn atlas_record(world: &mut World) -> engine::Result<TextureHandle> {
    let handles = world.resource_scope(|world, mut stored: Mut<Records>| {
        let mut assets = world.resource_mut::<AssetLibrary>();
        records::textures(
            (&mut stored, &mut assets),
            "highlands/birch_atlas".into(),
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

/// How a birch grows, which every level and its shadow proxy keep: the
/// kind its records are kept as; its height and its stem's radius at its
/// foot, each from the first value to the second; its crown's foot and its
/// widest radius, as shares of its height, each from the first to the
/// second; the meters between limbs up the stem, each spacing varying
/// about it; the limbs' rise from level (radians) at the crown's foot and
/// at its top; the share of their length limbs and sprays droop at their
/// tips; the meters between sprays along a limb, and a spray's length at a
/// limb's tip; the meters its stem sweeps aside at its top, at most; and
/// the meters its stem reaches into the ground.
pub(crate) struct Growth {
    kind: &'static str,
    height: [f32; 2],
    base_radius: [f32; 2],
    crown_base: [f32; 2],
    crown_radius: [f32; 2],
    limb_spacing: f32,
    rise: [f32; 2],
    droop: f32,
    spray_spacing: f32,
    spray_length: f32,
    sweep: f32,
    foot: f32,
}

/// A downy birch of the forest and the stands beyond it: 9 to 12 meters
/// tall, below the spruces it grows among, its stem bare for about a third
/// of its height and its ovoid crown about half as wide as it is tall, its
/// limbs rising more steeply toward its top, its twigs barely drooping.
pub(crate) const BIRCH: Growth = Growth {
    kind: "highlands_birch",
    height: [9.0, 12.0],
    base_radius: [0.12, 0.16],
    crown_base: [0.3, 0.4],
    crown_radius: [0.21, 0.26],
    limb_spacing: 0.32,
    rise: [0.5, 1.05],
    droop: 0.12,
    spray_spacing: 0.4,
    spray_length: 0.85,
    sweep: 0.35,
    foot: 0.4,
};

/// A young downy birch at the forest's edge: 3.2 to 4.6 meters tall and
/// slender, keeping its limbs low, its narrow crown about a third as wide
/// as it is tall, its limbs rising steeply.
pub(crate) const YOUNG_BIRCH: Growth = Growth {
    kind: "highlands_young_birch",
    height: [3.2, 4.6],
    base_radius: [0.03, 0.045],
    crown_base: [0.12, 0.22],
    crown_radius: [0.16, 0.2],
    limb_spacing: 0.24,
    rise: [0.75, 1.2],
    droop: 0.06,
    spray_spacing: 0.28,
    spray_length: 0.48,
    sweep: 0.12,
    foot: 0.15,
};

/// The bark strip's share of the atlas's width, on its left: about the
/// share of a birch's outline its white stem and limbs fill, so the
/// atlas's mean color, which the ground far away takes on where birches
/// retire (the palette's placed objects' color), is the tree's.
const STRIP: f32 = 0.125;
/// Bark texture coordinates span this much of the atlas's width: the strip
/// less a margin, so filtering never reaches the leaves.
const BARK: [f32; 2] = [0.01, STRIP - 0.01];
/// Texture coordinates amid the bark, which covers the strip throughout:
/// those of every vertex of a shadow proxy.
const BARK_MIDDLE: [f32; 2] = [0.5 * STRIP, 0.5];
/// The leafy twig in the atlas: right of the strip less a margin, its base
/// at the bottom.
const LEAVES: SprayMap = SprayMap {
    center: 0.5 + 0.5 * STRIP,
    half: 0.5 - 0.5 * STRIP - 0.012,
    base: 0.99,
    tip: 0.01,
};

/// The atlas's linear colors: the bark's white, reflecting about half the
/// visible light as birch bark does; the lenticels and scars marking it;
/// the twigs' dark red-brown; and the leaves' mean green, lighter and
/// yellower than the spruces' needles, as broadleaf canopies reflect more
/// than conifers'.
const WHITE: [f32; 3] = [0.6, 0.58, 0.53];
const MARK: [f32; 3] = [0.045, 0.04, 0.036];
const TWIG: [f32; 3] = [0.07, 0.035, 0.025];
const LEAF: [f32; 3] = [0.085, 0.16, 0.035];
/// The lenticels and the limbs' scars the bark carries.
const LENTICELS: u32 = 110;
const SCARS: u32 = 6;

/// The share of the crown's length, from its foot, at which it is widest:
/// a little below its middle, as birches' ovoid crowns are.
const WIDEST: f32 = 0.45;
/// The share of a limb's length, from its tip, its sprays grow along.
const OUTER: f32 = 0.7;
/// The turn round the stem from one limb to the next: the golden angle,
/// which spreads leaves and limbs evenly round a stem (phyllotaxis).
const GOLDEN_ANGLE: f32 = 2.399_963;
/// How far a spray's normal leans up from the crown's outline, toward the
/// sky lighting it.
const LEAN: f32 = 0.3;
/// A limb's radius where it leaves the stem: a centimeter, and 1.2 more
/// for each meter of its length.
const LIMB_RADIUS: [f32; 2] = [0.01, 0.012];
/// The shade of the limbs, inside the crown.
const LIMB_SHADE: f32 = 0.75;

/// The share of the crown's outline, about its widest, the solid crown
/// shadow maps draw fills along each axis: its open crown shades about
/// two thirds of its outline.
const SOLID: f32 = 0.8;
/// Bands of the solid crown from its foot to its top.
const CROWN_RINGS: u32 = 4;

/// How finely a level is built: its stem's sides and segments (each
/// mapping the bark once), its limbs' sides (none: their sprays alone),
/// the segments of its limbs and of its sprays' cards, and the share of
/// each limb's sprays it keeps, at the crown's foot and at its top
/// (between them by the limb's height), one at least: the outermost, as
/// the limb lists them from its tip inward.
struct Detail {
    sides: u32,
    segments: u32,
    limb_sides: u32,
    bends: usize,
    keep: [f32; 2],
}

/// The full tree, then each coarser level.
const DETAILS: [Detail; 4] = [
    Detail {
        sides: 7,
        segments: 8,
        limb_sides: 4,
        bends: 2,
        keep: [1.0, 1.0],
    },
    Detail {
        sides: 5,
        segments: 4,
        limb_sides: 3,
        bends: 2,
        keep: [0.6, 1.0],
    },
    Detail {
        sides: 4,
        segments: 2,
        limb_sides: 3,
        bends: 1,
        keep: [0.45, 0.7],
    },
    Detail {
        sides: 3,
        segments: 1,
        limb_sides: 0,
        bends: 1,
        keep: [0.3, 0.5],
    },
];

/// Birch `seed` grown as `growth` says, built at `detail` (0 the full
/// tree, up to 3): as tall as `growth` grows it before placement scales
/// it, its stem's foot as far into the ground as `growth` says, its cards
/// cut to the leaves' alpha.
fn birch_mesh(seed: u32, growth: &Growth, detail: usize) -> engine::Result<Mesh> {
    let build = &DETAILS[detail];
    let shape = Shape::of(seed, growth);
    let mut builder = Builder::default();
    // The stem greys toward its foot, where its bark thickens.
    let height = shape.height;
    builder.tube(&shape.stem(build.segments), build.sides, BARK, |y| {
        0.6 + 0.4 * math::clamp(y / (0.25 * height), 0.0, 1.0)
    });
    for limb in &shape.limbs {
        if build.limb_sides > 0 {
            builder.tube(&limb.rings(build.bends), build.limb_sides, BARK, |_| {
                LIMB_SHADE
            });
        }
        // The limb's outermost sprays, as many as its share keeps.
        let count = limb.sprays.len();
        let share = build.keep[0] + (build.keep[1] - build.keep[0]) * limb.rise;
        let kept = ((count as f32 * share).round() as usize).clamp(1, count);
        for spray in &limb.sprays[..kept] {
            cards(&mut builder, spray, build.bends);
        }
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
            engine::Error::from(error).context(format!("birch {seed} at detail {detail}"))
        })
}

/// `spray`'s cards along it in `bends` segments, drooping toward its tip:
/// a flat one across it and a hanging one through it, lit as the crown's
/// outline nearest it.
fn cards(builder: &mut Builder, spray: &Spray, bends: usize) {
    let spine: Vec<Vec3> = (0..=bends)
        .map(|step| spray.at(step as f32 / bends as f32))
        .collect();
    let across = Vec3::new(-spray.direction.z, 0.0, spray.direction.x).normalize_or(Vec3::X);
    let hanging = spray.direction.cross(across).normalize_or(Vec3::Y);
    builder.card(&spine, across * (0.42 * spray.length), spray.normal, spray.color, LEAVES);
    builder.card(&spine, hanging * (0.28 * spray.length), spray.normal, spray.color, LEAVES);
}

/// Birch `seed`'s shadow proxy, grown as `growth` says, which shadow maps
/// draw in place of its levels: a four-sided spike of a stem from its foot
/// below the ground to its top, and its crown closed and solid, [`SOLID`]
/// of its outline about its widest, of [`RIM_SIDES`] sides and
/// [`CROWN_RINGS`] bands, well within its cards' outline. Its vertices
/// carry the streams its levels' do, white and amid the bark.
fn shadow_proxy(seed: u32, growth: &Growth) -> engine::Result<Mesh> {
    let shape = Shape::of(seed, growth);
    let mut builder = Builder::default();
    builder.spike(
        Vec3::new(0.0, -shape.foot, 0.0),
        SPIKE * shape.radius(-shape.foot),
        shape.axis(shape.height),
        BARK_MIDDLE,
    );
    let crown = &shape.crown;
    let center = shape.axis(crown.middle);
    // The height of the outline's point `y` meters up, drawn in toward
    // its widest as its radius is.
    let drawn = |y: f32| center.y + (y - crown.middle) * SOLID;
    let low = builder.push(
        Vec3::new(center.x, drawn(crown.foot), center.z),
        Vec3::NEG_Y,
        BARK_MIDDLE,
        [1.0; 3],
    );
    let high = builder.push(
        Vec3::new(center.x, drawn(crown.top), center.z),
        Vec3::Y,
        BARK_MIDDLE,
        [1.0; 3],
    );
    let first = builder.vertices.len() as u32;
    for ring in 1..CROWN_RINGS {
        let y = crown.foot + (crown.top - crown.foot) * ring as f32 / CROWN_RINGS as f32;
        let radius = SOLID * crown.radius(y);
        for side in 0..RIM_SIDES {
            let (sin, cos) = math::sin_cos(TAU * side as f32 / RIM_SIDES as f32);
            let outward = Vec3::new(cos, 0.0, sin);
            let rim = Vec3::new(center.x, drawn(y), center.z) + outward * radius;
            builder.push(rim, outward, BARK_MIDDLE, [1.0; 3]);
        }
    }
    // Its cap below, its bands between its rings and its cap above, each
    // wound to face out.
    let at = |ring: u32, side: u32| first + ring * RIM_SIDES + side % RIM_SIDES;
    let bands = CROWN_RINGS - 2;
    for side in 0..RIM_SIDES {
        builder.indices.extend([low, at(0, side), at(0, side + 1)]);
        for ring in 0..bands {
            let [a, b, c, d] = [
                at(ring, side),
                at(ring, side + 1),
                at(ring + 1, side),
                at(ring + 1, side + 1),
            ];
            builder.indices.extend([a, c, b, b, c, d]);
        }
        builder.indices.extend([high, at(bands, side + 1), at(bands, side)]);
    }
    builder.mesh().map_err(|error| {
        engine::Error::from(error).context(format!("birch {seed}'s shadow proxy"))
    })
}

/// A birch's shape, which every level and its shadow proxy keep: its
/// height, its stem's radius at its foot, the way and meters its stem
/// sweeps aside at its top, the meters its stem reaches into the ground,
/// its crown's outline, and its limbs from the lowest up, its leader last.
struct Shape {
    height: f32,
    base_radius: f32,
    sweep: Vec2,
    foot: f32,
    crown: Crown,
    limbs: Vec<Limb>,
}

/// A crown's rounded outline about the stem: its foot, the height it is
/// widest at and its top, and its widest radius; half an ellipse turned
/// about the stem below its widest, and another above.
struct Crown {
    foot: f32,
    middle: f32,
    top: f32,
    radius: f32,
}

impl Crown {
    /// The half length of the half holding height `y`: the half below its
    /// widest, or the half above.
    fn half(&self, y: f32) -> f32 {
        let half = if y < self.middle {
            self.middle - self.foot
        } else {
            self.top - self.middle
        };
        math::max(half, 1.0e-3)
    }

    /// Its radius `y` meters up: 0 below its foot and above its top.
    fn radius(&self, y: f32) -> f32 {
        let across = (y - self.middle) / self.half(y);
        self.radius * math::max(1.0 - across * across, 0.0).sqrt()
    }
}

/// A limb: where it leaves the stem, the way it points, its length and its
/// radius at the stem, the share of its length its tip droops, how far up
/// the crown it stands (0 at its foot, 1 at its top), and its sprays, from
/// its tip inward.
struct Limb {
    start: Vec3,
    direction: Vec3,
    length: f32,
    radius: f32,
    droop: f32,
    rise: f32,
    sprays: Vec<Spray>,
}

/// A spray of leaves on a limb: where it leaves it, the way it points, its
/// length and the share of it its tip droops, the crown's normal nearest
/// it, which it is lit as, and its shade.
struct Spray {
    start: Vec3,
    direction: Vec3,
    length: f32,
    droop: f32,
    normal: Vec3,
    color: [f32; 3],
}

/// The point `t` of the way along a limb or a spray from `start` pointing
/// `direction`, `length` meters long, its tip drooping `droop` of its
/// length below the way it points.
fn drooping(start: Vec3, direction: Vec3, (length, droop): (f32, f32), t: f32) -> Vec3 {
    start + direction * (length * t) - Vec3::Y * (droop * length * t * t)
}

impl Limb {
    /// The point `t` of its length out (1 its tip).
    fn at(&self, t: f32) -> Vec3 {
        drooping(self.start, self.direction, (self.length, self.droop), t)
    }

    /// The way it runs `t` of its length out.
    fn way(&self, t: f32) -> Vec3 {
        (self.direction - Vec3::Y * (2.0 * self.droop * t)).normalize_or(self.direction)
    }

    /// Its tube's rings over `segments` segments: its axis and its radius,
    /// narrowing to a fifth of it at its tip.
    fn rings(&self, segments: usize) -> Vec<(Vec3, f32)> {
        (0..=segments)
            .map(|step| {
                let t = step as f32 / segments as f32;
                (self.at(t), self.radius * (1.0 - 0.8 * t))
            })
            .collect()
    }
}

impl Spray {
    /// The point `t` of its length out (1 its tip).
    fn at(&self, t: f32) -> Vec3 {
        drooping(self.start, self.direction, (self.length, self.droop), t)
    }
}

impl Shape {
    /// The shape of seed `seed` grown as `growth` says: its stem sweeping
    /// aside toward its top; its crown from a share of its height up to its
    /// top, widest [`WIDEST`] of the way up; limbs every limb spacing or so
    /// up the stem from the crown's foot, each a golden angle round from
    /// the last ([`GOLDEN_ANGLE`]), rising more steeply up the crown and
    /// reaching out to its outline ([`Shape::limb`]); and its leader, a limb
    /// carrying the stem on up into the crown's top.
    fn of(seed: u32, growth: &Growth) -> Self {
        let random = |index: u32| unit(golden_hash(seed.wrapping_mul(0x85eb_ca6b) ^ index));
        let between = |range: [f32; 2], draw: f32| range[0] + (range[1] - range[0]) * draw;
        let height = between(growth.height, random(1));
        let (sin, cos) = math::sin_cos(random(3) * TAU);
        let foot = between(growth.crown_base, random(5)) * height;
        let mut shape = Self {
            height,
            base_radius: between(growth.base_radius, random(2)),
            sweep: Vec2::new(cos, sin) * (growth.sweep * random(4)),
            foot: growth.foot,
            crown: Crown {
                foot,
                middle: foot + WIDEST * (height - foot),
                top: height,
                radius: between(growth.crown_radius, random(6)) * height,
            },
            limbs: Vec::new(),
        };
        // The highest limb stands below the leader's sprays.
        let last = height - 2.0 * growth.spray_length;
        let mut y = foot;
        let mut turn = random(7) * TAU;
        let mut number = 0_u32;
        while y < last {
            let key = 1000 + number * 256;
            let up = math::clamp((y - foot) / math::max(last - foot, 1.0e-3), 0.0, 1.0);
            let rise = between(growth.rise, up) + 0.2 * (random(key) - 0.5);
            turn += GOLDEN_ANGLE + 0.5 * (random(key + 1) - 0.5);
            let (rise_sin, rise_cos) = math::sin_cos(rise);
            let (turn_sin, turn_cos) = math::sin_cos(turn);
            let direction = Vec3::new(turn_cos * rise_cos, rise_sin, turn_sin * rise_cos);
            // As long as reaches the outline's radius at the height its tip
            // rises to, found in a few steps each halving the gap.
            let reach = |length: f32| {
                shape.crown.radius(y + length * rise_sin) / math::max(rise_cos, 0.25)
            };
            let mut length = reach(0.0);
            for _ in 0..3 {
                length = 0.5 * (length + reach(length));
            }
            let length = length * (0.8 + 0.2 * random(key + 2));
            if length > growth.spray_spacing {
                let limb = shape.limb(
                    (shape.axis(y), direction, length),
                    (up, growth),
                    (&random, key),
                );
                shape.limbs.push(limb);
            }
            y += growth.limb_spacing * (0.75 + 0.5 * random(key + 3));
            number += 1;
        }
        let start = shape.axis(last);
        let direction = (shape.axis(height) - start).normalize_or(Vec3::Y);
        let length = height - last + 0.5 * growth.spray_length;
        let leader = shape.limb(
            (start, direction, length),
            (1.0, growth),
            (&random, 1000 + number * 256),
        );
        shape.limbs.push(leader);
        shape
    }

    /// A limb leaving the stem at `start` along `direction`, `length`
    /// meters long, `up` of the way up the crown, with its sprays along its
    /// outer [`OUTER`] a spray's spacing apart from its tip inward: the
    /// tip's carrying on its way, the others turned to either side in
    /// turn, each reaching up a little and shorter toward the stem, lit as
    /// the crown's outline nearest it and darker the deeper in the crown
    /// and the lower it stands; its draws are `random`'s past `key`.
    fn limb(
        &self,
        (start, direction, length): (Vec3, Vec3, f32),
        (up, growth): (f32, &Growth),
        (random, key): (&impl Fn(u32) -> f32, u32),
    ) -> Limb {
        let mut limb = Limb {
            start,
            direction,
            length,
            radius: LIMB_RADIUS[0] + LIMB_RADIUS[1] * length,
            droop: growth.droop,
            rise: up,
            sprays: Vec::new(),
        };
        let count = (OUTER * length / growth.spray_spacing) as u32 + 1;
        for number in 0..count {
            let draw = |salt: u32| random(key + 4 + number * 4 + salt);
            let t = 1.0 - number as f32 * growth.spray_spacing / length;
            let way = limb.way(t);
            let across = way.cross(Vec3::Y).normalize_or(Vec3::X);
            let upward = across.cross(way);
            let pointing = if number == 0 {
                way + upward * 0.3
            } else {
                let side = if number.is_multiple_of(2) { 1.0 } else { -1.0 };
                let (sin, cos) = math::sin_cos(0.6 + 0.35 * draw(0));
                way * cos + across * (side * sin) + upward * 0.35
            };
            let at = limb.at(t);
            let direction = pointing.normalize_or(way);
            let long = growth.spray_length * (0.7 + 0.3 * t) * (0.85 + 0.3 * draw(1));
            let middle = at + direction * (0.5 * long);
            // How far out in the crown it stands: 0 at the stem, 1 at the
            // outline.
            let offset = middle - self.axis(middle.y);
            let out = math::clamp(
                Vec2::new(offset.x, offset.z).length()
                    / math::max(self.crown.radius(middle.y), 0.1),
                0.0,
                1.0,
            );
            let shade = 0.5 + 0.3 * out + 0.2 * up;
            limb.sprays.push(Spray {
                start: at,
                direction,
                length: long,
                droop: growth.droop,
                normal: (self.crown_normal(middle) + Vec3::Y * LEAN).normalize_or(Vec3::Y),
                color: [
                    shade * (0.9 + 0.2 * draw(2)),
                    shade,
                    shade * (0.9 + 0.2 * draw(3)),
                ],
            });
        }
        limb
    }

    /// The stem's axis `y` meters up: straight up below the ground, then
    /// sweeping aside as the square of its share of the height.
    fn axis(&self, y: f32) -> Vec3 {
        let share = math::max(y, 0.0) / self.height;
        let aside = self.sweep * (share * share);
        Vec3::new(aside.x, y, aside.y)
    }

    /// The stem's radius `y` meters up, narrowing toward its top.
    fn radius(&self, y: f32) -> f32 {
        self.base_radius * math::pow(math::max(1.0 - y / self.height, 0.0), 0.7) + 0.01
    }

    /// The stem's rings over `segments` segments, from its foot below the
    /// ground to its top.
    fn stem(&self, segments: u32) -> Vec<(Vec3, f32)> {
        (0..=segments)
            .map(|ring| {
                let y = -self.foot + (self.height + self.foot) * ring as f32 / segments as f32;
                (self.axis(y), self.radius(y))
            })
            .collect()
    }

    /// The outward normal at `point` of the crown's outline about the
    /// stem's axis at its widest: the gradient of its half's ellipse there.
    fn crown_normal(&self, point: Vec3) -> Vec3 {
        let offset = point - self.axis(self.crown.middle);
        let half = self.crown.half(point.y);
        let radius = math::max(self.crown.radius, 1.0e-3);
        Vec3::new(
            offset.x / (radius * radius),
            offset.y / (half * half),
            offset.z / (radius * radius),
        )
        .normalize_or(Vec3::Y)
    }
}

/// The atlas's texels ([`atlas`]), drawn once: what its texture record
/// cooks, and what every level's cards are cut to.
fn texels() -> &'static [u8] {
    static TEXELS: OnceLock<Vec<u8>> = OnceLock::new();
    TEXELS.get_or_init(atlas)
}

/// The atlas's RGBA8 texels, sRGB color with alpha coverage: white bark in
/// the strip on the left, banded across the stem where its papery layers
/// peel, marked with dark lenticels across it and dark scars drooping
/// either side of where limbs left it; on the rest a leafy twig rising
/// from the bottom edge, side twigs alternating up it, every twig set with
/// ovate leaves on short stalks, turning to either side in turn, and one
/// at its tip. Uncovered texels keep the leaves' mean color, so filtering
/// never darkens their edges.
fn atlas() -> Vec<u8> {
    let size = ATLAS as usize;
    let strip = (STRIP * ATLAS as f32) as usize;
    let mut color = vec![LEAF; size * size];
    let mut alpha = vec![0.0_f32; size * size];
    for y in 0..size {
        for x in 0..strip {
            let band = noise(Vec2::new(x as f32 / 23.0, y as f32 / 4.0));
            let mottle = noise(Vec2::new(x as f32 / 7.0, y as f32 / 29.0));
            let tone = 0.8 + 0.12 * band + 0.08 * mottle;
            color[y * size + x] = WHITE.map(|channel| channel * tone);
            alpha[y * size + x] = 1.0;
        }
    }
    let mut bark = Canvas {
        color: &mut color,
        alpha: &mut alpha,
        size,
        columns: [0, strip],
    };
    for mark in 0..LENTICELS {
        let draw = |salt: u32| unit(golden_hash(mark.wrapping_mul(0x2c1b_3c6d) ^ salt));
        let center = Vec2::new(draw(1) * strip as f32, draw(2) * size as f32);
        let half = Vec2::X * (1.5 + 4.5 * draw(3));
        bark.capsule(center - half, center + half, 0.5 + 0.4 * draw(4), MARK);
    }
    for scar in 0..SCARS {
        let draw = |salt: u32| unit(golden_hash(scar.wrapping_mul(0x7feb_352d) ^ salt ^ 0x5ca7));
        let stub = Vec2::new(
            (0.2 + 0.6 * draw(1)) * strip as f32,
            (0.05 + 0.9 * draw(2)) * size as f32,
        );
        let reach = 5.0 + 5.0 * draw(3);
        for side in [-1.0_f32, 1.0] {
            bark.capsule(stub, stub + Vec2::new(side * reach, 0.45 * reach), 1.3, MARK);
        }
    }
    let mut spray = Canvas {
        color: &mut color,
        alpha: &mut alpha,
        size,
        columns: [strip, size],
    };
    let width = (size - strip) as f32;
    let center = LEAVES.center * size as f32;
    let (base, tip) = (size as f32 - 6.0, 40.0);
    // Side twigs alternate up the main twig, longest at its base.
    let mut twigs = Vec::new();
    let mut along = 0.06;
    let mut side = 1.0;
    while along < 0.86 {
        let start = Vec2::new(center, base + (tip - base) * along);
        let reach = 0.34
            * width
            * (1.0 - 0.7 * along)
            * (0.85 + 0.3 * unit(golden_hash(twigs.len() as u32 + 11)));
        let end = start + Vec2::new(side * 0.78, -0.62).normalize() * reach;
        twigs.push((start, end));
        along += 0.075;
        side = -side;
    }
    for &(start, end) in &twigs {
        spray.capsule(start, end, 1.0, TWIG);
    }
    spray.capsule(Vec2::new(center, base), Vec2::new(center, tip), 1.8, TWIG);
    let mut leaves = 0_u32;
    let mut set = |spray: &mut Canvas<'_>, from: Vec2, to: Vec2, long: f32| {
        let span = to - from;
        let direction = span.normalize_or(Vec2::NEG_Y);
        let steps = (span.length() / 5.5) as u32;
        for step in 0..steps {
            let t = (step as f32 + 0.5) / steps as f32;
            let root = from + span * t;
            leaves += 1;
            let draw = |salt: u32| unit(golden_hash(leaves ^ salt));
            let turn = if leaves.is_multiple_of(2) { 1.0 } else { -1.0 };
            let (sin, cos) = math::sin_cos(turn * (0.7 + 0.35 * draw(0x68e3)));
            let pointing = Vec2::new(
                direction.x * cos - direction.y * sin,
                direction.x * sin + direction.y * cos,
            );
            let length = long * (1.0 - 0.3 * t) * (0.85 + 0.3 * draw(0x1b56));
            let shade = 0.8 + 0.4 * draw(0x2c1b);
            let yellow = 0.9 + 0.25 * draw(0x3d4f);
            let tint = [LEAF[0] * shade * yellow, LEAF[1] * shade, LEAF[2] * shade];
            let stalk = root + pointing * 3.0;
            spray.capsule(root, stalk, 0.6, TWIG);
            spray.leaf(stalk, stalk + pointing * length, 0.62 * length, tint);
        }
        let length = 0.75 * long;
        spray.leaf(to, to + direction * length, 0.62 * length, LEAF);
    };
    for &(start, end) in &twigs {
        let first = start + (end - start).normalize_or(Vec2::NEG_Y) * 6.0;
        set(&mut spray, first, end, 30.0);
    }
    set(
        &mut spray,
        Vec2::new(center, base - 10.0),
        Vec2::new(center, tip),
        32.0,
    );
    encoded(&color, &alpha)
}
