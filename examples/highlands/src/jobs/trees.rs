use engine_core::{Error, ErrorCode, Mesh, math};
use glam::Vec3;

use super::random::{golden_hash, unit};
use super::sprigs::{Atlas, DETAILS, Foliage, Sprig};
use super::wood::{End, Ring, Wood};
use super::{broadleaf, conifers, deadwood};

const HEADER: usize = 12;

const BASE: [f32; 5] = [-0.5, 0.0, 0.18, 0.4, 0.8];

const FLARE_HEIGHT: f32 = 0.35;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Spruce,
    Pine,
    Birch,
    Shrub,
    Snag,
    Log,
    Stump,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Age {
    Young,
    Grown,
    Old,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Piece {
    Foliage,
    Wood,
}

pub(crate) struct Plant {
    pub(crate) name: &'static str,
    kind: Kind,
    age: Age,
    seed: u32,
}

const fn plant(name: &'static str, kind: Kind, age: Age, seed: u32) -> Plant {
    Plant {
        name,
        kind,
        age,
        seed,
    }
}

pub(crate) const PLANTS: [Plant; 14] = [
    plant("spruce_young", Kind::Spruce, Age::Young, 11),
    plant("spruce_grown", Kind::Spruce, Age::Grown, 12),
    plant("spruce_tall", Kind::Spruce, Age::Grown, 13),
    plant("spruce_old", Kind::Spruce, Age::Old, 14),
    plant("pine_young", Kind::Pine, Age::Young, 21),
    plant("pine_grown", Kind::Pine, Age::Grown, 22),
    plant("pine_old", Kind::Pine, Age::Old, 23),
    plant("birch_young", Kind::Birch, Age::Young, 31),
    plant("birch_grown", Kind::Birch, Age::Grown, 32),
    plant("birch_old", Kind::Birch, Age::Old, 33),
    plant("shrub", Kind::Shrub, Age::Grown, 41),
    plant("snag", Kind::Snag, Age::Old, 51),
    plant("log", Kind::Log, Age::Old, 52),
    plant("stump", Kind::Stump, Age::Old, 53),
];

impl Plant {
    pub(crate) fn atlas(&self) -> Option<&'static str> {
        match self.kind {
            Kind::Spruce => Some("spruce_twigs"),
            Kind::Pine => Some("pine_twigs"),
            Kind::Birch | Kind::Shrub => Some("birch_leaves"),
            Kind::Snag | Kind::Log | Kind::Stump => None,
        }
    }

    pub(crate) fn pieces(&self) -> &'static [Piece] {
        match self.kind {
            Kind::Spruce | Kind::Pine | Kind::Birch => &[Piece::Foliage, Piece::Wood],
            Kind::Shrub => &[Piece::Foliage],
            Kind::Snag | Kind::Log | Kind::Stump => &[Piece::Wood],
        }
    }

    pub(crate) fn details(&self) -> usize {
        match self.kind {
            Kind::Spruce | Kind::Pine | Kind::Birch => DETAILS,
            Kind::Shrub | Kind::Snag | Kind::Log => 3,
            Kind::Stump => 2,
        }
    }

    pub(crate) fn mesh(&self, piece: Piece, detail: usize) -> String {
        let part = match self.pieces().first() == Some(&piece) {
            true => "",
            false => "_wood",
        };
        match detail {
            0 => format!("{}{part}", self.name),
            _ => format!("{}{part}_lod{detail}", self.name),
        }
    }
}

pub(crate) fn input(plant: usize, detail: usize, piece: Piece, atlas: Option<&Atlas>) -> Vec<u8> {
    let piece = match piece {
        Piece::Foliage => 0_u32,
        Piece::Wood => 1,
    };
    let mut input = Vec::new();
    for word in [plant as u32, detail as u32, piece] {
        input.extend_from_slice(&word.to_le_bytes());
    }
    if let Some(atlas) = atlas.filter(|_| piece == 0) {
        atlas.write(&mut input);
    }
    input
}

pub(crate) fn grow(input: &[u8]) -> Result<Mesh, Error> {
    let invalid = |field: &str| Error::new(ErrorCode::InvalidRecord, format!("$.plant.{field}"));
    let header = input
        .get(..HEADER)
        .ok_or_else(|| invalid("input").at_least(HEADER as u64).got(input.len()))?;
    let [index, detail, piece] = [0, 4, 8].map(|at| {
        u32::from_le_bytes([header[at], header[at + 1], header[at + 2], header[at + 3]]) as usize
    });
    let plant = PLANTS
        .get(index)
        .ok_or_else(|| invalid("plant").at_most(PLANTS.len() as u64 - 1).got(index))?;
    if detail >= plant.details() {
        return Err(invalid("detail")
            .at_most(plant.details() as u64 - 1)
            .got(detail));
    }
    let at = format!("$.plant.{}", plant.name);
    let mut build = match piece {
        0 => Build {
            foliage: Some(Foliage::new(&Atlas::read(&input[HEADER..], &at)?, detail)),
            wood: None,
            detail,
        },
        1 => Build {
            foliage: None,
            wood: Some(Wood::default()),
            detail,
        },
        other => return Err(invalid("piece").at_most(1).got(other)),
    };
    match plant.kind {
        Kind::Spruce => conifers::spruce(plant.age, plant.seed, &mut build),
        Kind::Pine => conifers::pine(plant.age, plant.seed, &mut build),
        Kind::Birch => broadleaf::birch(plant.age, plant.seed, &mut build),
        Kind::Shrub => broadleaf::shrub(plant.seed, &mut build),
        Kind::Snag => deadwood::snag(plant.seed, &mut build),
        Kind::Log => deadwood::log(plant.seed, &mut build),
        Kind::Stump => deadwood::stump(plant.seed, &mut build),
    }
    let mesh = match (build.foliage, build.wood) {
        (Some(foliage), _) => foliage.mesh(),
        (None, Some(wood)) => wood.mesh(),
        (None, None) => Err(invalid("piece")),
    };
    mesh.map_err(|error| error.within(&format!("{at}.details[{detail}]")))
}

pub(crate) struct Build {
    foliage: Option<Foliage>,
    wood: Option<Wood>,
    detail: usize,
}

impl Build {
    pub(crate) fn sprig(&mut self, sprig: &Sprig) {
        if let Some(foliage) = &mut self.foliage {
            foliage.sprig(sprig);
        }
    }

    pub(crate) fn tube(&mut self, rings: &[Ring], sides: [u32; DETAILS], ends: [End; 2]) {
        let Some(wood) = &mut self.wood else {
            return;
        };
        let last = rings.len().saturating_sub(1);
        let step = self.detail + 1;
        let kept: Vec<Ring> = rings
            .iter()
            .enumerate()
            .filter(|(index, _)| index % step == 0 || *index == last)
            .map(|(_, ring)| *ring)
            .collect();
        wood.tube(&kept, sides[self.detail], ends);
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Dice(u32);

impl Dice {
    pub(crate) fn new(seed: u32) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9))
    }

    pub(crate) fn roll(self, index: u32) -> f32 {
        unit(golden_hash(self.0 ^ index))
    }

    pub(crate) fn span(self, index: u32, [low, high]: [f32; 2]) -> f32 {
        low + (high - low) * self.roll(index)
    }

    pub(crate) fn pick(self, index: u32, count: u32) -> u32 {
        golden_hash(self.0 ^ index ^ 0x68bc_21eb) % count
    }
}

pub(crate) fn heading(angle: f32) -> Vec3 {
    let (sin, cos) = math::sin_cos(angle);
    Vec3::new(cos, 0.0, sin)
}

pub(crate) fn side_of(way: Vec3) -> Vec3 {
    Vec3::new(-way.z, 0.0, way.x).normalize_or(Vec3::X)
}

pub(crate) fn rolled(vector: Vec3, axis: Vec3, angle: f32) -> Vec3 {
    let (sin, cos) = math::sin_cos(angle);
    vector * cos + axis.cross(vector) * sin + axis * (axis.dot(vector) * (1.0 - cos))
}

pub(crate) fn blend(from: [f32; 3], to: [f32; 3], share: f32) -> [f32; 3] {
    let share = math::clamp(share, 0.0, 1.0);
    [0, 1, 2].map(|channel| from[channel] + (to[channel] - from[channel]) * share)
}

pub(crate) struct Stem {
    pub(crate) height: f32,
    pub(crate) girth: f32,
    pub(crate) taper: f32,
    pub(crate) flare: f32,
    pub(crate) lean: Vec3,
    pub(crate) bend: Vec3,
}

impl Stem {
    pub(crate) fn axis(&self, y: f32) -> Vec3 {
        let along = math::clamp(y / self.height, 0.0, 1.0);
        let sway = math::sin(std::f32::consts::PI * along);
        Vec3::new(0.0, y, 0.0) + self.lean * math::max(y, 0.0) + self.bend * sway
    }

    pub(crate) fn radius(&self, y: f32) -> f32 {
        let left = math::max(1.0 - y / self.height, 0.0);
        let flare = 1.0 + self.flare * math::exp(-math::max(y, 0.0) / FLARE_HEIGHT);
        self.girth * math::pow(left, self.taper) * flare + 0.012
    }

    pub(crate) fn rings(&self, count: u32, tint: impl Fn(f32) -> [f32; 3]) -> Vec<Ring> {
        let above = (1..=count).map(|ring| {
            let share = ring as f32 / count as f32;
            BASE[BASE.len() - 1] + (self.height - BASE[BASE.len() - 1]) * share
        });
        BASE.into_iter()
            .filter(|&y| y < self.height)
            .chain(above)
            .map(|y| Ring {
                at: self.axis(y),
                radius: self.radius(y),
                tint: tint(y),
            })
            .collect()
    }
}

pub(crate) struct Limb {
    pub(crate) start: Vec3,
    pub(crate) out: Vec3,
    pub(crate) length: f32,
    pub(crate) rise: f32,
    pub(crate) curl: f32,
}

impl Limb {
    pub(crate) fn at(&self, along: f32) -> Vec3 {
        let (sin, cos) = math::sin_cos(self.rise);
        self.start
            + self.out * (self.length * along * cos)
            + Vec3::Y * (self.length * (along * sin + self.curl * along * along))
    }

    pub(crate) fn way(&self, along: f32) -> Vec3 {
        let (sin, cos) = math::sin_cos(self.rise);
        (self.out * cos + Vec3::Y * (sin + 2.0 * self.curl * along)).normalize_or(self.out)
    }

    pub(crate) fn rings(&self, radius: [f32; 2], tint: [[f32; 3]; 2], count: u32) -> Vec<Ring> {
        (0..=count)
            .map(|ring| {
                let along = ring as f32 / count as f32;
                Ring {
                    at: self.at(along),
                    radius: radius[0] + (radius[1] - radius[0]) * along,
                    tint: blend(tint[0], tint[1], along),
                }
            })
            .collect()
    }
}

pub(crate) struct Stubs {
    pub(crate) count: u32,
    pub(crate) heights: [f32; 2],
    pub(crate) lengths: [f32; 2],
    pub(crate) tint: [f32; 3],
    pub(crate) key: u32,
}

pub(crate) fn stubs(build: &mut Build, dice: Dice, stem: &Stem, stubs: &Stubs) {
    for stub in 0..stubs.count {
        let key = stubs.key ^ stub.wrapping_mul(0x0100_0193);
        let y = dice.span(key ^ 1, stubs.heights);
        let out = heading(std::f32::consts::TAU * dice.roll(key ^ 2));
        let limb = Limb {
            start: stem.axis(y) + out * (0.6 * stem.radius(y)),
            out,
            length: dice.span(key ^ 3, stubs.lengths),
            rise: dice.span(key ^ 4, [-0.35, 0.25]),
            curl: -0.12,
        };
        let thick = math::clamp(0.14 * stem.radius(y) + 0.008, 0.012, 0.05);
        build.tube(
            &limb.rings([thick, 0.4 * thick], [stubs.tint; 2], 2),
            [4, 3, 0, 0],
            [End::Open, End::Broken],
        );
    }
}
