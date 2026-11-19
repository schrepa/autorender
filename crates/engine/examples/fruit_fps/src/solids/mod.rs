mod body;
mod openings;
mod peers;
mod shapes;
mod tree;
mod world;

use glam::{Mat3, Vec3};
use serde::{Deserialize, Serialize};

pub use body::{Body, Landed, Moved, Player, Wish, capsule_of, step};
pub use openings::{Opening, Probe, openings};
pub use peers::apart;
pub use world::Solids;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Kind(pub u32);

impl Kind {
    pub const STONE: Kind = Kind::named("stone");

    pub const fn named(name: &str) -> Kind {
        let bytes = name.as_bytes();
        let mut hash: u32 = 0x811c_9dc5;
        let mut at = 0;
        while at < bytes.len() {
            hash ^= bytes[at] as u32;
            hash = hash.wrapping_mul(0x0100_0193);
            at += 1;
        }
        Kind(hash)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ground {
    pub kind: Kind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub distance: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub kind: Kind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capsule {
    pub feet: Vec3,
    pub height: f32,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Block {
    pub id: u32,
    pub centre: Vec3,
    pub half_extents: Vec3,
    pub axes: Mat3,
    pub kind: Kind,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dropped {
    pub degenerate: u32,
    pub unlabelled: u32,
    pub out_of_range: u32,
}
