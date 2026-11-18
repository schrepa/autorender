mod body;
mod shapes;
mod tree;
mod world;

use glam::{Mat3, Vec3};
use serde::{Deserialize, Serialize};

pub use body::{Body, Landed, Moved, Player, Wish, capsule_of, step};
pub use world::Solids;
pub use world::Work;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Surface {
    Stone,
    Metal,
    Wood,
    Dirt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Solid {
    Triangle(u32),
    Block(u32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub distance: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub face_normal: Vec3,
    pub surface: Surface,
    pub solid: Solid,
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
    pub surface: Surface,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dropped {
    pub degenerate: u32,
    pub unlabelled: u32,
    pub out_of_range: u32,
}
