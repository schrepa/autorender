#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod ground;
mod physics;
mod place_smokes;
mod solids;
mod water;

pub use ground::*;
pub use solids::*;
pub use water::{
    WaterBodies, WaterFace, WaterKind, WaterObstacle, WaterObstacles, WaterSample, WaterSurfaces,
    WaterWorker,
};

use engine_core::smoke::SmokeEntry;
use engine_core::{Absent, CpuList, CpuSlot, Face, FaceType, List, Place, Slot};

const fn cpu<T: 'static>() -> Face {
    Face {
        cpu: Some(FaceType::of::<T>()),
        gpu: None,
        wgsl: None,
        absent: Absent::Required,
    }
}

pub struct GroundSlot;

impl Slot for GroundSlot {
    const NAME: &'static str = "ground";
    const FACE: Face = cpu::<Ground>();
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_ground",
        brick: Self::NAME,
        gpu: false,
        run: place_smokes::ground,
    }];
}

impl CpuSlot for GroundSlot {
    type Cpu = Ground;
}

pub struct PhysicsSlot;

impl Slot for PhysicsSlot {
    const NAME: &'static str = "physics";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "slot_physics",
        brick: Self::NAME,
        gpu: false,
        run: physics::physics,
    }];
}

pub struct WaterBodiesList;

impl List for WaterBodiesList {
    const NAME: &'static str = "water_bodies";
    const FACE: Face = cpu::<WaterBodies>();
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "list_water_bodies",
        brick: Self::NAME,
        gpu: false,
        run: place_smokes::water_bodies,
    }];
}

impl CpuList for WaterBodiesList {
    type Cpu = WaterBodies;
}

pub struct WorldSolidsList;

impl List for WorldSolidsList {
    const NAME: &'static str = "world_solids";
    const FACE: Face = cpu::<WorldSolids>();
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "list_world_solids",
        brick: Self::NAME,
        gpu: false,
        run: solids::world_solids,
    }];
}

impl CpuList for WorldSolidsList {
    type Cpu = WorldSolids;
}

pub static PLACES: &[Place] = &[
    Place::slot::<GroundSlot>(),
    Place::slot::<PhysicsSlot>(),
    Place::list::<WaterBodiesList>(),
    Place::list::<WorldSolidsList>(),
];
