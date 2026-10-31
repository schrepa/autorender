#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine_core::{Absent, CpuSlot, Face, FaceType, List, Place, Slot};

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
    const FACE: Face = cpu::<engine_core::Ground>();
}

impl CpuSlot for GroundSlot {
    type Cpu = engine_core::Ground;
}

pub struct PhysicsSlot;

impl Slot for PhysicsSlot {
    const NAME: &'static str = "physics";
}

pub struct WaterBodiesList;

impl List for WaterBodiesList {
    const NAME: &'static str = "water_bodies";
    const FACE: Face = cpu::<engine_core::WaterBodies>();
}

pub struct WorldSolidsList;

impl List for WorldSolidsList {
    const NAME: &'static str = "world_solids";
    const FACE: Face = cpu::<engine_core::WorldSolids>();
}

pub static PLACES: &[Place] = &[
    Place::slot::<GroundSlot>(),
    Place::slot::<PhysicsSlot>(),
    Place::list::<WaterBodiesList>(),
    Place::list::<WorldSolidsList>(),
];
