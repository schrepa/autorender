use std::any::Any;
use std::sync::Arc;

use bevy_ecs::prelude::{Component, Entity, Resource, World};
use engine_settings::Error;
use glam::{Mat4, Vec2, Vec3};

use crate::{Pyramid, RecordKey, Slot, TileKey, WaterBodies, WaterSample};

pub struct GroundSlot;

impl Slot for GroundSlot {
    const NAME: &'static str = "ground";
}

pub const BRIDGE_SOLIDS: u16 = u16::MAX;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundPoint {
    pub height: f32,
    pub normal: Vec3,
    pub level: u32,
    pub hole: bool,
    pub material: Option<u8>,
}

pub trait GroundFace: Send + Sync + 'static {
    fn point(&self, world: &World, x: f32, z: f32) -> Option<GroundPoint>;

    fn map(&self, world: &World, name: &str, x: f32, z: f32) -> Option<f32>;

    fn water(&self, world: &World, point: Vec3) -> Option<WaterSample>;

    fn revision(&self, world: &World) -> u64;

    fn patches(&self, world: &World) -> Vec<GroundPatch>;

    fn solids(&self, world: &World) -> Vec<GroundSolid>;

    fn obstacles(&self, world: &World, ground: Entity) -> Vec<GroundObstacle>;

    fn source(&self, world: &World, from: Vec2, to: Vec2) -> GroundRead<GroundSource>;

    fn heightfield(&self) -> bool;
}

pub trait GroundSamples: Send + Sync + 'static {
    fn height(&self, x: i32, z: i32) -> f32;

    fn hole(&self, x: u32, z: u32) -> bool;

    fn same(&self, other: &dyn GroundSamples) -> bool;

    fn as_any(&self) -> &dyn Any;
}

#[derive(Clone)]
pub struct GroundPatch {
    pub ground: Entity,
    pub key: TileKey,
    pub origin: Vec3,
    pub spacing: f32,
    pub bounds: [f32; 2],
    pub samples: Arc<dyn GroundSamples>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GroundShape {
    Cuboid { half_extents: Vec3 },
    Sphere { radius: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundSolid {
    pub ground: Entity,
    pub layer: u16,
    pub asset: u16,
    pub shape: GroundShape,
    pub model: Mat4,
    pub memberships: u32,
    pub filter: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundObstacle {
    pub center: Vec2,
    pub radius: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GroundRead<T> {
    Ready(T),
    Waiting,
    Off,
}

#[derive(Clone)]
pub struct GroundSource {
    pub ground: Entity,
    pub origin: Vec3,
    pub pyramid: Pyramid,
    pub identity: Arc<[u8]>,
    pub regions: Arc<dyn GroundRegions>,
}

impl PartialEq for GroundSource {
    fn eq(&self, other: &Self) -> bool {
        self.ground == other.ground
            && self.origin == other.origin
            && self.pyramid == other.pyramid
            && self.identity == other.identity
    }
}

impl std::fmt::Debug for GroundSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GroundSource")
            .field("ground", &self.ground)
            .field("origin", &self.origin)
            .field("pyramid", &self.pyramid)
            .field("identity", &self.identity.len())
            .finish_non_exhaustive()
    }
}

pub trait GroundRegions: Send + Sync + 'static {
    fn record(&self, key: TileKey) -> GroundRecord;

    fn grid(&self, key: TileKey, output: &[u8]) -> Result<GroundGrid, Error>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroundRecord {
    pub kind: &'static str,
    pub input: Vec<u8>,
    pub key: RecordKey,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroundGrid {
    pub key: TileKey,
    pub spacing: f32,
    pub heights: Box<[f32]>,
    pub water: Option<Box<[f32]>>,
    pub holes: Option<Box<[u64]>>,
    pub bridges: Arc<[GroundBridge]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroundBridge {
    pub deck: Vec<Vec3>,
    pub width: f32,
    pub tunnel: bool,
}

pub trait GroundWorker {
    fn grid(&self, identity: &[u8], key: TileKey) -> Result<GroundGrid, Error>;
}

#[derive(Resource, Clone)]
pub struct Ground {
    brick: &'static str,
    face: Arc<dyn GroundFace>,
}

impl Ground {
    pub fn new(brick: &'static str, face: impl GroundFace) -> Self {
        Self {
            brick,
            face: Arc::new(face),
        }
    }

    pub fn brick(&self) -> &'static str {
        self.brick
    }

    pub fn face(&self) -> &dyn GroundFace {
        &*self.face
    }

    pub fn point_in(world: &World, x: f32, z: f32) -> Option<GroundPoint> {
        world.get_resource::<Self>()?.face.point(world, x, z)
    }

    pub fn sample_in(world: &World, x: f32, z: f32) -> Option<GroundPoint> {
        Self::point_in(world, x, z).filter(|point| !point.hole)
    }

    pub fn revision_in(world: &World) -> u64 {
        world
            .get_resource::<Self>()
            .map_or(0, |ground| ground.face.revision(world))
    }

    pub fn water_in(world: &World, point: Vec3) -> Option<WaterSample> {
        world
            .get_resource::<Self>()
            .and_then(|ground| ground.face.water(world, point))
            .or_else(|| WaterBodies::water_in(world, point))
    }

    pub fn map_in(world: &World, name: &str, x: f32, z: f32) -> Option<f32> {
        world.get_resource::<Self>()?.face.map(world, name, x, z)
    }
}

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeightHeld(pub bool);

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamingFocus;
