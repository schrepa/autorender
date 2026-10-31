use std::any::Any;
use std::marker::PhantomData;

use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{SystemBuffer, SystemMeta};
use bevy_ecs::world::{DeferredWorld, World};
use engine_place::{DrawPoint, Error, List};

pub struct BrickRecord {
    pub package: &'static str,
    pub entries: Vec<Box<dyn Any + Send + Sync>>,
    pub render: Vec<Box<dyn Any + Send + Sync>>,
}

pub trait RecordExt {
    fn record(&mut self) -> &mut BrickRecord;
}

#[derive(Resource)]
pub struct MembersOf<L: List> {
    members: Vec<L::Member>,
    live: Vec<bool>,
}

impl<L: List> MembersOf<L> {
    pub fn live(&self) -> impl Iterator<Item = &L::Member> + '_ {
        self.members.iter().zip(&self.live).filter(|(_, live)| **live).map(|(member, _)| member)
    }
}

pub struct DrawMember<D: DrawPoint> {
    pub part: u32,
    pub layer: D::Layer,
    pub call: Box<dyn Fn(&dyn Any, &D::Args, &mut RawDraw<'_>) -> u32 + Send + Sync>,
}

#[derive(Default)]
pub struct InstallCore {
    pub part: u32,
}

#[derive(Default)]
pub struct FrameCore {
    pub frame: u64,
}

#[derive(Default)]
pub struct EncodeCore {
    pub pass: u32,
}

pub struct RawDraw<'e> {
    pub pass: PhantomData<&'e mut ()>,
}

#[derive(Default)]
pub struct JobList {
    pub names: Vec<&'static str>,
}

pub trait Shares: Send + Sync {
    fn share(&self, key: [u8; 32]) -> Option<&[u8]>;
}

#[derive(Resource, Default)]
pub struct JobListed {
    pub hash: [u8; 32],
}

#[derive(Default)]
pub struct JobOrders {
    pub orders: Vec<([u8; 32], u32)>,
}

impl SystemBuffer for JobOrders {
    fn queue(&mut self, _: &SystemMeta, _: DeferredWorld) {}
}

#[derive(Default)]
pub struct RecordStore;

#[derive(Clone, Copy)]
pub struct SceneRead(pub fn(&mut World, Entity, &serde_json::Value) -> Result<(), Error>);

#[derive(Default)]
pub struct ViewState;

#[derive(Default)]
pub struct PoseTable;

#[derive(Default)]
pub struct Residency;

#[derive(Default)]
pub struct InputState;
