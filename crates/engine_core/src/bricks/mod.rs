mod attribution;
mod scene;

#[doc(hidden)]
pub use attribution::under;
pub use attribution::{
    BrickCost, BrickCosts, BrickRow, BrickTable, CORE_ROW, GPU_IDLE_ROW, UNATTRIBUTED_ROW,
    WARMUP_FRAMES,
};
pub use scene::{AuthoredId, OperationScene, authored_entity, authored_id};

use std::marker::PhantomData;

use bevy_ecs::bundle::Bundle;
use bevy_ecs::component::{Immutable, Mutable};
use bevy_ecs::entity::Entity;
use bevy_ecs::query::{QueryFilter, ReadOnlyQueryData};
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_ecs::system::{
    Commands, IntoSystem, Query, ReadOnlySystemParam, Res, ScheduleSystem, SystemParam,
    SystemParamFunction,
};

use crate::{
    GpuStruct, JobKind, List, Module, Operation, PlaceDecl, Section, Slot, Smoke, Source,
};

pub trait Brick: 'static {
    const SOURCE: &'static Source;
    const JOBS: &'static [&'static JobKind] = &[];
    const SMOKES: &'static [Smoke] = &[];
    fn register(bricks: &mut Bricks<'_>);
}

pub struct Bricks<'r> {
    record: &'r mut engine_registry::BrickRecord,
}

impl engine_registry::RecordExt for Bricks<'_> {
    fn record(&mut self) -> &mut engine_registry::BrickRecord {
        &mut *self.record
    }
}

impl<'r> Bricks<'r> {
    #[doc(hidden)]
    pub fn recording(record: &'r mut engine_registry::BrickRecord) -> Self {
        Self { record }
    }
}

impl Bricks<'_> {
    pub fn when<M, F>(&mut self, gate: F) -> &mut Self
    where
        F: SystemParamFunction<M, In = (), Out = bool>,
        F::Param: Read,
    {
        todo!()
    }

    pub fn within<M, F>(&mut self, gate: F, items: impl FnOnce(&mut Self)) -> &mut Self
    where
        F: SystemParamFunction<M, In = (), Out = bool>,
        F::Param: Read,
    {
        todo!()
    }

    pub fn off<M>(
        &mut self,
        system: impl IntoSystem<(), (), M> + Send + Sync + 'static,
    ) -> &mut Self {
        todo!()
    }

    pub fn fill<S: Slot>(&mut self, wgsl: Option<Module>) -> &mut Self {
        todo!()
    }

    pub fn join<L: List>(&mut self, member: L::Member, wgsl: Option<Module>) -> &mut Self {
        todo!()
    }

    pub fn systems<M>(
        &mut self,
        schedule: Schedule,
        systems: impl IntoScheduleConfigs<ScheduleSystem, M>,
    ) -> &mut Self {
        todo!()
    }

    pub fn resource<R: Resource + Default>(&mut self) -> &mut Self {
        todo!()
    }

    pub fn settings<S: Section + Resource<Mutability = Immutable>>(&mut self) -> &mut Self {
        todo!()
    }

    pub fn stats<S: Section + Resource<Mutability = Mutable>>(&mut self) -> &mut Self {
        todo!()
    }

    pub fn piece<T: Resource<Mutability = Mutable> + Default + GpuStruct>(&mut self) -> &mut Self {
        todo!()
    }

    pub fn operation<O: Operation>(&mut self) -> &mut Self {
        todo!()
    }

    pub fn observation<O: Operation>(&mut self) -> &mut Self
    where
        O::Params: Read,
    {
        todo!()
    }
}

pub trait Listing: Sized {
    #[must_use]
    fn with<B: Brick>(self) -> Self;
    #[must_use]
    fn without<B: Brick>(self) -> Self;
    #[must_use]
    fn places(self, places: &'static [PlaceDecl]) -> Self;
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct Listed {
    source: &'static Source,
    jobs: &'static [&'static JobKind],
    smokes: &'static [Smoke],
    register: fn(&mut Bricks<'_>),
}

impl Listed {
    fn of<B: Brick>() -> Self {
        Self {
            source: B::SOURCE,
            jobs: B::JOBS,
            smokes: B::SMOKES,
            register: B::register,
        }
    }

    pub fn source(&self) -> &'static Source {
        self.source
    }

    pub fn package(&self) -> &'static str {
        self.source.package()
    }

    pub fn jobs(&self) -> &'static [&'static JobKind] {
        self.jobs
    }

    pub fn smokes(&self) -> &'static [Smoke] {
        self.smokes
    }

    pub fn register(&self, bricks: &mut Bricks<'_>) {
        (self.register)(bricks);
    }
}

#[doc(hidden)]
#[derive(Clone, Default)]
pub struct BrickList {
    bricks: Vec<Listed>,
    places: Vec<&'static [PlaceDecl]>,
}

impl BrickList {
    pub fn listed(&self) -> &[Listed] {
        &self.bricks
    }

    pub fn declared(&self) -> impl Iterator<Item = &'static PlaceDecl> + '_ {
        self.places.iter().flat_map(|places| places.iter())
    }

    pub fn contains(&self, package: &str) -> bool {
        self.bricks
            .iter()
            .any(|listed| listed.package() == package)
    }
}

impl Listing for BrickList {
    fn with<B: Brick>(mut self) -> Self {
        self.bricks.push(Listed::of::<B>());
        self
    }

    fn without<B: Brick>(mut self) -> Self {
        self.bricks
            .retain(|listed| !std::ptr::eq(listed.source, B::SOURCE));
        self
    }

    fn places(mut self, places: &'static [PlaceDecl]) -> Self {
        self.places.push(places);
        self
    }
}

mod sealed {
    pub trait Sealed {}
}

pub trait Read: ReadOnlySystemParam + sealed::Sealed {}

impl<T: Resource> sealed::Sealed for Res<'_, T> {}
impl<T: Resource> Read for Res<'_, T> {}
impl<D: ReadOnlyQueryData + 'static, F: QueryFilter + 'static> sealed::Sealed
    for Query<'_, '_, D, F>
{
}
impl<D: ReadOnlyQueryData + 'static, F: QueryFilter + 'static> Read for Query<'_, '_, D, F> {}
impl<L: List> sealed::Sealed for Joined<'_, L> {}
impl<L: List> Read for Joined<'_, L> {}

macro_rules! read_tuples {
    ($($param:ident),*) => {
        impl<$($param: Read),*> sealed::Sealed for ($($param,)*) {}
        impl<$($param: Read),*> Read for ($($param,)*) {}
    };
}

read_tuples!();
read_tuples!(P0);
read_tuples!(P0, P1);
read_tuples!(P0, P1, P2);
read_tuples!(P0, P1, P2, P3);
read_tuples!(P0, P1, P2, P3, P4);
read_tuples!(P0, P1, P2, P3, P4, P5);
read_tuples!(P0, P1, P2, P3, P4, P5, P6);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15);

#[derive(SystemParam)]
pub struct Joined<'w, L: List> {
    members: Res<'w, engine_registry::MembersOf<L>>,
}

impl<L: List> Joined<'_, L> {
    pub fn iter(&self) -> impl Iterator<Item = &L::Member> + '_ {
        self.members.live()
    }
}

#[derive(SystemParam)]
pub struct Spawner<'w, 's, B: Brick> {
    commands: Commands<'w, 's>,
    brick: PhantomData<fn() -> B>,
}

impl<B: Brick> Spawner<'_, '_, B> {
    pub fn spawn(&mut self, bundle: impl Bundle) -> Entity {
        todo!()
    }

    pub fn insert(&mut self, entity: Entity, bundle: impl Bundle) {
        todo!()
    }

    pub fn remove<T: Bundle>(&mut self, entity: Entity) {
        todo!()
    }

    pub fn despawn(&mut self, entity: Entity) {
        todo!()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Schedule {
    FixedPreSimulation,
    FixedMove,
    FixedSimulation,
    FixedPostSimulation,
    Presentation,
    WorldSync,
    ViewSync,
    BetweenFrames,
}
