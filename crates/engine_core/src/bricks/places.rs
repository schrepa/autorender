use std::any::{Any, TypeId};
use std::marker::PhantomData;

use bevy_ecs::prelude::{Resource, World};

use super::{Bricks, List, Place, Slot};

#[derive(Clone, Copy, Debug)]
pub struct FaceType {
    pub id: fn() -> TypeId,
    pub name: fn() -> &'static str,
}

impl FaceType {
    pub const fn of<T: ?Sized + 'static>() -> Self {
        Self {
            id: TypeId::of::<T>,
            name: std::any::type_name::<T>,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Absent {
    Required,
    Nobody,
}

impl Absent {
    pub fn name(self) -> &'static str {
        match self {
            Self::Required => "required",
            Self::Nobody => "nobody",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Face {
    pub cpu: Option<FaceType>,
    pub gpu: Option<FaceType>,
    pub wgsl: Option<&'static str>,
    pub absent: Absent,
}

impl Face {
    pub const NONE: Self = Self {
        cpu: None,
        gpu: None,
        wgsl: None,
        absent: Absent::Required,
    };
}

pub trait CpuSlot: Slot {
    type Cpu: Resource;
}

pub trait CpuList: List {
    type Cpu: Members;
}

pub trait Members:
    Resource + bevy_ecs::component::Component<Mutability = bevy_ecs::component::Mutable> + Default
{
    type Member: Send + Sync + 'static;

    fn add(&mut self, brick: &'static str, member: Self::Member);
}

#[derive(Clone, Copy, Debug)]
pub struct FaceOffer {
    pub face: fn() -> TypeId,
    pub type_name: fn() -> &'static str,
    pub cast: fn() -> Box<dyn Any + Send + Sync>,
}

impl FaceOffer {
    pub fn name(&self) -> &'static str {
        face_name((self.type_name)())
    }
}

pub fn face_name(type_name: &'static str) -> &'static str {
    let plain = type_name.strip_prefix("dyn ").unwrap_or(type_name);
    let plain = plain.split('<').next().unwrap_or(plain);
    plain.rsplit("::").next().unwrap_or(plain)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Member {
    pub module: &'static str,
    pub functions: &'static [&'static str],
    pub reads: &'static [&'static str],
}

impl Member {
    pub fn implements(&self, function: &str) -> bool {
        self.functions.contains(&function)
    }
}

type Insert = Box<dyn FnOnce(&mut World, &'static str) + Send>;

pub struct Fill<S: Slot> {
    cpu: Option<Insert>,
    gpu: Option<FaceOffer>,
    wgsl: Option<&'static str>,
    _place: PhantomData<fn() -> S>,
}

impl<S: Slot> Default for Fill<S> {
    fn default() -> Self {
        Self {
            cpu: None,
            gpu: None,
            wgsl: None,
            _place: PhantomData,
        }
    }
}

impl<S: Slot> Fill<S> {
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn gpu(mut self, offer: FaceOffer) -> Self {
        self.gpu = Some(offer);
        self
    }

    #[must_use]
    pub fn wgsl(mut self, module: &'static str) -> Self {
        self.wgsl = Some(module);
        self
    }
}

impl<S: CpuSlot> Fill<S> {
    #[must_use]
    pub fn cpu(mut self, value: S::Cpu) -> Self {
        self.cpu = Some(Box::new(move |world: &mut World, _: &'static str| {
            world.insert_resource(value);
        }));
        self
    }
}

pub struct Join<L: List> {
    cpu: Option<Insert>,
    wgsl: Option<Member>,
    _place: PhantomData<fn() -> L>,
}

impl<L: List> Default for Join<L> {
    fn default() -> Self {
        Self {
            cpu: None,
            wgsl: None,
            _place: PhantomData,
        }
    }
}

impl<L: List> Join<L> {
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn wgsl(mut self, member: Member) -> Self {
        self.wgsl = Some(member);
        self
    }
}

impl<L: CpuList> Join<L> {
    #[must_use]
    pub fn cpu(mut self, member: <L::Cpu as Members>::Member) -> Self {
        self.cpu = Some(Box::new(move |world: &mut World, brick: &'static str| {
            if !world.contains_resource::<L::Cpu>() {
                world.init_resource::<L::Cpu>();
            }
            world.resource_mut::<L::Cpu>().add(brick, member);
        }));
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Filled {
    pub place: &'static str,
    pub cpu: bool,
    pub gpu: Option<FaceOffer>,
    pub wgsl: Option<&'static str>,
}

#[derive(Clone, Copy, Debug)]
pub struct Joined {
    pub place: &'static str,
    pub cpu: bool,
    pub wgsl: Option<Member>,
}

impl Bricks<'_> {
    pub fn fill<S: Slot>(&mut self, fill: Fill<S>) -> &mut Self {
        let Fill { cpu, gpu, wgsl, .. } = fill;
        let filled = Filled {
            place: S::NAME,
            cpu: cpu.is_some(),
            gpu,
            wgsl,
        };
        if let Some(insert) = cpu {
            insert(self.app.world_mut(), self.record.name);
        }
        self.record.fill(Place::slot::<S>(), filled);
        self
    }

    pub fn join<L: List>(&mut self, join: Join<L>) -> &mut Self {
        let Join { cpu, wgsl, .. } = join;
        let joined = Joined {
            place: L::NAME,
            cpu: cpu.is_some(),
            wgsl,
        };
        if let Some(insert) = cpu {
            insert(self.app.world_mut(), self.record.name);
        }
        self.record.join(Place::list::<L>(), joined);
        self
    }

    pub fn places(&mut self, places: &'static [Place]) -> &mut Self {
        for place in places {
            self.record.know(*place);
        }
        self
    }
}
