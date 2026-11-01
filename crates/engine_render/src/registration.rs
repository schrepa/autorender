use std::any::{Any, TypeId};
use std::marker::PhantomData;

use bevy_ecs::system::SystemState;
use bevy_ecs::world::World;
use engine_core::{DrawPoint, Error, ErrorCode};
use engine_registry::{DrawMember, EncodeCore, FrameCore, InstallCore, RawDraw};

use crate::{Draw, Encode, GpuPart, Install, Pass, Prepare};

pub(crate) type InstallFn = fn(&mut World, &mut InstallCore) -> Result<Box<dyn Installed>, Error>;

pub(crate) struct PartRecord {
    pub(crate) part: TypeId,
    pub(crate) name: &'static str,
    pub(crate) install: InstallFn,
}

impl PartRecord {
    pub(crate) fn of<P: GpuPart>() -> Self {
        Self {
            part: TypeId::of::<P>(),
            name: std::any::type_name::<P>(),
            install: install::<P>,
        }
    }
}

type MakeMember<D> = Box<dyn Fn(u32) -> DrawMember<D> + Send + Sync>;

pub(crate) struct DrawRecord<D: DrawPoint> {
    pub(crate) part: TypeId,
    pub(crate) layer: D::Layer,
    make: MakeMember<D>,
}

impl<D: DrawPoint> DrawRecord<D> {
    pub(crate) fn of<P: GpuPart>(
        layer: D::Layer,
        draw: fn(&P, &D::Args, &mut Draw<'_, P, D>) -> u32,
    ) -> Self {
        Self {
            part: TypeId::of::<P>(),
            layer,
            make: Box::new(move |part| DrawMember {
                part,
                layer,
                call: Box::new(trampoline::<P, D>(draw)),
            }),
        }
    }

    pub(crate) fn member(&self, part: u32) -> DrawMember<D> {
        (self.make)(part)
    }
}

fn trampoline<P: GpuPart, D: DrawPoint>(
    draw: fn(&P, &D::Args, &mut Draw<'_, P, D>) -> u32,
) -> impl Fn(&dyn Any, &D::Args, &mut RawDraw<'_>) -> u32 + Send + Sync {
    move |part, args, raw| match part.downcast_ref::<P>() {
        Some(part) => draw(part, args, &mut Draw::retyped(raw)),
        None => 0,
    }
}

impl<P: GpuPart, S: 'static> Draw<'_, P, S> {
    pub(crate) fn retyped<'r>(raw: &'r mut RawDraw<'_>) -> Draw<'r, P, S> {
        todo!()
    }
}

pub(crate) trait Installed: Send + Sync {
    fn part(&self) -> &dyn Any;

    fn prepare(&mut self, world: &World, core: &mut FrameCore) -> Result<(), Error>;

    fn encode(&mut self, pass: u32, core: &mut EncodeCore) -> u32;

    fn refresh(&mut self, world: &mut World);
}

struct Held<P: GpuPart> {
    part: P,
    reads: SystemState<P::Reads>,
}

fn install<P: GpuPart>(world: &mut World, core: &mut InstallCore) -> Result<Box<dyn Installed>, Error> {
    let reads = SystemState::<P::Reads>::new(world);
    let part = P::install(&mut Install {
        core,
        part: PhantomData,
    })?;
    Ok(Box::new(Held { part, reads }))
}

impl<P: GpuPart> Installed for Held<P> {
    fn part(&self) -> &dyn Any {
        &self.part
    }

    fn prepare(&mut self, world: &World, core: &mut FrameCore) -> Result<(), Error> {
        let reads = self.reads.get(world).map_err(|_| {
            Error::new(
                ErrorCode::PlaceNotFilled,
                format!("$.parts.{}.reads", std::any::type_name::<P>()),
            )
        })?;
        self.part.prepare(
            reads,
            &mut Prepare {
                core,
                part: PhantomData,
            },
        );
        Ok(())
    }

    fn encode(&mut self, pass: u32, core: &mut EncodeCore) -> u32 {
        self.part.encode(
            Pass(pass, PhantomData),
            &mut Encode {
                core,
                part: PhantomData,
            },
        )
    }

    fn refresh(&mut self, world: &mut World) {
        self.reads = SystemState::new(world);
    }
}
