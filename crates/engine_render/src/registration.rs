use std::any::{Any, TypeId};
use std::marker::PhantomData;
use std::sync::Arc;

use bevy_ecs::system::SystemState;
use bevy_ecs::world::World;
use engine_core::{DrawPoint, Error, ErrorCode};
use engine_registry::Scope;

use crate::draws::{DrawMember, RawDraw};
use crate::encode::{EncodeCore, FrameCore};
use crate::install::InstallCore;
use crate::profiling::GpuProfiler;
use crate::{Draw, Encode, GpuPart, Install, Pass, Prepare};

pub(crate) type InstallFn = fn(&mut World, &mut InstallCore) -> Result<Box<dyn Installed>, Error>;

pub(crate) struct PartRecord {
    pub(crate) part: TypeId,
    pub(crate) name: &'static str,
    pub(crate) lazy: bool,
    pub(crate) labels: &'static [&'static str],
    pub(crate) scope: Scope,
    pub(crate) install: InstallFn,
}

impl PartRecord {
    pub(crate) fn of<P: GpuPart>(scope: Scope) -> Self {
        Self {
            part: TypeId::of::<P>(),
            name: if P::NAME.is_empty() {
                std::any::type_name::<P>()
            } else {
                P::NAME
            },
            lazy: P::LAZY,
            labels: P::LABELS,
            scope,
            install: install::<P>,
        }
    }
}

pub(crate) struct Generated(pub(crate) &'static [&'static str]);

pub(crate) struct ResolvePass(pub(crate) crate::PassNames);

pub(crate) struct DrawAny {
    pub(crate) part: TypeId,
    pub(crate) record: Arc<dyn Any + Send + Sync>,
}

impl DrawAny {
    pub(crate) fn of<P: GpuPart, D: DrawPoint>(
        layer: D::Layer,
        draw: fn(&P, &D::Args, &mut Draw<'_, P, D>) -> u32,
    ) -> Self {
        Self {
            part: TypeId::of::<P>(),
            record: Arc::new(DrawRecord::<D> {
                layer,
                make: Box::new(move |part| DrawMember {
                    part,
                    layer,
                    call: Box::new(trampoline::<P, D>(draw)),
                }),
            }),
        }
    }
}

type MakeMember<D> = Box<dyn Fn(u32) -> DrawMember<D> + Send + Sync>;

pub(crate) struct DrawRecord<D: DrawPoint> {
    pub(crate) layer: D::Layer,
    make: MakeMember<D>,
}

impl<D: DrawPoint> DrawRecord<D> {
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
        Draw {
            raw: raw.reborrow(),
            part: PhantomData,
        }
    }
}

pub(crate) trait Installed: Send + Sync {
    fn part(&self) -> &dyn Any;

    fn prepare(&mut self, world: &World, core: &mut FrameCore) -> Result<(), Error>;

    fn encode(&mut self, pass: u32, core: &mut EncodeCore, timer: &GpuProfiler) -> u32;

    fn refresh(&mut self, world: &mut World);
}

struct Held<P: GpuPart> {
    part: P,
    reads: SystemState<P::Reads>,
    fresh: bool,
}

fn install<P: GpuPart>(world: &mut World, core: &mut InstallCore) -> Result<Box<dyn Installed>, Error> {
    let reads = SystemState::<P::Reads>::new(world);
    let part = P::install(&mut Install {
        core,
        part: PhantomData,
    })?;
    Ok(Box::new(Held {
        part,
        reads,
        fresh: true,
    }))
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
                fresh: self.fresh,
                part: PhantomData,
            },
        );
        self.fresh = false;
        Ok(())
    }

    fn encode(&mut self, pass: u32, core: &mut EncodeCore, timer: &GpuProfiler) -> u32 {
        self.part.encode(
            Pass(pass, PhantomData),
            &mut Encode {
                core,
                timer,
                part: PhantomData,
            },
        )
    }

    fn refresh(&mut self, world: &mut World) {
        self.reads = SystemState::new(world);
        self.fresh = true;
    }
}
