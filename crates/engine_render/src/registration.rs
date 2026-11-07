use std::any::{Any, TypeId};
use std::marker::PhantomData;
use std::sync::Arc;

use bevy_ecs::system::SystemState;
use bevy_ecs::world::World;
use engine_core::{DrawKind, Error, ErrorCode};
use engine_registry::Scope;

use crate::draws::RawDraw;
use crate::encode::{EncodeCore, FrameCore};
use crate::install::InstallCore;
use crate::profiling::GpuProfiler;
use crate::{Draw, Encode, GpuPart, Install, Pass, Prepare, Scene, SceneDraw};

pub(crate) type InstallFn = fn(&mut World, &mut InstallCore) -> Result<Box<dyn Installed>, Error>;

pub(crate) struct PartRecord {
    pub(crate) part: TypeId,
    pub(crate) name: &'static str,
    pub(crate) lazy: bool,
    pub(crate) turns_on_in_play: bool,
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
            turns_on_in_play: P::TURNS_ON_IN_PLAY,
            labels: P::LABELS,
            scope,
            install: install::<P>,
        }
    }
}

pub(crate) struct Generated(pub(crate) &'static [&'static str]);

pub(crate) struct ResolvePass(pub(crate) crate::PassNames);

pub(crate) type KindCall =
    Arc<dyn Fn(&dyn Any, &SceneDraw, &mut RawDraw<'_>) -> u32 + Send + Sync>;

pub(crate) struct KindDraw {
    pub(crate) part: TypeId,
    pub(crate) kind: DrawKind,
    pub(crate) call: KindCall,
}

impl KindDraw {
    pub(crate) fn of<P: GpuPart>(
        kind: DrawKind,
        draw: fn(&P, &SceneDraw, &mut Draw<'_, P, Scene>) -> u32,
    ) -> Self {
        Self {
            part: TypeId::of::<P>(),
            kind,
            call: Arc::new(move |part: &dyn Any, scene: &SceneDraw, raw: &mut RawDraw<'_>| {
                match part.downcast_ref::<P>() {
                    Some(part) => draw(part, scene, &mut Draw::retyped(raw)),
                    None => 0,
                }
            }),
        }
    }
}

#[derive(Clone)]
pub(crate) struct KindMember {
    pub(crate) part: u32,
    pub(crate) kind: DrawKind,
    pub(crate) call: KindCall,
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

    fn rebind(&mut self, world: &mut World);
}

struct Held<P: GpuPart> {
    part: P,
    reads: SystemState<P::Reads>,
}

fn install<P: GpuPart>(
    world: &mut World,
    core: &mut InstallCore,
) -> Result<Box<dyn Installed>, Error> {
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

    fn rebind(&mut self, world: &mut World) {
        self.reads = SystemState::new(world);
    }
}
