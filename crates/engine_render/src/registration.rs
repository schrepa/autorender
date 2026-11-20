use std::any::{Any, TypeId};
use std::marker::PhantomData;
use std::sync::Arc;

use bevy_ecs::system::SystemState;
use bevy_ecs::world::World;
use engine_core::{DrawKind, Error, ErrorCode, KindDecl};

use crate::draws::RawDraw;
use crate::drawn::Batches;
use crate::encode::{EncodeCore, FrameCore};
use crate::install::InstallCore;
use crate::profiling::GpuProfiler;
use crate::{Draw, Encode, GpuPart, Install, Pass, Prepare, Scene};

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
            name: if P::NAME.is_empty() {
                std::any::type_name::<P>()
            } else {
                P::NAME
            },
            install: install::<P>,
        }
    }
}

pub(crate) struct Generated(pub(crate) &'static [&'static str]);

pub(crate) type KindCall = Arc<dyn Fn(&dyn Any, &mut RawDraw<'_>) + Send + Sync>;

pub(crate) struct KindDraw {
    pub(crate) part: TypeId,
    pub(crate) kind: DrawKind,
    pub(crate) call: KindCall,
}

impl KindDraw {
    pub(crate) fn of<P: GpuPart>(
        kind: DrawKind,
        draw: fn(&P, &mut Draw<'_, P, Scene>),
    ) -> Self {
        Self {
            part: TypeId::of::<P>(),
            kind,
            call: Arc::new(move |part: &dyn Any, raw: &mut RawDraw<'_>| {
                if let Some(part) = part.downcast_ref::<P>() {
                    draw(part, &mut Draw::retyped(raw));
                }
            }),
        }
    }
}

#[derive(Clone)]
pub(crate) struct KindMember {
    pub(crate) part: u32,
    pub(crate) kind: u32,
    pub(crate) decl: KindDecl,
    pub(crate) call: KindCall,
    pub(crate) batches: Batches,
}

impl KindMember {
    pub(crate) fn new(part: u32, (kind, decl): (u32, KindDecl), call: KindCall) -> Self {
        Self {
            part,
            kind,
            decl,
            call,
            batches: Batches::default(),
        }
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

    fn view(&mut self, view: &engine_core::ViewFacts, core: &mut FrameCore);

    fn encode(&mut self, pass: u32, core: &mut EncodeCore, timer: &GpuProfiler);

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

    fn view(&mut self, view: &engine_core::ViewFacts, core: &mut FrameCore) {
        self.part.view(
            view,
            &mut Prepare {
                core,
                part: PhantomData,
            },
        );
    }

    fn encode(&mut self, pass: u32, core: &mut EncodeCore, timer: &GpuProfiler) {
        self.part.encode(
            Pass(pass, PhantomData),
            &mut Encode {
                core,
                timer,
                part: PhantomData,
            },
        );
    }

    fn rebind(&mut self, world: &mut World) {
        self.reads = SystemState::new(world);
    }
}

#[cfg(feature = "dev")]
pub(crate) fn kind_passes(
    declared: &crate::install::Declared,
    members: &[KindMember],
) -> serde_json::Value {
    let mut kinds = serde_json::Map::new();
    for member in members {
        let listed = kinds
            .entry(member.decl.kind.name())
            .or_insert_with(|| serde_json::json!([]));
        let serde_json::Value::Array(passes) = listed else {
            continue;
        };
        for pass in &declared.passes {
            let joins = pass.members.contains(&member.part)
                && pass
                    .scene
                    .is_some_and(|scene| member.decl.joins(scene.name));
            if joins && !passes.iter().any(|known| known == pass.name) {
                passes.push(serde_json::json!(pass.name));
            }
        }
    }
    serde_json::Value::Object(kinds)
}
