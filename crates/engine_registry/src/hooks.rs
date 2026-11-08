use std::any::TypeId;
use std::collections::VecDeque;

use bevy_ecs::change_detection::CheckChangeTicks;
use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{BoxedSystem, In, RunSystemError};
use bevy_ecs::world::{DeferredWorld, World};
use engine_place::{Error, ErrorCode};
use serde_json::Value;

use crate::state::{BrickStates, GateId, fault};

pub struct Removal {
    pub(crate) component: TypeId,
    pub(crate) name: &'static str,
    pub(crate) hook: fn(&mut World) -> bool,
    pub(crate) system: BoxedSystem<In<Entity>>,
}

impl Removal {
    pub fn of<C: Component>(system: BoxedSystem<In<Entity>>) -> Self {
        Self {
            component: TypeId::of::<C>(),
            name: std::any::type_name::<C>(),
            hook: hook_removal::<C>,
            system,
        }
    }
}

type FieldNames = Box<dyn Fn(&World) -> Vec<&'static str> + Send + Sync>;

pub struct NamedField {
    pub section: &'static str,
    pub field: &'static str,
    pub kept: &'static [&'static str],
    pub(crate) names: FieldNames,
}

impl NamedField {
    pub fn of(
        section: &'static str,
        field: &'static str,
        kept: &'static [&'static str],
        names: impl Fn(&World) -> Vec<&'static str> + Send + Sync + 'static,
    ) -> Self {
        Self {
            section,
            field,
            kept,
            names: Box::new(names),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Named {
    pub(crate) section: &'static str,
    pub(crate) field: &'static str,
    pub(crate) variants: Vec<&'static str>,
}

impl Named {
    pub(crate) fn check(&self, value: &Value, path: &str) -> Result<(), Error> {
        let Some(Value::String(got)) = value.get(self.field) else {
            return Ok(());
        };
        if self.variants.contains(&got.as_str()) {
            return Ok(());
        }
        Err(
            Error::new(ErrorCode::UnknownVariant, format!("{path}.{}", self.field))
                .got(got)
                .variants(self.variants.iter().copied()),
        )
    }

    pub(crate) fn schema(&self, schema: &mut schemars::Schema) {
        let pointer = format!("/properties/{}", self.field);
        if let Some(Value::Object(field)) = schema.pointer_mut(&pointer) {
            field.insert(
                "enum".to_owned(),
                Value::Array(
                    self.variants
                        .iter()
                        .map(|&variant| Value::String(variant.to_owned()))
                        .collect(),
                ),
            );
        }
    }
}

struct Hooked<S> {
    brick: &'static str,
    gate: GateId,
    system: S,
}

#[derive(Resource, Default)]
pub struct BrickHooks {
    hooked: Vec<TypeId>,
    removals: Vec<(TypeId, Hooked<BoxedSystem<In<Entity>>>)>,
}

impl BrickHooks {
    pub fn check_change_ticks(&mut self, check: CheckChangeTicks) {
        for (_, hooked) in &mut self.removals {
            hooked.system.check_change_tick(check);
        }
    }

    fn run(&mut self, world: &mut World, run: HookRun) {
        let HookRun::Removal(component, entity) = run;
        for (of, hooked) in &mut self.removals {
            if *of == component && on(world, hooked.gate) {
                let ran = hooked.system.run(entity, world);
                failed(world, hooked, "on_remove", ran);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum HookPhase {
    #[default]
    Idle,
    Running,
}

#[derive(Clone, Copy, Debug)]
enum HookRun {
    Removal(TypeId, Entity),
}

#[derive(Resource, Default)]
struct HookQueue {
    phase: HookPhase,
    queued: VecDeque<HookRun>,
}

pub(crate) fn add_removal(
    world: &mut World,
    brick: &'static str,
    gate: GateId,
    removal: Removal,
) -> Result<(), Error> {
    let hooked = world
        .get_resource_or_init::<BrickHooks>()
        .hooked
        .contains(&removal.component);
    if !hooked {
        if !(removal.hook)(world) {
            return Err(Error::new(
                ErrorCode::NameTwice,
                format!("$.bricks.{brick}.on_remove.{}", removal.name),
            )
            .got(removal.name));
        }
        world
            .resource_mut::<BrickHooks>()
            .hooked
            .push(removal.component);
    }
    world.resource_mut::<BrickHooks>().removals.push((
        removal.component,
        Hooked {
            brick,
            gate,
            system: removal.system,
        },
    ));
    Ok(())
}

fn hook_removal<C: Component>(world: &mut World) -> bool {
    let id = world.register_component::<C>();
    if world
        .archetypes()
        .iter()
        .any(|archetype| archetype.contains(id))
    {
        return false;
    }
    world
        .register_component_hooks::<C>()
        .try_on_remove(removed::<C>)
        .is_some()
}

fn removed<C: Component>(mut world: DeferredWorld, context: HookContext) {
    let entity = context.entity;
    world.commands().queue(move |world: &mut World| {
        hook_run(world, HookRun::Removal(TypeId::of::<C>(), entity));
    });
}

fn hook_run(world: &mut World, run: HookRun) {
    let mut queue = world.get_resource_or_init::<HookQueue>();
    queue.queued.push_back(run);
    if queue.phase == HookPhase::Running {
        return;
    }
    queue.phase = HookPhase::Running;
    loop {
        let next = world.resource_mut::<HookQueue>().queued.pop_front();
        let Some(next) = next else {
            break;
        };
        world.try_resource_scope::<BrickHooks, _>(|world, mut hooks| hooks.run(world, next));
    }
    world.resource_mut::<HookQueue>().phase = HookPhase::Idle;
}

fn on(world: &World, gate: GateId) -> bool {
    world
        .get_resource::<BrickStates>()
        .is_some_and(|states| states.on(gate))
}

fn failed<S>(world: &mut World, hooked: &Hooked<S>, at: &str, ran: Result<(), RunSystemError>) {
    let Err(error) = ran else {
        return;
    };
    let brick = world
        .get_resource::<BrickStates>()
        .and_then(|states| states.brick(hooked.gate));
    let refused = Error::new(
        ErrorCode::MissingResource,
        format!("$.bricks.{}.{at}", hooked.brick),
    )
    .got(error.to_string());
    if let Some(brick) = brick {
        fault(world, brick, refused);
    }
}
