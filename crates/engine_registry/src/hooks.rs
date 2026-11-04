use std::any::TypeId;

use bevy_ecs::change_detection::CheckChangeTicks;
use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{BoxedSystem, In, RunSystemError};
use bevy_ecs::world::{DeferredWorld, World};
use engine_place::{Error, ErrorCode};
use serde_json::Value;

use crate::gates::{GateId, Gates};
use crate::registry::Registry;

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

pub struct Settle {
    pub(crate) operation: &'static str,
    pub(crate) system: BoxedSystem,
}

impl Settle {
    pub fn of(operation: &'static str, system: BoxedSystem) -> Self {
        Self { operation, system }
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
    system: Option<S>,
}

#[derive(Resource, Default)]
pub struct BrickHooks {
    hooked: Vec<TypeId>,
    removals: Vec<(TypeId, Hooked<BoxedSystem<In<Entity>>>)>,
    settles: Vec<(&'static str, Hooked<BoxedSystem>)>,
}

impl BrickHooks {
    pub fn check_change_ticks(&mut self, check: CheckChangeTicks) {
        for (_, hooked) in &mut self.removals {
            if let Some(system) = &mut hooked.system {
                system.check_change_tick(check);
            }
        }
        for (_, hooked) in &mut self.settles {
            if let Some(system) = &mut hooked.system {
                system.check_change_tick(check);
            }
        }
    }
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
            system: Some(removal.system),
        },
    ));
    Ok(())
}

pub(crate) fn add_settle(world: &mut World, brick: &'static str, gate: GateId, settle: Settle) {
    world.get_resource_or_init::<BrickHooks>().settles.push((
        settle.operation,
        Hooked {
            brick,
            gate,
            system: Some(settle.system),
        },
    ));
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
        run_removals(world, TypeId::of::<C>(), entity);
    });
}

fn run_removals(world: &mut World, component: TypeId, entity: Entity) {
    let count = world
        .get_resource::<BrickHooks>()
        .map_or(0, |hooks| hooks.removals.len());
    for index in 0..count {
        let taken = world
            .get_resource_mut::<BrickHooks>()
            .and_then(|mut hooks| {
                let (of, hooked) = hooks.removals.get_mut(index)?;
                if *of != component {
                    return None;
                }
                Some((hooked.brick, hooked.system.take()?))
            });
        let Some((brick, mut system)) = taken else {
            continue;
        };
        let ran = system.run(entity, world);
        if let Some(mut hooks) = world.get_resource_mut::<BrickHooks>()
            && let Some((_, hooked)) = hooks.removals.get_mut(index)
        {
            hooked.system = Some(system);
        }
        if let Err(error) = ran {
            fault(world, brick, "on_remove", &error);
        }
    }
}

pub fn run_settles(world: &mut World, operation: &str) {
    let count = world
        .get_resource::<BrickHooks>()
        .map_or(0, |hooks| hooks.settles.len());
    for index in 0..count {
        let gate = world
            .get_resource::<BrickHooks>()
            .and_then(|hooks| hooks.settles.get(index))
            .filter(|(of, _)| *of == operation)
            .map(|(_, hooked)| hooked.gate);
        let Some(gate) = gate else {
            continue;
        };
        if !world
            .get_resource::<Gates>()
            .is_some_and(|gates| gates.on(gate))
        {
            continue;
        }
        let taken = world
            .get_resource_mut::<BrickHooks>()
            .and_then(|mut hooks| {
                let (_, hooked) = hooks.settles.get_mut(index)?;
                Some((hooked.brick, hooked.system.take()?))
            });
        let Some((brick, mut system)) = taken else {
            continue;
        };
        let ran = system.run((), world);
        if let Some(mut hooks) = world.get_resource_mut::<BrickHooks>()
            && let Some((_, hooked)) = hooks.settles.get_mut(index)
        {
            hooked.system = Some(system);
        }
        if let Err(error) = ran {
            fault(world, brick, operation, &error);
        }
    }
}

fn fault(world: &mut World, brick: &str, at: &str, error: &RunSystemError) {
    if let Some(mut registry) = world.get_resource_mut::<Registry>() {
        registry.report(
            Error::new(ErrorCode::MissingResource, format!("$.bricks.{brick}.{at}"))
                .got(error.to_string()),
        );
    }
}
