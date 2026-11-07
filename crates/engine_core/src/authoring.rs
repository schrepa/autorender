use std::sync::LazyLock;

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_registry::{CORE_ROW, MembersOf, OperationScene};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    ComponentKind, Error, ErrorCode, Expected, GlobalTransforms, List, Mover, Node, Parent, Placed,
    Placement, SceneComponents, SceneContext, scene_component, scene_components,
};

pub(crate) const COMPONENT: &str = "component";

static CORE: LazyLock<[ComponentKind; 2]> = LazyLock::new(|| {
    let node = ComponentKind::of::<Node>("node");
    let mover = ComponentKind::of::<Mover>("mover");
    #[cfg(feature = "schema")]
    let (node, mover) = (node.schema(schema_of::<Node>), mover.schema(schema_of::<Mover>));
    [node.into(), mover.into()]
});

#[cfg(feature = "schema")]
fn schema_of<C: schemars::JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(C)).unwrap_or(Value::Null)
}

#[doc(hidden)]
pub fn core_components() -> &'static [ComponentKind] {
    &*CORE
}

pub(crate) fn keys_taken(world: &World) -> Vec<Error> {
    let Some(members) = world.get_resource::<MembersOf<SceneComponents>>() else {
        return Vec::new();
    };
    members
        .all()
        .iter()
        .zip(members.owners())
        .filter(|(kind, _)| core_components().iter().any(|core| core.key() == kind.key()))
        .map(|(kind, &owner)| {
            Error::new(
                ErrorCode::NameTwice,
                format!("$.places.{}.{}", SceneComponents::NAME, kind.key()),
            )
            .got(CORE_ROW)
            .nearest(owner)
        })
        .collect()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ComponentControl {
    entity: String,
    key: String,
    #[serde(default)]
    value: Value,
}

pub(crate) fn run_component(world: &mut World, request: Value) -> Result<Value, Error> {
    let ComponentControl { entity, key, value } = engine_settings::parse_value(request)?;
    let node = world
        .get_resource::<OperationScene>()
        .and_then(|scene| scene.entity(world, &entity))
        .ok_or_else(|| Error::new(ErrorCode::UnknownEntity, "$.entity").got(&entity))?;
    let kind = scene_component(world, &key).ok_or_else(|| {
        Error::new(ErrorCode::UnknownVariant, "$.key")
            .got(&key)
            .variants(scene_components(world).map(ComponentKind::key))
    })?;
    let context = world.get_resource::<SceneContext>().cloned();
    let set = || author(world, &kind, node, &value);
    match &context {
        Some(context) => context.scope(set),
        None => set(),
    }?;
    let value = kind.value(world, node).unwrap_or(Value::Null);
    Ok(json!({ "entity": entity, "key": key, "value": value }))
}

fn author(
    world: &mut World,
    kind: &ComponentKind,
    node: Entity,
    value: &Value,
) -> Result<(), Error> {
    if value.is_null() {
        kind.remove(world, node);
        return Ok(());
    }
    kind.validate(value, "$.value")?;
    match kind.placement() {
        Placement::Roots(code) if world.get::<Parent>(node).is_some() => {
            return Err(Error::new(code, "$.entity").expected(Expected::Absent));
        }
        Placement::Roots(_) | Placement::Anywhere | Placement::Entities(_) => {}
    }
    let placed = world
        .get_resource::<GlobalTransforms>()
        .and_then(|globals| globals.get(node))
        .and_then(|global| global.matrix());
    if let Some(matrix) = placed {
        kind.validate_placed(value, Placed { world: matrix }, "$.value")?;
    }
    kind.read(world, node, value)
        .map_err(|error| error.within("$.value"))
}
