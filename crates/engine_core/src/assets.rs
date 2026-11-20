use bevy_ecs::world::World;
use engine_registry::{MembersOf, Registry};

use crate::{AssetKind, Assets, List};

pub struct AssetKinds;

impl List for AssetKinds {
    const NAME: &'static str = "asset_kinds";
    type Member = AssetKind;
}

pub(crate) fn install_kinds(world: &mut World) {
    let kinds: Vec<AssetKind> = world
        .get_resource::<MembersOf<AssetKinds>>()
        .map(|members| members.all().to_vec())
        .unwrap_or_default();
    let mut problems = Vec::new();
    let mut assets = world.resource_mut::<Assets>();
    for kind in kinds {
        if let Err(problem) = assets.kind(kind) {
            problems.push(problem);
        }
    }
    if let Some(mut registry) = world.get_resource_mut::<Registry>() {
        for problem in problems {
            registry.report(problem);
        }
    }
}
