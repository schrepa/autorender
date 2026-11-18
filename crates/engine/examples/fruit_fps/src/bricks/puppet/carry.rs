use std::collections::BTreeMap;

use engine::prelude::*;
use engine_core::{Error, ErrorCode, Mobility, Refusals, SceneContext};

use super::{CarriedWeapon, Memories, PuppetBrick, PuppetStats};
use crate::names::RIGHT_HAND;
use crate::play::figures::{GROUND, Held};
use crate::play::puppet::{ActorMuzzle, Muzzles, PuppetFile, Tasks};

const HELD_PART: &str = "puppet_held";

type Reads<'w, 's> = (
    Res<'w, FixedTime>,
    Option<Res<'w, SceneContext>>,
    Query<'w, 's, &'static PuppetFile>,
    Res<'w, Memories>,
);

type Carried<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static CarriedWeapon,
        &'static Drawn,
        &'static Held,
        &'static Node,
    ),
>;

type Writes<'w, 's> = (
    (ResMut<'w, Muzzles>, ResMut<'w, Tasks>),
    ResMut<'w, PuppetStats>,
    ResMut<'w, Refusals>,
    Spawner<'w, 's, PuppetBrick>,
);

fn moving(visible: bool) -> Node {
    Node {
        visible,
        mobility: Some(Mobility::Moving),
        ..Node::default()
    }
}

pub(super) fn carry(
    (fixed, scene, files, memories): Reads,
    carried: Carried,
    ((mut muzzles, mut tasks), mut stats, mut refusals, mut spawner): Writes,
) {
    let mut busy: Vec<_> = memories
        .held
        .values()
        .filter_map(|memory| Some((memory.carry?.slot, memory.task()?)))
        .collect();
    busy.sort_unstable_by_key(|(slot, _)| *slot);
    tasks.set_if_neq(Tasks {
        tick: fixed.tick,
        actors: busy,
    });
    let Some(book) = files.iter().find_map(PuppetFile::book) else {
        return;
    };
    let mut existing: BTreeMap<u32, (Entity, &Drawn, &Held, &Node)> = BTreeMap::new();
    for (entity, weapon, drawn, held, node) in &carried {
        existing.insert(weapon.slot, (entity, drawn, held, node));
    }
    let mut actors = Vec::new();
    let mut shown = 0;
    let mut slots = Vec::new();
    let mut owners = Vec::new();
    for memory in memories.held.values() {
        let Some(carry) = memory.carry else {
            continue;
        };
        slots.push(carry.slot);
        owners.push(carry.actor);
        let found = existing.get(&carry.slot).copied();
        let look = book.weapons.get(carry.weapon);
        let Some(look) = look else {
            if let Some((entity, _, _, node)) = found
                && node.visible
            {
                spawner.insert(entity, moving(false));
            }
            refusals.accept(carry.actor, HELD_PART);
            continue;
        };
        if let Some((position, direction)) = carry.muzzle {
            actors.push(ActorMuzzle {
                slot: carry.slot,
                weapon: carry.weapon,
                position,
                direction,
            });
        }
        let assets = scene.as_deref().map(|scene| &*scene.assets);
        let mesh = assets.and_then(|assets| assets.meshes.get(&look.files.held_mesh).copied());
        let material =
            assets.and_then(|assets| assets.materials.get(&look.files.held_material).copied());
        let (Some(mesh), Some(material)) = (mesh, material) else {
            refusals.refuse(
                carry.actor,
                HELD_PART,
                Error::new(ErrorCode::UnknownReference, "$.puppet.weapons").got(format!(
                    "{} / {}",
                    look.files.held_mesh, look.files.held_material
                )),
            );
            continue;
        };
        refusals.accept(carry.actor, HELD_PART);
        let drawn = Drawn::new(GROUND, mesh, material);
        let held = Held::on(carry.actor, RIGHT_HAND, carry.offset);
        if carry.shown {
            shown += 1;
        }
        match found {
            None => {
                spawner.spawn((
                    Transform::default(),
                    moving(carry.shown),
                    drawn,
                    held,
                    CarriedWeapon { slot: carry.slot },
                ));
            }
            Some((entity, was_drawn, was_held, node)) => {
                if was_drawn.geometry != drawn.geometry || was_drawn.material != drawn.material {
                    spawner.insert(entity, drawn);
                }
                if *was_held != held {
                    spawner.insert(entity, held);
                }
                if node.visible != carry.shown {
                    spawner.insert(entity, moving(carry.shown));
                }
            }
        }
    }
    for (slot, (entity, ..)) in &existing {
        if !slots.contains(slot) {
            spawner.despawn(*entity);
        }
    }
    refusals.keep_part(HELD_PART, |entity| owners.contains(&entity));
    actors.sort_unstable_by_key(|muzzle| muzzle.slot);
    let fresh = Muzzles {
        tick: fixed.tick,
        actors,
    };
    muzzles.set_if_neq(fresh);
    if stats.held != shown {
        stats.held = shown;
    }
}
