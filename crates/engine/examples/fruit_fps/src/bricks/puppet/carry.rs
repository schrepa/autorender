use std::collections::BTreeMap;

use engine::prelude::*;
use engine_core::{Mobility, Refusals, Stats};

use engine_core::{GlobalTransforms, Parent};

use super::{CarriedStats, CarriedWeapon, Memories, PuppetBrick, PuppetStats};
use crate::play::figures::GROUND;
use crate::play::puppet::{ActorMuzzle, Muzzles, Puppet, Seen, Tasks};
use crate::play::skinned::{FigureClock, FigureJoints, Held, posed};

const HELD_PART: &str = "puppet_held";

type Reads<'w, 's> = (
    Res<'w, FixedTime>,
    Res<'w, Puppet>,
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
    Stats<'w, PuppetStats>,
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
    (fixed, puppet, memories): Reads,
    carried: Carried,
    ((mut muzzles, mut tasks), mut stats, mut refusals, mut spawner): Writes,
) {
    let mut busy: Vec<_> = memories
        .held
        .values()
        .filter_map(|memory| Some((memory.carry?.slot, memory.task()?)))
        .collect();
    busy.sort_unstable_by_key(|(slot, _)| *slot);
    if tasks.actors != busy {
        *tasks = Tasks {
            tick: fixed.tick,
            actors: busy,
        };
    }
    let Some(book) = puppet.book.as_deref() else {
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
        refusals.accept(carry.actor, HELD_PART);
        let drawn = Drawn::new(GROUND, look.files.held_mesh, look.files.held_material);
        let held = Held::on(carry.actor, carry.hand, carry.offset);
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
    if muzzles.actors != actors {
        *muzzles = Muzzles {
            tick: fixed.tick,
            actors,
        };
    }
    stats.edit(|stats| stats.held = shown);
}

type Checked<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        (&'static CarriedWeapon, &'static Held),
        &'static Node,
        Option<&'static Parent>,
    ),
>;

fn grip_gap(
    (weapon, held): (Entity, &Held),
    holder: Option<(&FigureJoints, Option<&FigureClock>)>,
    globals: &GlobalTransforms,
) -> Option<f32> {
    let (joints, clock) = holder?;
    let hand = joints.joints.get(posed::RIGHT_HAND).filter(|joint| joint.found)?;
    let matrix = globals.get(weapon)?.matrix()?;
    let offset = Quat::from_array(held.offset.rotation).normalize();
    let fit = clock.map_or(1.0, |clock| clock.fit);
    let grip = -(offset.inverse() * Vec3::from_array(held.offset.translation)) * fit;
    Some(matrix.transform_point3(grip).distance(hand.position))
}

pub(super) fn checked(
    (globals, seen): (Res<GlobalTransforms>, Option<Res<Seen>>),
    carried: Checked,
    holders: Query<(&FigureJoints, Option<&FigureClock>)>,
    mut stats: Stats<PuppetStats>,
) {
    let mut listed: Vec<CarriedStats> = carried
        .iter()
        .map(|(entity, (weapon, held), node, parent)| {
            let holder = held.figure;
            let parented = parent.map(Parent::parent) == Some(holder);
            let holder_shown = seen.as_deref().is_some_and(|seen| seen.shows(holder));
            CarriedStats {
                slot: weapon.slot,
                parented,
                shown: node.visible,
                holder_shown,
                drawn: parented && node.visible && holder_shown,
                grip_gap_m: grip_gap(
                    (entity, held),
                    holders.get(holder).ok(),
                    &globals,
                ),
            }
        })
        .collect();
    listed.sort_unstable_by_key(|row| row.slot);
    stats.edit(|stats| stats.carried = listed);
}
