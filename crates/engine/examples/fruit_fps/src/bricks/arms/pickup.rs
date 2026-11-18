use engine::prelude::*;
use glam::Vec3;

use super::{Setup, ticks_per_second};
use crate::play::actors::{Actor, Alive, Body, Wish};
use crate::play::arms::{Arms, Drops};
use crate::play::events::arms::{ArmsEvents, PickupEvent};
use crate::play::events::round::RoundEvents;
use crate::rules::{Event, Phase};
use crate::weapons::{CombatNumbers, Gun, forward};

type Picking<'a> = (&'a Actor, &'a Wish, &'a Body, &'a mut Arms, Has<Alive>);

pub(super) fn pick(
    time: Res<FixedTime>,
    setup: Res<Setup>,
    round: Option<Res<RoundEvents>>,
    mut drops: ResMut<Drops>,
    mut events: ResMut<ArmsEvents>,
    mut actors: Query<Picking<'_>>,
) {
    let fresh_round = round.as_deref().is_some_and(|round| {
        round.events.iter().any(|event| {
            matches!(
                event,
                Event::Phase {
                    phase: Phase::Buy,
                    ..
                }
            )
        })
    });
    if fresh_round {
        drops.clear();
    }
    let Some(numbers) = setup.numbers.as_ref() else {
        return;
    };
    let Some(tps) = ticks_per_second(&time) else {
        return;
    };
    let mut order: Vec<_> = actors.iter_mut().collect();
    order.sort_by_key(|(actor, ..)| actor.slot);
    for (actor, wish, body, mut arms, alive) in order {
        let pressed = wish.interact_held && !arms.use_held;
        arms.use_held = wish.interact_held;
        arms.pickable = None;
        if !alive {
            continue;
        }
        let eye = body.eye;
        let look = forward(wish.yaw_radians, wish.pitch_radians);
        arms.pickable = reachable(&numbers.combat, &drops, eye, body.feet, look);
        let Some(id) = arms.pickable.filter(|_| pressed) else {
            continue;
        };
        let Some(found) = drops.take(id) else {
            continue;
        };
        let gun = Gun::with_ammo(found.weapon, found.magazine, found.reserve);
        let left = arms.loadout.pick_up(&numbers.weapons, gun, tps);
        if let Some(left) = &left {
            drops.push(body.feet, left);
        }
        arms.pickable = None;
        events.pickups.push(PickupEvent {
            slot: actor.slot,
            weapon: found.weapon,
            left: left.map(|gun| gun.kind()),
        });
    }
}

fn reachable(
    combat: &CombatNumbers,
    drops: &Drops,
    eye: Vec3,
    feet: Vec3,
    look: Vec3,
) -> Option<u32> {
    drops
        .items
        .iter()
        .filter_map(|item| {
            let flat = Vec3::new(item.position.x - feet.x, 0.0, item.position.z - feet.z);
            if flat.length() > combat.pickup_reach_m {
                return None;
            }
            let along = (item.position - eye).dot(look);
            let off = (item.position - (eye + look * along)).length();
            (along > 0.0 && off <= combat.pickup_aim_m).then_some((item.id, off))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
        .map(|(id, _)| id)
}
