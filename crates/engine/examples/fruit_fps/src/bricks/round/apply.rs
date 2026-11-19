use engine::prelude::*;

use super::RoundBook;
use crate::play::actors::{Actor, Alive, Body, Health, Motion, Wish};
use crate::play::arms::Arms;
use crate::play::events::round::RoundEvents;
use crate::play::numbers::{GameNumbers, Numbers};
use crate::play::round::Round;
use crate::play::world::{Arena, Spawn};
use crate::rules::{Event, PlayerId, Side, View};
use crate::weapons::{GunKind, Loadout, actor_rng};

fn mixed(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn shuffled_spawn(
    spawns: &[Spawn],
    seed: u64,
    round: u32,
    side: Side,
    spawn: usize,
) -> Option<Spawn> {
    if spawns.is_empty() {
        return None;
    }
    let side_bit = match side {
        Side::Attack => 0,
        Side::Defend => 1,
    };
    let mut state = mixed(seed ^ (u64::from(round) << 1 | side_bit));
    let mut order: Vec<usize> = (0..spawns.len()).collect();
    for i in (1..order.len()).rev() {
        state = mixed(state);
        let j = (state % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    spawns.get(order[spawn % order.len()]).copied()
}

fn gun_kind(view: &View<'_>, player: PlayerId) -> Option<GunKind> {
    let weapon = view.players.get(player)?.primary?;
    let key = &view.catalog.weapon(weapon)?.key;
    serde_json::from_value(serde_json::Value::String(key.clone())).ok()
}

fn place(numbers: &GameNumbers, spawn: Spawn, actor: &mut Actor, body: &mut Body, wish: &mut Wish) {
    *body = Body {
        feet: spawn.position,
        velocity: Vec3::ZERO,
        grounded: true,
        crouched: false,
        crouch: 0.0,
        eye: spawn.position + Vec3::Y * numbers.player.eye_height_m,
        motion: Motion::Standing,
        ground: body.ground,
        landed: None,
    };
    actor.yaw_radians = spawn.yaw_radians;
    wish.yaw_radians = spawn.yaw_radians;
    wish.pitch_radians = 0.0;
    wish.direction = Vec3::ZERO;
}

type ActorParts<'a> = (
    &'a mut Actor,
    &'a mut Body,
    &'a mut Wish,
    &'a mut Health,
    Option<&'a mut Arms>,
);

#[expect(
    clippy::too_many_arguments,
    reason = "applying rule events touches every actor part"
)]
pub(super) fn apply(
    time: Res<FixedTime>,
    numbers: Res<Numbers>,
    arena: Res<Arena>,
    round: Res<Round>,
    events: Res<RoundEvents>,
    mut book: ResMut<RoundBook>,
    mut actors: Query<ActorParts<'_>>,
    mut commands: Commands,
) {
    let (Some(numbers), Some(view)) = (numbers.get(), round.view()) else {
        return;
    };
    let entity_of = |player: PlayerId| round.seats.get(player).and_then(|seat| seat.entity);
    let mut rearm: Vec<PlayerId> = Vec::new();
    for event in &events.events {
        match *event {
            Event::Respawn {
                player,
                side,
                spawn,
                health,
            } => {
                let Some(entity) = entity_of(player) else {
                    continue;
                };
                let Ok((mut actor, mut body, mut wish, mut vitals, _)) = actors.get_mut(entity)
                else {
                    continue;
                };
                match shuffled_spawn(arena.spawns(side), round.seed, view.round, side, spawn) {
                    Some(at) => place(numbers, at, &mut actor, &mut body, &mut wish),
                    None => book.unplaced += 1,
                }
                let armour = view
                    .players
                    .get(player)
                    .and_then(|p| p.armor)
                    .map_or(0, |a| a.points);
                *vitals = Health {
                    health: health as f32,
                    armour: armour as f32,
                };
                commands.entity(entity).insert(Alive);
                rearm.push(player);
            }
            Event::Armor { player, armor } => {
                let Some(entity) = entity_of(player) else {
                    continue;
                };
                if let Ok((_, _, _, mut vitals, _)) = actors.get_mut(entity) {
                    vitals.armour = armor.map_or(0, |a| a.points) as f32;
                }
            }
            Event::GiveWeapon { player, .. } | Event::TakeWeapon { player, .. } => {
                rearm.push(player)
            }
            Event::Phase { .. }
            | Event::Charges { .. }
            | Event::Credits { .. }
            | Event::BuyRefused { .. }
            | Event::AbilityUsed { .. }
            | Event::AbilityRefused { .. }
            | Event::DeviceGiven { .. }
            | Event::DeviceDropped { .. }
            | Event::DevicePickedUp { .. }
            | Event::PlantStarted { .. }
            | Event::PlantCancelled { .. }
            | Event::PlantCompleted { .. }
            | Event::DefuseStarted { .. }
            | Event::DefuseHalf { .. }
            | Event::DefuseCancelled { .. }
            | Event::DefuseCompleted { .. }
            | Event::DeviceExploded { .. }
            | Event::KillFeed { .. }
            | Event::UltimatePoint { .. }
            | Event::UltimateReady { .. }
            | Event::RoundEnd { .. }
            | Event::SideSwap { .. }
            | Event::MatchEnd { .. } => {}
        }
    }
    rearm.sort_unstable();
    rearm.dedup();
    let tps = time.rate;
    for player in rearm {
        let Some(entity) = entity_of(player) else {
            continue;
        };
        let Ok((_, _, _, _, Some(mut arms))) = actors.get_mut(entity) else {
            continue;
        };
        let rng = if events.tick == 1 {
            actor_rng(round.seed, player as u32)
        } else {
            arms.loadout.rng()
        };
        let kind = gun_kind(&view, player);
        let held = arms.loadout.primary.as_ref().map(crate::weapons::Gun::kind);
        let respawned = events
            .events
            .iter()
            .any(|e| matches!(e, Event::Respawn { player: p, .. } if *p == player));
        if respawned || held != kind {
            arms.loadout = Loadout::new(&numbers.weapons, kind, rng, tps);
            let unarmed = view
                .players
                .get(player)
                .is_some_and(|p| p.sidearm.is_none());
            if unarmed {
                arms.loadout.take_sidearm(&numbers.weapons, tps);
            }
        } else if arms.loadout.sidearm.is_none()
            && view
                .players
                .get(player)
                .is_some_and(|p| p.sidearm.is_some())
        {
            arms.loadout
                .give(&numbers.weapons, crate::weapons::GunKind::Sidearm, tps);
        }
    }
}
