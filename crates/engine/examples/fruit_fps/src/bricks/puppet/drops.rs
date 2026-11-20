use std::collections::BTreeMap;

use engine::prelude::*;
use engine_core::math::clamp;
use engine_core::{Assets, Error, ErrorCode, Material, MeshFile, Mobility, Refusals, Stats};
use glam::{Quat, Vec3};

use super::{DroppedWeapon, PuppetBrick, PuppetStats};
use crate::play::arms::{Dropped, Drops};
use crate::play::events::arms::Weapon;
use crate::play::figures::GROUND;
use crate::play::puppet::{Puppet, PuppetBook, PuppetFile};

const DROPPED_PART: &str = "puppet_dropped";

#[derive(Clone, Copy, Debug, PartialEq)]
struct Flight {
    from: Vec3,
    elapsed: f32,
}

#[derive(Resource, Default)]
pub(super) struct Flights {
    held: BTreeMap<u32, Flight>,
}

type Reads<'w, 's> = (
    (Res<'w, FixedTime>, Option<Res<'w, Drops>>),
    Res<'w, Puppet>,
    Query<'w, 's, Entity, With<PuppetFile>>,
    Query<'w, 's, (Entity, &'static DroppedWeapon)>,
);

type Writes<'w, 's> = (
    (Stats<'w, PuppetStats>, ResMut<'w, Flights>),
    ResMut<'w, Refusals>,
    Spawner<'w, 's, PuppetBrick>,
    Moves<'w, 's, PuppetBrick>,
);

pub(super) fn lying_at(book: &PuppetBook, position: Vec3, id: u32, share: f32) -> Transform {
    let look = book.table.dropped;
    let turn = look.turn_radians_per_id * (id % 64) as f32;
    Transform {
        translation: position + Vec3::new(0.0, look.lift_m, 0.0),
        rotation: Quat::from_rotation_y(turn) * Quat::from_rotation_z(look.roll_radians * share),
        scale: Vec3::ONE,
    }
}

fn flown(book: &PuppetBook, item: &Dropped, flight: Flight) -> Transform {
    let throw = book.table.throw;
    let share = if throw.seconds > 0.0 {
        clamp(flight.elapsed / throw.seconds, 0.0, 1.0)
    } else {
        1.0
    };
    let mut placed = lying_at(book, item.position, item.id, share);
    let lift = 4.0 * throw.lift_m * share * (1.0 - share);
    placed.translation = flight.from.lerp(placed.translation, share) + Vec3::new(0.0, lift, 0.0);
    placed
}

pub(super) fn drawn_of(assets: &Assets, name: &str) -> Result<Drawn, Error> {
    let mesh = assets.named::<MeshFile>(name);
    let material = assets.named::<Material>(name);
    let (Some(mesh), Some(material)) = (mesh, material) else {
        return Err(Error::new(ErrorCode::UnknownReference, "$.puppet").got(name));
    };
    Ok(Drawn::new(GROUND, mesh, material))
}

fn gun_drawn(book: &PuppetBook, key: &str) -> Result<Drawn, Error> {
    let look = book.weapons.get(key).ok_or_else(|| {
        Error::new(ErrorCode::UnknownReference, "$.puppet.weapons").got(key.to_owned())
    })?;
    Ok(Drawn::new(GROUND, look.files.held_mesh, look.files.held_material))
}

pub(super) fn ground_node() -> Node {
    Node {
        visible: true,
        mobility: Some(Mobility::Moving),
        ..Node::default()
    }
}

pub(super) fn lie(
    ((fixed, drops), puppet, files, lying_now): Reads,
    ((mut stats, mut flights), mut refusals, mut spawner, mut moves): Writes,
) {
    let (Some(owner), Some(book)) = (files.iter().next(), puppet.book.as_deref()) else {
        return;
    };
    let items = drops.as_deref().map_or(&[][..], |drops| &drops.items[..]);
    let existing: BTreeMap<u32, Entity> = lying_now
        .iter()
        .map(|(entity, dropped)| (dropped.id, entity))
        .collect();
    let mut refusal = None;
    for item in items {
        if let Some(entity) = existing.get(&item.id) {
            let Some(flight) = flights.held.get_mut(&item.id) else {
                continue;
            };
            flight.elapsed += fixed.delta_seconds;
            let placed = flown(book, item, *flight);
            if flight.elapsed >= book.table.throw.seconds {
                flights.held.remove(&item.id);
            }
            if let Some(mut transform) = moves.get_mut(*entity) {
                transform.set_if_neq(placed);
            }
            continue;
        }
        match gun_drawn(book, Weapon::Gun(item.weapon).key()) {
            Ok(drawn) => {
                let placed = match item.thrown_from {
                    Some(from) => {
                        let flight = Flight { from, elapsed: 0.0 };
                        flights.held.insert(item.id, flight);
                        flown(book, item, flight)
                    }
                    None => lying_at(book, item.position, item.id, 1.0),
                };
                let id = item.id;
                spawner.spawn((placed, ground_node(), drawn, DroppedWeapon { id }));
            }
            Err(error) => refusal = Some(error),
        }
    }
    for (id, entity) in &existing {
        if !items.iter().any(|item| item.id == *id) {
            spawner.despawn(*entity);
        }
    }
    flights
        .held
        .retain(|id, _| items.iter().any(|item| item.id == *id));
    let count = u32::try_from(items.len()).unwrap_or(u32::MAX);
    stats.edit(|stats| stats.dropped = count);
    refusals.judge(owner, DROPPED_PART, refusal.map_or(Ok(()), Err));
}
