use std::collections::BTreeMap;

use engine::prelude::*;
use engine_core::{Error, ErrorCode, Mobility, Refusals, SceneContext};
use glam::{Quat, Vec3};

use super::{DroppedWeapon, PuppetBrick, PuppetStats};
use crate::play::arms::{Dropped, Drops};
use crate::play::events::arms::Weapon;
use crate::play::figures::GROUND;
use crate::play::puppet::{PuppetBook, PuppetFile};

const DROPPED_PART: &str = "puppet_dropped";

type Reads<'w, 's> = (
    Option<Res<'w, Drops>>,
    Option<Res<'w, SceneContext>>,
    Query<'w, 's, (Entity, &'static PuppetFile)>,
    Query<'w, 's, (Entity, &'static DroppedWeapon)>,
);

type Writes<'w, 's> = (
    ResMut<'w, PuppetStats>,
    ResMut<'w, Refusals>,
    Spawner<'w, 's, PuppetBrick>,
);

fn lying(book: &PuppetBook, item: &Dropped) -> Transform {
    let look = book.table.dropped;
    let turn = look.turn_radians_per_id * (item.id % 64) as f32;
    Transform {
        translation: item.position + Vec3::new(0.0, look.lift_m, 0.0),
        rotation: Quat::from_rotation_y(turn) * Quat::from_rotation_z(look.roll_radians),
        scale: Vec3::ONE,
    }
}

fn drawn_of(book: &PuppetBook, scene: Option<&SceneContext>, key: &str) -> Result<Drawn, Error> {
    let at = "$.puppet.weapons";
    let look = book
        .weapons
        .get(key)
        .ok_or_else(|| Error::new(ErrorCode::UnknownReference, at).got(key.to_owned()))?;
    let assets = scene.map(|scene| &*scene.assets);
    let mesh = assets.and_then(|assets| assets.meshes.get(&look.files.held_mesh).copied());
    let material =
        assets.and_then(|assets| assets.materials.get(&look.files.held_material).copied());
    let (Some(mesh), Some(material)) = (mesh, material) else {
        return Err(Error::new(ErrorCode::UnknownReference, at).got(format!(
            "{} / {}",
            look.files.held_mesh, look.files.held_material
        )));
    };
    Ok(Drawn::new(GROUND, mesh, material))
}

pub(super) fn lie(
    (drops, scene, files, lying_now): Reads,
    (mut stats, mut refusals, mut spawner): Writes,
) {
    let Some((owner, book)) = files
        .iter()
        .find_map(|(entity, file)| file.book().map(|book| (entity, book)))
    else {
        return;
    };
    let items = drops.as_deref().map_or(&[][..], |drops| &drops.items[..]);
    let mut existing: BTreeMap<u32, Entity> = BTreeMap::new();
    for (entity, dropped) in &lying_now {
        existing.insert(dropped.id, entity);
    }
    let mut refusal = None;
    for item in items {
        if existing.contains_key(&item.id) {
            continue;
        }
        match drawn_of(book, scene.as_deref(), Weapon::Gun(item.weapon).key()) {
            Ok(drawn) => {
                spawner.spawn((
                    lying(book, item),
                    Node {
                        visible: true,
                        mobility: Some(Mobility::Moving),
                        ..Node::default()
                    },
                    drawn,
                    DroppedWeapon { id: item.id },
                ));
            }
            Err(error) => refusal = Some(error),
        }
    }
    for (id, entity) in &existing {
        if !items.iter().any(|item| item.id == *id) {
            spawner.despawn(*entity);
        }
    }
    let count = u32::try_from(items.len()).unwrap_or(u32::MAX);
    if stats.dropped != count {
        stats.dropped = count;
    }
    refusals.judge(owner, DROPPED_PART, refusal.map_or(Ok(()), Err));
}
